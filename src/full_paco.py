"""
PACO: per-pixel caching PACO variant with optional GPU pre-processing.

Computes background statistics on demand for each rotated patch position
and caches them to avoid redundant computation across test pixels that
share common rotated positions.
"""
import multiprocessing
import time
import warnings
from multiprocessing import shared_memory
from typing import Optional, Tuple

import numpy as np
import psutil
import tqdm
from vip_hci.preproc.rescaling import frame_shift

from .gpu import GPU_AVAILABLE, GPUFrameShifter
from .math_utils import al, bl, compute_statistics_at_pixel, wfl
from .paco_base import PACOBase, deprecated_alias
from .utils import create_boolean_circular_mask, extract_patch_window, get_rotated_pixel_coords

# ---------------------------------------------------------------------------
# Module-level worker state and functions for multiprocessing.
#
# History of this design (see git-less changelog below — kept here since
# each step only made sense after measuring the previous one):
#
# 1. Original: dispatched a *bound method* (self._process_pixel_for_calc) to
#    pool.map(). Pickling a bound method pickles the whole instance `self`,
#    so every task in every chunk re-serialised the full cube. Cache was a
#    cross-process multiprocessing.Manager().dict() storing the full
#    (apatch, m, Cinv) tuple per key — correct, but every hit/miss is an IPC
#    round-trip to the Manager's server process for potentially large arrays.
#
# 2. Module-level functions + a plain per-worker-LOCAL cache dict (fork
#    copy-on-write, no cross-process sharing). Removed the self-pickling
#    cost, but PACO's whole design leans on *many* different candidate
#    trajectories revisiting the *same* background pixels — losing that
#    sharing made every worker redundantly recompute shared pixels. Measured
#    regression: 0.74x-1.00x across nproc=2..6, worse at higher nproc.
#
# 3. Module-level functions + a shared multiprocessing.Manager().dict() cache
#    (hits shared again) storing the full arrays. Removed self-pickling,
#    kept sharing. Measured 1.02x-1.06x — a real but small gain: the Manager
#    IPC round-trip for every access (now the dominant cost) wasn't touched.
#
# 4. This version: real shared memory for the bulk array data (apatch/m/Cinv
#    live in a fixed-capacity pool of raw SharedMemory buffers, read/written
#    as plain memory accesses — no pickling, no IPC per read). Only a small
#    key->slot_index map stays in a Manager().dict() (cheap: it holds ints,
#    not arrays), guarded by a fork-inherited Lock for the (rare, post
#    warm-up) cache-miss path. Since the Pool is created with the "fork"
#    context *after* these SharedMemory-backed arrays exist in the parent,
#    forked workers share the same underlying mmap (MAP_SHARED survives
#    fork, unlike regular copy-on-write heap memory) — no name-based
#    re-attachment needed, writes in one worker are immediately visible to
#    the others and to the parent.
#
# Capacity is bounded (see _estimate_cache_capacity): PACO exists
# specifically for cases where precomputing statistics for *every* pixel
# (as PACOBase.compute_statistics_parallel does) is too much memory (e.g. NIRC2,
# K=197: an H*W*K*K array would be ~322 GB). Once the pool is full, further
# misses are computed but not cached (correct, just uncached) rather than
# growing without bound.
# ---------------------------------------------------------------------------
_G: dict = {}

# Share of the run's RAM budget (PACOBase.memory_fraction of available RAM)
# given to the shared statistics cache; the sweep batches get what the
# cache actually leaves. 0.3 reproduces the previous default split
# (cache 0.25 / batches 0.65 of available RAM, i.e. ~0.28 of their sum).
CACHE_BUDGET_SHARE = 0.3


def _g_get_patch(p0: Tuple[int, int]) -> np.ndarray:
    """Module-level equivalent of PACOBase.get_patch, reading globals. See
    utils.extract_patch_window for why this is a small windowed extraction
    rather than a full-frame-mask boolean index."""
    return extract_patch_window(_G["cube"], p0, _G["patch_width"], _G["patch_area"])


def _get_or_compute_stats(row_rot: int, col_rot: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Look up (apatch, m, Cinv) for a background pixel, computing and caching
    it on a miss.

    The key->slot index is itself a shared-memory array `slot_map` of shape
    (H, W) (int32, -1 = uncached) — NOT a Manager().dict(). Measured: even
    after moving the *value* arrays to shared memory, keeping the key->slot
    lookup in a Manager().dict() still cost one IPC round-trip per (pixel,
    frame) pair — i.e. on every access, hit or miss, since a dict .get() is
    itself an IPC call — which is far more frequent than the per-unique-
    pixel misses it was meant to gate, and is what actually dominated the
    cost (measured regression: 0.37x-0.76x). Background-pixel positions are
    bounded by the image size regardless of patch size K, so the *index*
    (unlike the value pool, which must stay capacity-limited for large K —
    see module docstring) can safely cover the whole image at negligible
    memory cost (H*W*4 bytes, e.g. ~410 KB for a 321x321 NIRC2 frame).
    Reading it is then a plain, lock-free shared-memory access — no IPC at
    all on the hot (repeat-visit) path. Only claiming a new slot on a miss
    takes the (fork-inherited, not Manager-based) lock.

    Falls back to a plain local dict with no locking for cpu==1
    (`_G["lock"] is None` in that case; see PACOCalc).
    """
    lock = _G["lock"]

    if lock is None:
        # Serial path: no multiprocessing, no shared pool — a plain local
        # cache is simplest and correct. Bounded by the same capacity as
        # the shared pool: once full, misses are computed but not stored.
        cache = _G["local_cache"]
        key = (row_rot, col_rot)
        cached = cache.get(key)
        if cached is not None:
            return cached
        _G["miss_counter"][0] += 1
        apatch = _g_get_patch((row_rot, col_rot))
        m, Cinv = compute_statistics_at_pixel(apatch)
        m_safe = m if m is not None else np.full(apatch.shape[1], np.nan)
        Cinv_safe = Cinv if Cinv is not None else np.full(
            (apatch.shape[1], apatch.shape[1]), np.nan)
        if len(cache) < _G["capacity"]:
            cache[key] = (apatch, m_safe, Cinv_safe)
        return apatch, m_safe, Cinv_safe

    slot_map = _G["slot_map"]
    slot = slot_map[row_rot, col_rot]
    if slot >= 0:
        return (_G["apatch_pool"][slot], _G["m_pool"][slot], _G["cinv_pool"][slot])

    # Compute OUTSIDE the lock: this is the expensive part (patch gather +
    # covariance + shrinkage + K x K inversion) and is pure/stateless, so it
    # is safe (and essential for parallelism) to let multiple workers do it
    # concurrently for different misses. Holding the lock across this
    # computation was measured to fully serialise the miss path across all
    # workers regardless of nproc (flat ~37s at both nproc=2 and nproc=4 on
    # a workload with ~11k unique misses) — the lock must protect only the
    # short slot-claim + memcpy below, not the computation itself. Two
    # workers racing on the *same* miss will occasionally both do this
    # redundant work (resolved by the second re-check inside the lock,
    # exactly as before) — cheap compared to serialising everyone. Both are
    # counted in total_misses (diagnostic only, see PACOCalc's
    # last_cache_stats) since both really did perform the expensive
    # computation — that redundancy is real wasted work, and a fair hit-rate
    # metric should reflect it.
    with _G["total_misses"].get_lock():
        _G["total_misses"].value += 1
    apatch = _g_get_patch((row_rot, col_rot))
    m, Cinv = compute_statistics_at_pixel(apatch)
    K = apatch.shape[1]
    m_safe = m if m is not None else np.full(K, np.nan)
    Cinv_safe = Cinv if Cinv is not None else np.full((K, K), np.nan)

    with lock:
        slot = slot_map[row_rot, col_rot]
        if slot >= 0:
            # Someone else already cached it while we were computing —
            # discard our redundant result and use theirs.
            return (_G["apatch_pool"][slot], _G["m_pool"][slot], _G["cinv_pool"][slot])

        next_slot = _G["next_slot"]
        cur = next_slot.value
        if cur >= _G["capacity"]:
            # Pool exhausted: still correct, just uncached from here on.
            return apatch, m_safe, Cinv_safe
        next_slot.value = cur + 1
        _G["apatch_pool"][cur] = apatch
        _G["m_pool"][cur] = m_safe
        _G["cinv_pool"][cur] = Cinv_safe
        slot_map[row_rot, col_rot] = cur
        return apatch, m_safe, Cinv_safe


def _process_pixel_worker(args_tuple: tuple) -> Tuple[int, float, float]:
    """
    Compute a and b for a single test pixel using the shared statistics
    cache. Logically identical to the previous _process_pixel_for_calc,
    reading fork-inherited globals instead of `self`.

    Parameters
    ----------
    args_tuple : tuple
        (pixel_index, p0_xy, precomputed_shifts_for_this_pixel).

    Returns
    -------
    tuple
        (pixel_index, a_val, b_val). Returns (index, nan, nan) on error.
    """
    i, p0, precomputed_pixel_data = args_tuple

    try:
        angles_px = get_rotated_pixel_coords(
            _G["grid_x"], _G["grid_y"], p0, _G["angles"])

        col_coords_rot = angles_px[:, 0]
        row_coords_rot = angles_px[:, 1]
        width, height = _G["width"], _G["height"]
        if (np.any(col_coords_rot < 0) or np.any(col_coords_rot >= width) or
                np.any(row_coords_rot < 0) or np.any(row_coords_rot >= height)):
            return i, np.nan, np.nan

        psf = _G["psf"]
        psf_template_mask = _G["psf_template_mask"]
        use_subpixel = _G["use_subpixel_psf_astrometry"]

        patches, mlst, clst, hlst = [], [], [], []

        for l, ang in enumerate(angles_px):
            col_rot = int(np.round(ang[0]))
            row_rot = int(np.round(ang[1]))

            if not (0 <= row_rot < height and 0 <= col_rot < width):
                return i, np.nan, np.nan

            apatch_cached, m_cached, Cinv_cached = _get_or_compute_stats(row_rot, col_rot)
            current_patch = apatch_cached[l]

            if (m_cached is None or Cinv_cached is None or
                    np.any(np.isnan(m_cached)) or np.any(np.isnan(Cinv_cached)) or
                    np.all(np.isnan(current_patch))):
                return i, np.nan, np.nan

            mlst.append(m_cached)
            clst.append(Cinv_cached)
            patches.append(current_patch)

            if l in precomputed_pixel_data:
                offax = precomputed_pixel_data[l]
            elif use_subpixel:
                subpix_row = ang[1] - row_rot
                subpix_col = ang[0] - col_rot
                offax = frame_shift(
                    psf, subpix_row, subpix_col,
                    imlib='vip-fft', interpolation='lanczos4',
                    border_mode='reflect',
                )[psf_template_mask]
            else:
                offax = psf[psf_template_mask]

            hlst.append(offax)

        if not hlst:
            return i, np.nan, np.nan

        h_arr = np.array(hlst)
        Cinv_arr = np.array(clst)
        r_arr = np.array(patches)
        m_arr = np.array(mlst)

        if (h_arr.size == 0 or
                np.any(np.isnan(h_arr)) or np.any(np.isnan(Cinv_arr)) or
                np.any(np.isnan(r_arr)) or np.any(np.isnan(m_arr))):
            return i, np.nan, np.nan

        w_arr = wfl(h_arr, Cinv_arr, use_gpu=False)
        a_val = al(h_arr, w_arr, use_gpu=False)
        b_val = bl(w_arr, r_arr, m_arr, use_gpu=False)

        return i, a_val, b_val

    except Exception:
        return i, np.nan, np.nan


class PACO(PACOBase):
    """
    PACO variant that computes and caches background statistics per pixel
    during the main detection sweep, rather than pre-computing them globally.

    This avoids the large memory allocation required by
    :class:`~src.paco_base.PACO` for the full statistics maps while
    retaining the flexibility to process each pixel independently.
    GPU-based PSF shift pre-computation is available but currently
    disabled by default (see :meth:`_precompute_shifts_for_batch`).
    """

    def __init__(self, *args, cache_memory_fraction: Optional[float] = None,
                 radius_bin_width: float = 2.0, **kwargs):
        """
        Parameters
        ----------
        cache_memory_fraction : float, optional
            Deprecated; use `memory_fraction` (see PACOBase.__init__), from which
            the cache budget is derived (CACHE_BUDGET_SHARE of it). Kept so
            that existing benchmark scripts still run: if given, the cache
            pool is sized to this fraction of *available* RAM, as before,
            independently of `memory_fraction`.
        radius_bin_width : float
            Width (pixels) of the radius bins candidates are grouped into
            before sweeping (see PACOCalc docstring: "radius-ring
            traversal"). Smaller = tighter cache-sharing groups but more
            pool resets; larger = fewer resets but each ring's working set
            (and hence needed cache capacity) grows. 2.0 px is a reasonable
            default; override per call via
            PACOCalc(..., radius_bin_width=...).

        Also accepts `gpu_memory_fraction` via **kwargs (forwarded to
        PACOBase.__init__, which stores it as `self.gpu_memory_fraction`) --
        the VRAM analogue of `memory_fraction`, consumed by
        `_gpu_prefill_ring_cache`.
        """
        super().__init__(*args, **kwargs)
        self.precomputed_shifts: dict = {}
        if cache_memory_fraction is not None:
            warnings.warn("cache_memory_fraction is deprecated; use memory_fraction",
                          DeprecationWarning, stacklevel=2)
        self.cache_memory_fraction = cache_memory_fraction
        self.radius_bin_width = radius_bin_width

    def PACOCalc(self,
                 phi0s: np.ndarray,
                 use_subpixel_psf_astrometry: bool = True,
                 use_gpu: bool = True,
                 cpu: int = 1,
                 cache_memory_fraction: Optional[float] = None,
                 radius_bin_width: Optional[float] = None,
                 gpu_memory_fraction: Optional[float] = None,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute (a, b) maps with on-demand statistics caching.

        Candidates are swept in *radius-ring order*, not input order. A
        candidate's trajectory (see get_rotated_pixel_coords) keeps its
        polar radius exactly constant across all T frames — only the angle
        rotates — so two candidates at different radii can never touch the
        same background pixel, while candidates at the same radius share
        many of them. Grouping the sweep by radius bin therefore (a)
        maximises the shared-cache hit rate within a bin, and (b) lets the
        cache be reset to empty between bins (a completed bin's background
        pixels are provably never needed again), bounding the *effective*
        working set to one ring instead of the whole image. This does NOT
        change *which* pixels are processed or their individual results —
        only the order, and hence the caching behaviour — and coverage of
        every input position is verified below (see `_covered` assertion).

        Parameters
        ----------
        phi0s : np.ndarray, shape (N, 2)
            Array of (x, y) test positions.
        use_subpixel_psf_astrometry : bool
            Shift the PSF template to the sub-pixel rotated position.
        use_gpu : bool
            Enable GPU acceleration (requires CuPy and a CUDA device).
        cpu : int
            Number of CPU cores for parallel pixel processing.
        cache_memory_fraction : float, optional
            Deprecated; see __init__. Overrides the cache budget for this
            call only.
        radius_bin_width : float, optional
            Overrides `self.radius_bin_width` for this call.
        gpu_memory_fraction : float, optional
            Overrides `self.gpu_memory_fraction` (see PACOBase.__init__) for
            this call's per-ring GPU cache prefill.

        Returns
        -------
        a : np.ndarray, shape (N,)
            Normalization scalars.
        b : np.ndarray, shape (N,)
            Detection statistics.
        """
        npx = len(phi0s)

        a = np.zeros(npx)
        b = np.zeros(npx)

        # Radius-ring traversal order (see docstring above). `order` is a
        # permutation of range(npx) by construction (argsort never drops or
        # duplicates an index), and `sorted_bins` is sorted so every bin's
        # members are contiguous in `order` — both properties are relied on
        # below and checked defensively.
        eff_radius_bin_width = (radius_bin_width if radius_bin_width is not None
                                else self.radius_bin_width)
        cy, cx = self.height / 2.0, self.width / 2.0
        radii = np.hypot(phi0s[:, 0] - cy, phi0s[:, 1] - cx)
        bin_of = np.floor(radii / max(eff_radius_bin_width, 1e-9)).astype(np.int64)
        order = np.argsort(bin_of, kind="stable")
        sorted_bins = bin_of[order]
        assert order.shape[0] == npx and len(np.unique(order)) == npx, (
            "radius-ring ordering must be a permutation of all input indices")

        self.psf_template_mask = create_boolean_circular_mask(
            self.psf.shape, radius=self.patch_width // 2)
        # Use separate half-widths so non-square cubes are handled correctly.
        self.centered_grid_x_values, self.centered_grid_y_values = np.meshgrid(
            np.arange(-self.width / 2, self.width / 2),
            np.arange(-self.height / 2, self.height / 2))

        self.current_use_subpixel_psf_astrometry = use_subpixel_psf_astrometry
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self._effective_gpu_memory_fraction = (
            gpu_memory_fraction if gpu_memory_fraction is not None
            else self.gpu_memory_fraction)

        if cache_memory_fraction is not None:
            warnings.warn("cache_memory_fraction is deprecated; use memory_fraction",
                          DeprecationWarning, stacklevel=2)
        else:
            cache_memory_fraction = self.cache_memory_fraction
        budget = self._memory_budget_bytes()
        T = self.num_frames
        K = self.patch_area_pixels
        bytes_per_slot = (T * K + K + K * K) * 8

        # Populate globals BEFORE forking: each worker inherits the large,
        # static arrays (cube, psf, ...) via copy-on-write, at zero pickling
        # cost. The statistics cache pool below is backed by real shared
        # memory (see module docstring) so cache *hits* are plain memory
        # reads, not IPC.
        _G["cube"] = self.cube
        _G["patch_width"] = self.patch_width
        _G["patch_area"] = self.patch_area_pixels
        _G["grid_x"] = self.centered_grid_x_values
        _G["grid_y"] = self.centered_grid_y_values
        _G["angles"] = self.angles
        _G["width"] = self.width
        _G["height"] = self.height
        _G["psf"] = self.psf
        _G["psf_template_mask"] = self.psf_template_mask
        _G["use_subpixel_psf_astrometry"] = use_subpixel_psf_astrometry

        pool = None
        shm_blocks = []
        # Same capacity bound for the serial (plain dict) and the parallel
        # (shared-memory) cache, so both stay within the run's RAM budget.
        if cache_memory_fraction is not None:
            cache_budget = int(psutil.virtual_memory().available * cache_memory_fraction)
        else:
            cache_budget = int(budget * CACHE_BUDGET_SHARE)
        capacity = self._estimate_cache_capacity(cache_budget)
        cache_bytes = capacity * bytes_per_slot
        _G["capacity"] = capacity
        if self.verbose:
            print(f"Statistics cache: capacity={capacity} slots (T={T}, K={K})")
        if cpu > 1:
            H, W = self.height, self.width

            apatch_shm = shared_memory.SharedMemory(create=True, size=max(capacity * T * K * 8, 1))
            m_shm = shared_memory.SharedMemory(create=True, size=max(capacity * K * 8, 1))
            cinv_shm = shared_memory.SharedMemory(create=True, size=max(capacity * K * K * 8, 1))
            # slot_map is the key->slot index, covering the *whole* image
            # (int32, tiny: H*W*4 bytes) — see _get_or_compute_stats
            # docstring for why this must NOT be a Manager().dict().
            slot_map_shm = shared_memory.SharedMemory(create=True, size=max(H * W * 4, 1))
            shm_blocks = [apatch_shm, m_shm, cinv_shm, slot_map_shm]

            _G["apatch_pool"] = np.ndarray((capacity, T, K), dtype=np.float64, buffer=apatch_shm.buf)
            _G["m_pool"] = np.ndarray((capacity, K), dtype=np.float64, buffer=m_shm.buf)
            _G["cinv_pool"] = np.ndarray((capacity, K, K), dtype=np.float64, buffer=cinv_shm.buf)
            slot_map = np.ndarray((H, W), dtype=np.int32, buffer=slot_map_shm.buf)
            slot_map.fill(-1)
            _G["slot_map"] = slot_map

            # fork-inherited (no Manager involved anywhere in this path —
            # multiprocessing.Manager() itself starts its server process via
            # the platform's *global default* start method, "spawn" on
            # macOS, regardless of the fork context used elsewhere; measured
            # as a severe regression when it was on the hot path).
            ctx = multiprocessing.get_context("fork")
            _G["lock"] = ctx.Lock()
            _G["next_slot"] = ctx.Value('i', 0)
            _G["total_misses"] = ctx.Value('l', 0)  # diagnostic: see last_cache_stats
            pool = ctx.Pool(processes=cpu)
        else:
            _G["local_cache"] = {}
            _G["miss_counter"] = [0]
            _G["lock"] = None

        if cache_memory_fraction is not None:
            # Deprecated path: previous fixed split (batches: 65% of available).
            batch_budget = int(psutil.virtual_memory().available * 0.65)
        else:
            batch_budget = max(budget - cache_bytes, 0)
        pixel_batch_size = self._estimate_cpu_pixel_batch_size(batch_budget)

        covered = np.zeros(npx, dtype=bool)  # coverage proof, checked below
        total_rings = int(len(np.unique(sorted_bins)))
        progress_start_time = time.time()
        pixels_done = 0
        ring_counter = 0
        if self.verbose:
            print(f"Radius-ring sweep starting: {npx} pixels, "
                  f"{total_rings} rings, cpu={cpu}", flush=True)
        try:
            ring_start = 0
            first_ring = True
            while ring_start < npx:
                # Contiguous run of `order` sharing the same radius bin
                # (contiguous because `order` is sorted by bin_of).
                ring_end = ring_start + 1
                while (ring_end < npx and
                      sorted_bins[ring_end] == sorted_bins[ring_start]):
                    ring_end += 1
                ring_orig_idx = order[ring_start:ring_end]
                ring_phi0s = phi0s[ring_orig_idx]

                # A completed ring's background pixels are provably never
                # revisited by any other ring (constant-radius trajectories,
                # see docstring) — reset the cache so the *next* ring starts
                # from a clean, small working set instead of accumulating
                # entries from every ring processed so far.
                if not first_ring:
                    if pool is not None:
                        _G["slot_map"].fill(-1)
                        _G["next_slot"].value = 0
                    else:
                        _G["local_cache"].clear()
                first_ring = False

                self._gpu_prefill_ring_cache(ring_phi0s, pool)

                for j in range(0, len(ring_phi0s), pixel_batch_size):
                    sub_phi0s = ring_phi0s[j:j + pixel_batch_size]
                    sub_orig_idx = ring_orig_idx[j:j + pixel_batch_size]

                    precomputed_subset = {
                        int(pixel_idx): self.precomputed_shifts[int(pixel_idx)]
                        for pixel_idx in sub_orig_idx
                        if int(pixel_idx) in self.precomputed_shifts
                    }

                    process_args = [
                        (int(orig_i), p0, precomputed_subset.get(int(orig_i), {}))
                        for orig_i, p0 in zip(sub_orig_idx, sub_phi0s)
                    ]

                    if pool is not None:
                        results = pool.map(_process_pixel_worker, process_args)
                    else:
                        results = [
                            _process_pixel_worker(args)
                            for args in tqdm.tqdm(process_args,
                                                  desc="  - CPU calculations",
                                                  leave=False,
                                                  disable=not self.verbose)
                        ]

                    for res_i, res_a, res_b in results:
                        if res_i is not None:
                            a[res_i] = res_a
                            b[res_i] = res_b
                            covered[res_i] = True

                ring_counter += 1
                pixels_done += len(ring_orig_idx)
                if self.verbose:
                    elapsed = time.time() - progress_start_time
                    pct = 100.0 * pixels_done / npx
                    rate = pixels_done / elapsed if elapsed > 0 else float("nan")
                    eta = (npx - pixels_done) / rate if rate > 0 else float("nan")
                    print(f"  [ring {ring_counter}/{total_rings}] "
                          f"bin={int(sorted_bins[ring_start])} ring_size={len(ring_orig_idx)} "
                          f"-> {pixels_done}/{npx} pixels ({pct:.1f}%) "
                          f"elapsed={elapsed:.1f}s eta={eta:.1f}s "
                          f"rate={rate:.1f}px/s", flush=True)

                ring_start = ring_end

            n_covered = int(covered.sum())
            if n_covered != npx:
                raise RuntimeError(
                    f"Radius-ring traversal only covered {n_covered}/{npx} "
                    f"input positions — this is a bug, not a data issue "
                    f"(every position must be visited exactly once).")
        finally:
            if pool is not None:
                pool.close()
                pool.join()
            # Drop numpy views onto the shared buffers before closing/
            # unlinking them (releases the mmap cleanly), then remove the
            # underlying OS-level shared memory objects — SharedMemory does
            # NOT free itself on garbage collection, it leaks (e.g. under
            # /dev/shm) until unlink() is called.
            for key in ("apatch_pool", "m_pool", "cinv_pool", "slot_map"):
                _G.pop(key, None)
            for shm in shm_blocks:
                shm.close()
                shm.unlink()

        total_accesses = npx * self.num_frames
        misses = (_G["total_misses"].value if "total_misses" in _G
                 else _G["miss_counter"][0])
        self.last_cache_stats = {
            "total_accesses": total_accesses,
            "misses": misses,
            "hits": total_accesses - misses,
            "hit_rate": (1.0 - misses / total_accesses) if total_accesses else float("nan"),
        }
        for key in ("total_misses", "miss_counter", "lock", "next_slot", "local_cache"):
            _G.pop(key, None)

        if self.verbose:
            print("FullPACO processing finished.")
            print(f"Cache stats: {self.last_cache_stats}")

        return a, b

    # ------------------------------------------------------------------
    # GPU statistics pre-fill (per ring)
    # ------------------------------------------------------------------

    def _gpu_prefill_ring_cache(self, ring_phi0s: np.ndarray, pool) -> None:
        """
        Precompute background statistics for every unique pixel this ring's
        candidates will touch, batched on the GPU, BEFORE dispatching the
        ring's per-candidate worker pool.

        A ring's candidates share background pixels heavily by construction
        (see PACOCalc's docstring on radius-ring traversal), so the set of
        *distinct* (row, col) positions any candidate in this ring will ever
        look up is usually much smaller than
        len(ring_phi0s) * num_frames. Computing that whole set up front with
        one batched GPU pass (GPUStatisticsBatcher.compute_stats_streamed)
        turns every subsequent lookup during this ring's pool.map into a
        guaranteed cache hit.

        This is a performance-only optimisation: the existing on-demand
        _get_or_compute_stats path remains the correctness fallback for
        anything this prefill doesn't cover (cache-capacity limit reached,
        an edge/rounding case not swept up here, or GPU unavailable) — a bug
        in this method can only leave work for the lazy path to redo
        correctly, never produce a wrong (a, b).

        No-op if self.use_gpu is False (GPU_AVAILABLE is folded into
        self.use_gpu already, see PACOCalc).

        Vectorised, not a per-candidate Python loop: an earlier version
        called get_rotated_pixel_coords once per candidate in a plain
        `for p0 in ring_phi0s:` loop, entirely in the parent process
        (unparallelised). Summed over a full sweep that's one call per
        candidate in the *whole field*, regardless of how many rings it is
        split into or how many worker processes are available — a fixed,
        nproc-independent cost. Measured on a 1028x1028 field: this was the
        dominant cause of SHM_RING running 54-78% *slower* than a
        no-ring design that has no such step, with the overhead fraction
        *growing* with nproc (the classic signature of an unparallelised
        serial bottleneck under Amdahl's law) rather than staying roughly
        constant as a pure cache-reset cost would. Batching the whole
        ring's trajectories into a handful of NumPy array ops (using the
        same broadcasting cart_to_pol/pol_to_cart already support) turns
        that O(ring_size) — and, cumulatively, O(field_size) — Python-level
        call count into O(1) per ring.
        """
        if not self.use_gpu:
            return

        from vip_hci.var import cart_to_pol, pol_to_cart

        # Vectorised equivalent of calling get_rotated_pixel_coords once per
        # candidate in ring_phi0s: `rad`/`theta` computed for all N
        # candidates at once (shape (N,)), then broadcast against the T
        # frame angles to get every candidate's full trajectory in one shot
        # (shape (N,T)). Must match get_rotated_pixel_coords exactly (same
        # width/2, height/2 reference on both the subtraction and the
        # add-back -- see that function's docstring for why using (N-1)/2
        # on one side only silently introduced a constant 0.5px offset).
        width, height = self.width, self.height
        phi0_x = ring_phi0s[:, 1] - width / 2.0    # shape (N,)
        phi0_y = ring_phi0s[:, 0] - height / 2.0   # shape (N,)
        rad, theta = cart_to_pol(phi0_x, phi0_y)   # shape (N,) each
        angles_rad = theta[:, None] - self.angles[None, :]   # shape (N, T)
        nx, ny = pol_to_cart(rad[:, None], angles_rad)       # shape (N, T), broadcasts rad
        nx = nx + width / 2.0
        ny = ny + height / 2.0

        valid = (nx >= 0) & (nx < width) & (ny >= 0) & (ny < height)
        if not np.any(valid):
            return
        rows_all = np.round(ny[valid]).astype(np.int64)
        cols_all = np.round(nx[valid]).astype(np.int64)
        # Bounds-check AGAIN after rounding: a continuous coordinate in
        # [width-0.5, width) (or the [height-0.5, height) row analogue)
        # passes the continuous check above but rounds to `width`/`height`
        # -- one past the last valid index. The lazy path
        # (_process_pixel_worker) re-checks bounds per-frame after
        # rounding for exactly this reason (returns NaN there instead of
        # indexing out of bounds); this prefill path was previously missing
        # that second check, which could raise an uncaught IndexError on
        # slot_map/the shared-memory pools for candidates near the field's
        # outer edge.
        in_bounds = ((rows_all >= 0) & (rows_all < height) &
                    (cols_all >= 0) & (cols_all < width))
        rows_all = rows_all[in_bounds]
        cols_all = cols_all[in_bounds]
        if rows_all.size == 0:
            return

        # Unique (row, col) pairs via a linear-index encode + np.unique
        # (a single C-level pass) instead of building a Python set from
        # zipped tuples.
        lin = rows_all * np.int64(width) + cols_all
        unique_lin = np.unique(lin)
        n_unique_total = int(unique_lin.size)

        capacity = _G["capacity"] if pool is not None else None
        if capacity is not None and n_unique_total > capacity:
            # Cache pool is capacity-bounded (see _estimate_cache_capacity);
            # anything beyond it is left for the lazy on-demand path, which
            # already handles "compute but don't cache beyond capacity"
            # correctly.
            unique_lin = unique_lin[:capacity]
        rows_u = (unique_lin // width).astype(np.int64)
        cols_u = (unique_lin % width).astype(np.int64)
        n_unique = int(unique_lin.size)

        # Patch extraction stays a per-position loop (each position has a
        # distinct circular mask location, not trivially vectorisable) --
        # but this is now bounded by the *unique* miss count for this ring,
        # not by len(ring_phi0s) * num_frames, and was never the dominant
        # cost (see docstring measurement).
        patches = np.array([_g_get_patch((int(r), int(c)))
                            for r, c in zip(rows_u, cols_u)])
        flat = patches.reshape(n_unique, -1)
        valid_mask = ~np.any(np.isnan(flat), axis=1)
        if not np.any(valid_mask):
            return
        rows_v = rows_u[valid_mask]
        cols_v = cols_u[valid_mask]
        valid_patches = patches[valid_mask]
        n_valid = int(valid_patches.shape[0])

        from .gpu import GPUStatisticsBatcher
        batcher = GPUStatisticsBatcher(
            gpu_memory_safety_factor=self._effective_gpu_memory_fraction,
            verbose=self.verbose)
        m_batch, Cinv_batch = batcher.compute_stats_streamed(valid_patches)

        if pool is not None:
            slot_map = _G["slot_map"]
            next_slot = _G["next_slot"]
            # Bulk write instead of a per-position Python loop with a
            # per-iteration "already cached?" check: safe because this
            # method only ever runs in the parent process, sequentially,
            # immediately after that ring's cache reset and strictly
            # before any worker is dispatched for this ring -- no other
            # writer can have claimed any of these slots yet, so the
            # check was always False in practice.
            cur = next_slot.value
            n_fit = max(0, min(n_valid, _G["capacity"] - cur))
            if n_fit > 0:
                sl = slice(cur, cur + n_fit)
                _G["apatch_pool"][sl] = valid_patches[:n_fit]
                _G["m_pool"][sl] = m_batch[:n_fit]
                _G["cinv_pool"][sl] = Cinv_batch[:n_fit]
                slot_map[rows_v[:n_fit], cols_v[:n_fit]] = np.arange(cur, cur + n_fit)
                next_slot.value = cur + n_fit
                with _G["total_misses"].get_lock():
                    _G["total_misses"].value += n_fit
        else:
            n_fit = n_valid
            cache = _G["local_cache"]
            for i in range(n_valid):
                cache[(int(rows_v[i]), int(cols_v[i]))] = (
                    valid_patches[i], m_batch[i], Cinv_batch[i])
            _G["miss_counter"][0] += n_valid

        if self.verbose:
            print(f"  [GPU prefill] ring covered {n_fit}/{n_unique_total} "
                  f"unique background pixels", flush=True)

    # ------------------------------------------------------------------
    # Memory estimation
    # ------------------------------------------------------------------

    def _estimate_cpu_pixel_batch_size(self, budget_bytes: int) -> int:
        """
        How many candidate pixels per sweep batch fit in `budget_bytes`
        (the run's RAM budget minus the shared cache, see PACOCalc), with a
        20% margin on the per-candidate PSF-template arrays.
        """
        float_size = np.dtype(np.float64).itemsize
        pixels_in_psf_mask = int(np.sum(self.psf_template_mask))
        memory_per_pixel = len(self.angles) * pixels_in_psf_mask * float_size * 1.2
        if memory_per_pixel == 0:
            return 500
        batch_size = max(1, int(budget_bytes / memory_per_pixel))
        if self.verbose:
            print(f"Sweep batch budget: {budget_bytes / 1024**3:.2f} GiB, "
                  f"{memory_per_pixel / 1024**2:.2f} MB/pixel -> "
                  f"batch size={batch_size}")
        return batch_size

    def _estimate_cache_capacity(self, budget_bytes: int) -> int:
        """
        How many (apatch, m, Cinv) entries the shared statistics cache can
        hold in `budget_bytes`. Capped at H*W: keys are image pixels, so
        more slots than that could never be used.
        """
        T = self.num_frames
        K = self.patch_area_pixels
        bytes_per_slot = (T * K + K + K * K) * 8
        if bytes_per_slot == 0:
            return 2000
        capacity = max(1, min(int(budget_bytes / bytes_per_slot),
                              self.height * self.width))
        if self.verbose:
            print(f"Cache pool: {budget_bytes / 1024**3:.2f} GiB budget, "
                  f"{bytes_per_slot / 1024:.1f} KiB/slot -> "
                  f"capacity={capacity}")
        return capacity

    # ------------------------------------------------------------------
    # GPU PSF pre-computation (optional, disabled by default)
    # ------------------------------------------------------------------

    def _precompute_shifts_for_batch(self,
                                      pixel_batch: np.ndarray,
                                      original_start_index: int) -> None:
        """
        Pre-compute sub-pixel PSF shifts on GPU for an entire pixel batch.

        Results are stored in :attr:`precomputed_shifts` keyed by
        pixel index. Called before the CPU processing loop when GPU is
        enabled. Currently unused in the default execution path — activate
        by un-commenting the call site in :meth:`PACOCalc`.

        Parameters
        ----------
        pixel_batch : np.ndarray
            Subset of test positions for the current processing batch.
        original_start_index : int
            Flat index of the first pixel in pixel_batch within the global
            phi0s array.
        """
        self.precomputed_shifts = {}
        gpu_shifter = None
        try:
            gpu_shifter = GPUFrameShifter(
                self.psf, self.psf_template_mask, verbose=self.verbose)

            all_shifts, shift_to_pixel, valid_mask = [], {}, []
            shift_idx = 0

            for j, p0 in enumerate(pixel_batch):
                orig_idx = original_start_index + j
                is_valid = False
                try:
                    angles_px = get_rotated_pixel_coords(
                        self.centered_grid_x_values,
                        self.centered_grid_y_values,
                        p0, self.angles)
                    col_c, row_c = angles_px[:, 0], angles_px[:, 1]
                    if not (np.any(col_c < 0) or np.any(col_c >= self.width) or
                            np.any(row_c < 0) or np.any(row_c >= self.height)):
                        is_valid = True
                except Exception:
                    angles_px = np.zeros((len(self.angles), 2))

                for l, ang in enumerate(angles_px):
                    if is_valid:
                        yi = int(np.round(ang[0]))
                        xi = int(np.round(ang[1]))
                        all_shifts.append((ang[0] - yi, ang[1] - xi))
                        shift_to_pixel[shift_idx] = (orig_idx, l)
                        valid_mask.append(True)
                    else:
                        all_shifts.append((0.0, 0.0))
                        valid_mask.append(False)
                    shift_idx += 1

            valid_mask_array = np.array(valid_mask)
            shift_gen = gpu_shifter.stream_precomputed_shifts(
                all_shifts, valid_mask_array)

            num_valid = int(np.sum(valid_mask_array))
            pbar = tqdm.tqdm(shift_gen, total=num_valid,
                             desc="  - GPU preprocessing",
                             leave=False, disable=not self.verbose)

            num_processed = 0
            for shift_idx, shifted_psf_flat in pbar:
                if shift_idx in shift_to_pixel:
                    pixel_idx, angle_idx = shift_to_pixel[shift_idx]
                    if pixel_idx not in self.precomputed_shifts:
                        self.precomputed_shifts[pixel_idx] = {}
                    self.precomputed_shifts[pixel_idx][angle_idx] = shifted_psf_flat
                    num_processed += 1

            if self.verbose:
                print(f"GPU preprocessing: {num_processed} shifts computed.")

        finally:
            if gpu_shifter is not None:
                del gpu_shifter
            try:
                import cupy
                cupy.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass


# Former class name (placeholder from development).
FullPACOv2 = deprecated_alias(PACO, "FullPACOv2")
