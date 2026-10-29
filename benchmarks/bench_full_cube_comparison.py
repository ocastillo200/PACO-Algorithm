#!/usr/bin/env python3
"""
Full-cube, 3-way comparison for FullPACOv2's evolution:

  1. ORIGINAL   - the design this session started with: a fresh pool_map()
                  Pool per pixel batch, cross-process Manager().dict() cache
                  storing full (apatch, m, Cinv) arrays per key.
  2. SHM_NORING - real shared memory for the cache pool + a (H, W) slot_map
                  index (no Manager at all), but candidates are swept in
                  their *input* (native raster) order — no radius grouping.
  3. SHM_RING   - same shared-memory design, but candidates are swept in
                  *radius-ring* order with the cache reset between rings
                  (src.full_paco.FullPACOv2, the current, real class). This
                  is the "innovative" technique: a candidate's trajectory
                  keeps constant radius across all frames (see
                  src/utils.py:get_rotated_pixel_coords), so candidates at
                  different radii never share a background pixel, and a
                  completed ring's cache entries are provably never needed
                  again.

By default this processes the ENTIRE native pixel grid of the cube (every
(row, col) in the field of view) as candidate positions — i.e. exactly what
PACO.run() does for a real detection map, not a subset. "Candidate
positions" in PACO *are* the image's pixels: every one of them is tested as
a possible companion location. Use --npix for a faster partial run (a dense
block, not random — see _common.make_test_positions for why that matters
for SHM_RING specifically).

For ORIGINAL and SHM_NORING vs SHM_RING, this also reports:
  - wall-clock time
  - peak memory (parent / single worker / naive combined — see
    _common.PeakMemoryMonitor for exactly what each of these means and its
    caveats)
  - cache hit rate (SHM_NORING and SHM_RING only; ORIGINAL's Manager dict
    has no equivalent cheap instrumentation point without changing its
    design, and it isn't the point of comparison here)

Usage
-----
    python3 benchmarks/bench_full_cube_comparison.py
    python3 benchmarks/bench_full_cube_comparison.py --nproc 2,4,8,16
    python3 benchmarks/bench_full_cube_comparison.py --npix 5000   # faster partial run
    python3 benchmarks/bench_full_cube_comparison.py --skip-old    # ORIGINAL can be very slow on a full grid
    python3 benchmarks/bench_full_cube_comparison.py --data synthetic --shape 4800,200,200
    python3 benchmarks/bench_full_cube_comparison.py --csv full_cube_comparison.csv
"""
import multiprocessing
import time
from multiprocessing import shared_memory

import numpy as np

from _common import (arrays_equal, make_arg_parser, make_test_positions,
                     load_dataset_from_args, print_system_info, ResultTable,
                     parse_nproc_list, make_memory_monitor, mem_columns, mem_str)

from src.paco_base import PACO
from src.full_paco import FullPACOv2
from src.math_utils import al, bl, compute_statistics_at_pixel, wfl
from src.utils import create_boolean_circular_mask, extract_patch_window, get_rotated_pixel_coords


# ═════════════════════════════════════════════════════════════════════════
# 1. ORIGINAL — the design this whole optimization effort started from.
# ═════════════════════════════════════════════════════════════════════════
class FullPACOv2_Original(PACO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.precomputed_shifts = {}

    def PACOCalc(self, phi0s, use_subpixel_psf_astrometry=True, use_gpu=True,
                 cpu=1, **kwargs):
        from src.gpu import GPU_AVAILABLE
        from vip_hci.config.utils_conf import iterable, pool_map

        npx = len(phi0s)
        a = np.zeros(npx); b = np.zeros(npx)
        self.psf_template_mask = create_boolean_circular_mask(
            self.psf.shape, radius=self.patch_width // 2)
        self.centered_grid_x_values, self.centered_grid_y_values = np.meshgrid(
            np.arange(-self.width / 2, self.width / 2),
            np.arange(-self.height / 2, self.height / 2))
        self.current_use_subpixel_psf_astrometry = use_subpixel_psf_astrometry
        self.use_gpu = use_gpu and GPU_AVAILABLE

        if cpu > 1:
            manager = multiprocessing.Manager()
            self.cached_pixel_data = manager.dict()
        else:
            self.cached_pixel_data = {}

        pixel_batch_size = self._estimate_cpu_pixel_batch_size()
        n_batches = -(-npx // pixel_batch_size)
        progress_t0 = time.perf_counter()
        if self.verbose:
            print(f"[ORIGINAL] starting: {npx} pixels, batch_size={pixel_batch_size}, "
                  f"{n_batches} batches, cpu={cpu}", flush=True)

        for bi, i in enumerate(range(0, npx, pixel_batch_size)):
            batch_slice = slice(i, min(i + pixel_batch_size, npx))
            pixel_batch = phi0s[batch_slice]
            precomputed_subset = {
                pixel_idx: self.precomputed_shifts[pixel_idx]
                for pixel_idx in range(i, i + len(pixel_batch))
                if pixel_idx in self.precomputed_shifts
            }
            process_args = [(i + j, p0, precomputed_subset.get(i + j, {}))
                            for j, p0 in enumerate(pixel_batch)]
            if cpu > 1:
                results = pool_map(cpu, self._process_pixel_for_calc,
                                   iterable(process_args))
            else:
                results = [self._process_pixel_for_calc(args) for args in process_args]
            for res_i, res_a, res_b in results:
                if res_i is not None:
                    a[res_i] = res_a; b[res_i] = res_b

            if self.verbose:
                done = min(i + pixel_batch_size, npx)
                elapsed = time.perf_counter() - progress_t0
                rate = done / elapsed if elapsed > 0 else float("nan")
                eta = (npx - done) / rate if rate > 0 else float("nan")
                print(f"  [ORIGINAL batch {bi + 1}/{n_batches}] "
                      f"{done}/{npx} pixels ({100.0 * done / npx:.1f}%) "
                      f"elapsed={elapsed:.1f}s eta={eta:.1f}s rate={rate:.1f}px/s",
                      flush=True)
        return a, b

    def _process_pixel_for_calc(self, args_tuple):
        from vip_hci.preproc.rescaling import frame_shift
        i, p0, precomputed_pixel_data = args_tuple
        try:
            angles_px = get_rotated_pixel_coords(
                self.centered_grid_x_values, self.centered_grid_y_values,
                p0, self.angles)
            col_coords_rot = angles_px[:, 0]
            row_coords_rot = angles_px[:, 1]
            if (np.any(col_coords_rot < 0) or np.any(col_coords_rot >= self.width) or
                    np.any(row_coords_rot < 0) or np.any(row_coords_rot >= self.height)):
                return i, np.nan, np.nan

            patches, mlst, clst, hlst = [], [], [], []
            for l, ang in enumerate(angles_px):
                col_rot = int(np.round(ang[0]))
                row_rot = int(np.round(ang[1]))
                if not (0 <= row_rot < self.height and 0 <= col_rot < self.width):
                    return i, np.nan, np.nan

                cache_key = (row_rot, col_rot)
                if cache_key not in self.cached_pixel_data:
                    apatch = self.get_patch((row_rot, col_rot))
                    m, Cinv = compute_statistics_at_pixel(apatch)
                    self.cached_pixel_data[cache_key] = (apatch, m, Cinv)
                apatch_cached, m_cached, Cinv_cached = self.cached_pixel_data[cache_key]
                current_patch = apatch_cached[l]

                if (m_cached is None or Cinv_cached is None or
                        np.any(np.isnan(m_cached)) or np.any(np.isnan(Cinv_cached)) or
                        np.all(np.isnan(current_patch))):
                    return i, np.nan, np.nan

                mlst.append(m_cached); clst.append(Cinv_cached); patches.append(current_patch)

                if l in precomputed_pixel_data:
                    offax = precomputed_pixel_data[l]
                elif self.current_use_subpixel_psf_astrometry:
                    subpix_row = ang[1] - row_rot
                    subpix_col = ang[0] - col_rot
                    offax = frame_shift(self.psf, subpix_row, subpix_col,
                                        imlib='vip-fft', interpolation='lanczos4',
                                        border_mode='reflect')[self.psf_template_mask]
                else:
                    offax = self.psf[self.psf_template_mask]
                hlst.append(offax)

            if not hlst:
                return i, np.nan, np.nan
            h_arr = np.array(hlst); Cinv_arr = np.array(clst)
            r_arr = np.array(patches); m_arr = np.array(mlst)
            if (h_arr.size == 0 or np.any(np.isnan(h_arr)) or np.any(np.isnan(Cinv_arr)) or
                    np.any(np.isnan(r_arr)) or np.any(np.isnan(m_arr))):
                return i, np.nan, np.nan

            w_arr = wfl(h_arr, Cinv_arr, use_gpu=False)
            a_val = al(h_arr, w_arr, use_gpu=False)
            b_val = bl(w_arr, r_arr, m_arr, use_gpu=False)
            return i, a_val, b_val
        except Exception:
            return i, np.nan, np.nan

    def _estimate_cpu_pixel_batch_size(self):
        import psutil
        try:
            available_ram = psutil.virtual_memory().available
            usable_ram = available_ram * 0.65
            float_size = np.dtype(np.float64).itemsize
            pixels_in_psf_mask = int(np.sum(self.psf_template_mask))
            memory_per_pixel = len(self.angles) * pixels_in_psf_mask * float_size * 1.2
            if memory_per_pixel == 0:
                return 500
            return max(1, int(usable_ram / memory_per_pixel))
        except Exception:
            return 500


# ═════════════════════════════════════════════════════════════════════════
# 2. SHM_NORING — real shared memory (pool + slot_map, no Manager at all),
# but candidates swept in *input* order, no ring grouping/reset. This is
# the intermediate design from earlier in this session, preserved here
# since it was superseded in src/full_paco.py by the ring-order version.
# Instrumented with the same last_cache_stats hit-rate tracking as the
# real class, for a fair 2-vs-3 comparison.
# ═════════════════════════════════════════════════════════════════════════
_G2: dict = {}


def _g2_get_patch(px):
    # Same windowed extraction as src/utils.py:extract_patch_window (see
    # its docstring) -- kept in sync deliberately so this benchmark's
    # SHM_NORING reference isolates the ring-vs-no-ring traversal
    # difference only, not an unrelated patch-extraction inefficiency that
    # only one side had fixed.
    return extract_patch_window(_G2["cube"], px, _G2["patch_width"], _G2["patch_area"])


def _g2_get_or_compute_stats(row_rot, col_rot):
    lock = _G2["lock"]
    if lock is None:
        cache = _G2["local_cache"]
        key = (row_rot, col_rot)
        cached = cache.get(key)
        if cached is not None:
            return cached
        _G2["miss_counter"][0] += 1
        apatch = _g2_get_patch((row_rot, col_rot))
        m, Cinv = compute_statistics_at_pixel(apatch)
        m_safe = m if m is not None else np.full(apatch.shape[1], np.nan)
        Cinv_safe = Cinv if Cinv is not None else np.full((apatch.shape[1],) * 2, np.nan)
        cache[key] = (apatch, m_safe, Cinv_safe)
        return apatch, m_safe, Cinv_safe

    slot_map = _G2["slot_map"]
    slot = slot_map[row_rot, col_rot]
    if slot >= 0:
        return (_G2["apatch_pool"][slot], _G2["m_pool"][slot], _G2["cinv_pool"][slot])

    with _G2["total_misses"].get_lock():
        _G2["total_misses"].value += 1
    apatch = _g2_get_patch((row_rot, col_rot))
    m, Cinv = compute_statistics_at_pixel(apatch)
    K = apatch.shape[1]
    m_safe = m if m is not None else np.full(K, np.nan)
    Cinv_safe = Cinv if Cinv is not None else np.full((K, K), np.nan)

    with lock:
        slot = slot_map[row_rot, col_rot]
        if slot >= 0:
            return (_G2["apatch_pool"][slot], _G2["m_pool"][slot], _G2["cinv_pool"][slot])
        next_slot = _G2["next_slot"]
        cur = next_slot.value
        if cur >= _G2["capacity"]:
            return apatch, m_safe, Cinv_safe
        next_slot.value = cur + 1
        _G2["apatch_pool"][cur] = apatch
        _G2["m_pool"][cur] = m_safe
        _G2["cinv_pool"][cur] = Cinv_safe
        slot_map[row_rot, col_rot] = cur
        return apatch, m_safe, Cinv_safe


def _g2_process_pixel_worker(args_tuple):
    from vip_hci.preproc.rescaling import frame_shift
    i, p0, precomputed_pixel_data = args_tuple
    try:
        angles_px = get_rotated_pixel_coords(_G2["grid_x"], _G2["grid_y"], p0, _G2["angles"])
        width, height = _G2["width"], _G2["height"]
        if (np.any(angles_px[:, 0] < 0) or np.any(angles_px[:, 0] >= width) or
                np.any(angles_px[:, 1] < 0) or np.any(angles_px[:, 1] >= height)):
            return i, np.nan, np.nan

        psf = _G2["psf"]; psf_template_mask = _G2["psf_template_mask"]
        use_subpixel = _G2["use_subpixel_psf_astrometry"]
        patches, mlst, clst, hlst = [], [], [], []

        for l, ang in enumerate(angles_px):
            col_rot = int(np.round(ang[0])); row_rot = int(np.round(ang[1]))
            if not (0 <= row_rot < height and 0 <= col_rot < width):
                return i, np.nan, np.nan
            apatch_cached, m_cached, Cinv_cached = _g2_get_or_compute_stats(row_rot, col_rot)
            current_patch = apatch_cached[l]
            if (m_cached is None or Cinv_cached is None or
                    np.any(np.isnan(m_cached)) or np.any(np.isnan(Cinv_cached)) or
                    np.all(np.isnan(current_patch))):
                return i, np.nan, np.nan
            mlst.append(m_cached); clst.append(Cinv_cached); patches.append(current_patch)

            if l in precomputed_pixel_data:
                offax = precomputed_pixel_data[l]
            elif use_subpixel:
                subpix_row = ang[1] - row_rot; subpix_col = ang[0] - col_rot
                offax = frame_shift(psf, subpix_row, subpix_col, imlib='vip-fft',
                                    interpolation='lanczos4', border_mode='reflect')[psf_template_mask]
            else:
                offax = psf[psf_template_mask]
            hlst.append(offax)

        if not hlst:
            return i, np.nan, np.nan
        h_arr = np.array(hlst); Cinv_arr = np.array(clst)
        r_arr = np.array(patches); m_arr = np.array(mlst)
        if (h_arr.size == 0 or np.any(np.isnan(h_arr)) or np.any(np.isnan(Cinv_arr)) or
                np.any(np.isnan(r_arr)) or np.any(np.isnan(m_arr))):
            return i, np.nan, np.nan
        w_arr = wfl(h_arr, Cinv_arr, use_gpu=False)
        a_val = al(h_arr, w_arr, use_gpu=False)
        b_val = bl(w_arr, r_arr, m_arr, use_gpu=False)
        return i, a_val, b_val
    except Exception:
        return i, np.nan, np.nan


class FullPACOv2_SharedMemNoRing(PACO):
    def __init__(self, *args, cache_memory_fraction=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.precomputed_shifts = {}
        self.cache_memory_fraction = cache_memory_fraction
        self.last_cache_stats = None

    def _estimate_cpu_pixel_batch_size(self):
        import psutil
        try:
            available_ram = psutil.virtual_memory().available
            usable_ram = available_ram * 0.65
            float_size = np.dtype(np.float64).itemsize
            pixels_in_psf_mask = int(np.sum(self.psf_template_mask))
            memory_per_pixel = len(self.angles) * pixels_in_psf_mask * float_size * 1.2
            return max(1, int(usable_ram / memory_per_pixel)) if memory_per_pixel else 500
        except Exception:
            return 500

    def _estimate_cache_capacity(self, memory_fraction=None):
        import psutil
        if memory_fraction is None:
            memory_fraction = self.cache_memory_fraction
        try:
            usable_ram = psutil.virtual_memory().available * memory_fraction
            T = self.num_frames; K = self.patch_area_pixels
            bytes_per_slot = (T * K + K + K * K) * 8
            return max(1, int(usable_ram / bytes_per_slot)) if bytes_per_slot else 2000
        except Exception:
            return 2000

    def PACOCalc(self, phi0s, use_subpixel_psf_astrometry=True, use_gpu=True,
                 cpu=1, **kwargs):
        npx = len(phi0s)
        a = np.zeros(npx); b = np.zeros(npx)
        self.psf_template_mask = create_boolean_circular_mask(
            self.psf.shape, radius=self.patch_width // 2)
        self.centered_grid_x_values, self.centered_grid_y_values = np.meshgrid(
            np.arange(-self.width / 2, self.width / 2),
            np.arange(-self.height / 2, self.height / 2))

        _G2["cube"] = self.cube
        _G2["patch_width"] = self.patch_width
        _G2["patch_area"] = self.patch_area_pixels
        _G2["grid_x"] = self.centered_grid_x_values
        _G2["grid_y"] = self.centered_grid_y_values
        _G2["angles"] = self.angles
        _G2["width"] = self.width
        _G2["height"] = self.height
        _G2["psf"] = self.psf
        _G2["psf_template_mask"] = self.psf_template_mask
        _G2["use_subpixel_psf_astrometry"] = use_subpixel_psf_astrometry

        pixel_batch_size = self._estimate_cpu_pixel_batch_size()
        pool = None
        shm_blocks = []
        if cpu > 1:
            T = self.num_frames; K = self.patch_area_pixels
            H, W = self.height, self.width
            capacity = self._estimate_cache_capacity()
            apatch_shm = shared_memory.SharedMemory(create=True, size=max(capacity * T * K * 8, 1))
            m_shm = shared_memory.SharedMemory(create=True, size=max(capacity * K * 8, 1))
            cinv_shm = shared_memory.SharedMemory(create=True, size=max(capacity * K * K * 8, 1))
            slot_map_shm = shared_memory.SharedMemory(create=True, size=max(H * W * 4, 1))
            shm_blocks = [apatch_shm, m_shm, cinv_shm, slot_map_shm]
            _G2["apatch_pool"] = np.ndarray((capacity, T, K), dtype=np.float64, buffer=apatch_shm.buf)
            _G2["m_pool"] = np.ndarray((capacity, K), dtype=np.float64, buffer=m_shm.buf)
            _G2["cinv_pool"] = np.ndarray((capacity, K, K), dtype=np.float64, buffer=cinv_shm.buf)
            slot_map = np.ndarray((H, W), dtype=np.int32, buffer=slot_map_shm.buf)
            slot_map.fill(-1)
            _G2["slot_map"] = slot_map
            _G2["capacity"] = capacity

            ctx = multiprocessing.get_context("fork")
            _G2["lock"] = ctx.Lock()
            _G2["next_slot"] = ctx.Value('i', 0)
            _G2["total_misses"] = ctx.Value('l', 0)
            pool = ctx.Pool(processes=cpu)
        else:
            _G2["local_cache"] = {}
            _G2["miss_counter"] = [0]
            _G2["lock"] = None

        n_batches = -(-npx // pixel_batch_size)
        progress_t0 = time.perf_counter()
        if self.verbose:
            print(f"[SHM_NORING] starting: {npx} pixels, batch_size={pixel_batch_size}, "
                  f"{n_batches} batches, cpu={cpu}", flush=True)
        try:
            for bi, i in enumerate(range(0, npx, pixel_batch_size)):
                batch_slice = slice(i, min(i + pixel_batch_size, npx))
                pixel_batch = phi0s[batch_slice]
                process_args = [(i + j, p0, {}) for j, p0 in enumerate(pixel_batch)]
                if pool is not None:
                    results = pool.map(_g2_process_pixel_worker, process_args)
                else:
                    results = [_g2_process_pixel_worker(args) for args in process_args]
                for res_i, res_a, res_b in results:
                    if res_i is not None:
                        a[res_i] = res_a; b[res_i] = res_b

                if self.verbose:
                    done = min(i + pixel_batch_size, npx)
                    elapsed = time.perf_counter() - progress_t0
                    rate = done / elapsed if elapsed > 0 else float("nan")
                    eta = (npx - done) / rate if rate > 0 else float("nan")
                    print(f"  [SHM_NORING batch {bi + 1}/{n_batches}] "
                          f"{done}/{npx} pixels ({100.0 * done / npx:.1f}%) "
                          f"elapsed={elapsed:.1f}s eta={eta:.1f}s rate={rate:.1f}px/s",
                          flush=True)
        finally:
            if pool is not None:
                pool.close(); pool.join()
            for key in ("apatch_pool", "m_pool", "cinv_pool", "slot_map"):
                _G2.pop(key, None)
            for shm in shm_blocks:
                shm.close(); shm.unlink()

        total_accesses = npx * self.num_frames
        misses = _G2["total_misses"].value if "total_misses" in _G2 else _G2["miss_counter"][0]
        self.last_cache_stats = {
            "total_accesses": total_accesses, "misses": misses,
            "hits": total_accesses - misses,
            "hit_rate": (1.0 - misses / total_accesses) if total_accesses else float("nan"),
        }
        for key in ("total_misses", "miss_counter", "lock", "next_slot", "local_cache"):
            _G2.pop(key, None)
        return a, b


# ═════════════════════════════════════════════════════════════════════════
def main():
    parser = make_arg_parser(__doc__)
    parser.add_argument("--skip-old", action="store_true",
                        help="Skip the ORIGINAL variant (can be very slow "
                             "on a full-cube grid — no caching benefit at "
                             "all across candidates).")
    parser.add_argument("--no-progress", action="store_true",
                        help="Suppress per-batch/per-ring progress lines "
                             "(keeps the one summary line per variant/nproc).")
    parser.add_argument("--cache-memory-fraction", type=float, default=None,
                        help="Override cache_memory_fraction for SHM_NORING "
                             "and SHM_RING (default: each class's own "
                             "default, 0.25). Set this LOW (e.g. 0.01-0.05) "
                             "to deterministically force SHM_NORING's cache "
                             "past capacity regardless of how much RAM the "
                             "machine actually has — useful to make the "
                             "ring-vs-noring memory-bound crossover show up "
                             "reliably on a large field instead of "
                             "depending on whether this run happens to "
                             "exceed capacity naturally. Not passed to "
                             "ORIGINAL (Manager-dict has no capacity bound "
                             "to begin with).")
    parser.add_argument("--radius-bin-width", type=float, default=None,
                        help="Override SHM_RING's radius_bin_width (default: "
                             "the class's own default, 2.0px). Ring COUNT "
                             "scales as field_radius/bin_width, and each "
                             "ring's slot_map reset costs O(H*W) regardless "
                             "of ring width -- on a field much larger than "
                             "sph3 (e.g. 1028x1028 vs 160x160), the default "
                             "2.0px produces ~6x more rings AND ~40x more "
                             "per-reset work than sph3 saw, and this was "
                             "measured to make SHM_RING 54-78%% SLOWER than "
                             "SHM_NORING rather than the ~5-15%% overhead "
                             "seen at sph3 scale. Widening this (e.g. to "
                             "15-20 for a ~1000px-radius field) trades "
                             "finer cache-sharing granularity for far fewer "
                             "resets -- test whether it recovers most of "
                             "the lost speed while memory stays bounded.")
    parser.add_argument("--use-gpu", dest="use_gpu", action="store_true", default=None,
                        help="Force GPU statistics prefill on for SHM_RING "
                             "(src.full_paco.FullPACOv2 only -- SHM_NORING's "
                             "stand-in class never implements this). "
                             "Default (neither flag given): FullPACOv2's own "
                             "default (True if a CUDA device is available).")
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false",
                        help="Force GPU statistics prefill OFF for SHM_RING. "
                             "_gpu_prefill_ring_cache runs entirely in the "
                             "parent process, serially, once per ring, "
                             "summing to one full field-size pass regardless "
                             "of nproc -- suspected to be the dominant cause "
                             "of SHM_RING's disproportionate slowdown on "
                             "large fields (its cost is fixed in absolute "
                             "time but grows as a FRACTION of total time as "
                             "nproc increases, the classic signature of an "
                             "unparallelized serial bottleneck -- see the "
                             "nproc-scaling discussion). Compare against a "
                             "run WITHOUT this flag to isolate its effect.")
    args = parser.parse_args()
    print_system_info()

    cube, pa, psf = load_dataset_from_args(args)
    fwhm = args.fwhm
    T, H, W = cube.shape

    # Default: EVERY pixel of the field, matching real PACO.run() usage.
    if args.npix is None:
        phi0s = make_test_positions(H, W, H * W, sampling="full")
    else:
        phi0s = make_test_positions(H, W, args.npix, seed=args.seed,
                                    sampling=args.sampling)

    print(f"Dataset: cube={cube.shape}, fwhm={fwhm}, "
          f"N candidate positions={len(phi0s)} "
          f"({'FULL native grid' if args.npix is None else args.sampling})\n")

    nprocs = parse_nproc_list(args.nproc, __import__("os").cpu_count())
    print(f"nproc sweep: {nprocs}\n")

    table = ResultTable(["variant", "nproc", "time_s", "mem_parent_mb",
                        "mem_maxworker_mb", "mem_combined_mb",
                        "mem_real_mb", "hit_rate", "correct"])

    variants = []
    if not args.skip_old:
        variants.append(("ORIGINAL", FullPACOv2_Original, False))
    variants.append(("SHM_NORING", FullPACOv2_SharedMemNoRing, True))
    variants.append(("SHM_RING", FullPACOv2, True))

    reference_ab = {}  # nproc -> (a, b) from the first variant run, for correctness

    total_configs = len(nprocs) * len(variants)
    config_counter = 0
    script_t0 = time.perf_counter()

    for nproc in nprocs:
        print(f"--- nproc={nproc} ---", flush=True)
        for name, cls, has_hitrate in variants:
            config_counter += 1
            since_start = time.perf_counter() - script_t0
            print(f">>> [{config_counter}/{total_configs}] variant={name} nproc={nproc} "
                  f"(script running {since_start:.1f}s so far)", flush=True)

            extra_kwargs = {}
            if args.cache_memory_fraction is not None and cls is not FullPACOv2_Original:
                extra_kwargs["cache_memory_fraction"] = args.cache_memory_fraction
            if args.radius_bin_width is not None and cls is FullPACOv2:
                extra_kwargs["radius_bin_width"] = args.radius_bin_width
            obj = cls(cube, pa, psf, fwhm=fwhm, verbose=not args.no_progress, **extra_kwargs)
            calc_kwargs = {}
            if args.use_gpu is not None and cls is FullPACOv2:
                calc_kwargs["use_gpu"] = args.use_gpu
            with make_memory_monitor(args) as mon:
                t0 = time.perf_counter()
                a, b = obj.PACOCalc(phi0s, cpu=nproc, **calc_kwargs)
                elapsed = time.perf_counter() - t0

            if nproc not in reference_ab:
                reference_ab[nproc] = (a, b)
                ok = True
            else:
                ok, _ = arrays_equal(reference_ab[nproc][0], reference_ab[nproc][1], a, b)

            hit_rate = getattr(obj, "last_cache_stats", None)
            hit_rate_str = f"{hit_rate['hit_rate']*100:.1f}%" if hit_rate else "n/a"

            print(f"  {name:12s} time={elapsed:8.2f}s  {mem_str(mon)}  "
                  f"hit_rate={hit_rate_str}  correct={ok}", flush=True)

            table.add(variant=name, nproc=nproc, time_s=f"{elapsed:.2f}",
                      **mem_columns(mon),
                      hit_rate=hit_rate_str, correct=("OK" if ok else "MISMATCH"))
        print()

    table.print_table()
    if args.csv:
        table.write_csv(args.csv)


if __name__ == "__main__":
    main()
