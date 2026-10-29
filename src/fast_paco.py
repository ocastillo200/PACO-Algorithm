"""
FastPACOv2: batch-processing PACO variant with spatial interpolation.

Uses pre-computed c and d maps (Stage 1) then interpolates them at each
rotated test position (Stage 2), enabling vectorised processing.
"""
import multiprocessing
import time
import warnings
from typing import Generator, Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates
from vip_hci.fm import normalize_psf

from .gpu import GPU_AVAILABLE
from .math_utils import compute_statistics_at_pixel
from .paco_base import PACO
from .utils import create_boolean_circular_mask, extract_patch_window, get_rotated_pixel_coords

# ---------------------------------------------------------------------------
# Module-level worker state and functions for multiprocessing.
#
# Earlier versions dispatched *bound methods* (self.process_pixel etc.) to
# pool.map(). Pickling a bound method pickles the whole instance `self`
# (via `(getattr, (self, name))`), so every task in every chunk re-serialised
# the full cube (and, in Stage 2, the c_map/d_map too — tens of MB each for
# a modest dataset). Measured: this made a "reusable Pool" change a wash or
# even a regression (pool creation is cheap; this per-chunk self-pickling is
# not, and is identical whether the Pool is fresh or reused).
#
# Fix: worker functions are plain module-level functions that read from a
# module-global dict populated in the *parent* process before the Pool is
# forked. With the "fork" start method, forked workers get that state via
# copy-on-write with zero pickling; only the small per-task argument (a
# pixel index/position tuple) travels through the task queue.
# ---------------------------------------------------------------------------
_G: dict = {}


def _g_get_patch(p0: Tuple[int, int]) -> np.ndarray:
    """Module-level equivalent of PACO.get_patch, reading globals set by
    _set_stage1_globals(). Must stay logically identical to PACO.get_patch.
    See utils.extract_patch_window for why this is a small windowed
    extraction rather than a full-frame-mask boolean index."""
    return extract_patch_window(_G["cube"], p0, _G["patch_width"], _G["patch_area"])


def _stage1_worker(p0: tuple) -> tuple:
    """Fused patch extraction + (m, Cinv, c, d) computation for one pixel."""
    patch = _g_get_patch((int(p0[0]), int(p0[1])))
    psf_vectorized = _G["vectorized_psf"]
    T = patch.shape[0]
    m, Cinv = compute_statistics_at_pixel(patch)
    if (m is None) or (Cinv is None) or np.any(np.isnan(patch)):
        return (None, None, np.nan, np.full(T, np.nan))
    w = Cinv @ psf_vectorized
    c = float(psf_vectorized.T @ w)
    d_vals = (patch - m) @ w
    return (m, Cinv, c, d_vals)


def _stage2_worker(args: tuple) -> Tuple[Optional[int], float, float]:
    """Interpolate the precomputed c_map/d_map at one test pixel's trajectory."""
    i, p0, interpolation_order = args
    try:
        angles_px = get_rotated_pixel_coords(_G["patch_y"], _G["patch_x"],
                                             p0, _G["angles"])
        if (np.any(angles_px < 0) or
                np.any(angles_px[:, 0] >= _G["width"]) or
                np.any(angles_px[:, 1] >= _G["height"])):
            return i, np.nan, np.nan

        coords_spatial = np.array([angles_px[:, 1], angles_px[:, 0]])
        c_interpolated = map_coordinates(_G["c_map"], coords_spatial,
                                         order=interpolation_order,
                                         mode='nearest')
        a_val = np.sum(c_interpolated)

        time_coords = np.arange(_G["num_frames"])
        coords_d = np.vstack([coords_spatial, time_coords])
        d_interpolated = map_coordinates(_G["d_map"], coords_d,
                                         order=interpolation_order,
                                         mode='nearest')
        b_val = np.sum(d_interpolated)

        if np.isnan(a_val) or np.isnan(b_val):
            return i, np.nan, np.nan
        return i, a_val, b_val
    except Exception:
        return i, np.nan, np.nan


class FastPACOv2(PACO):
    """
    Memory-efficient PACO implementation based on batch processing and
    spatial interpolation of pre-computed statistics maps.

    Stage 1 builds a c-map (scalar) and d-map (temporal vector) by running
    the statistics once over the full image in manageable batches. Stage 2
    processes each test pixel by interpolating c and d at the rotated
    positions, which avoids re-computing covariances per test point.
    """

    def PACOCalc(self,
                 phi0s: np.ndarray,
                 interpolation_order: int = 1,
                 cpu: int = 1,
                 use_gpu: bool = False,
                 gpu_memory_fraction: Optional[float] = None,
                 memory_threshold_gb: Optional[float] = None,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute (a, b) maps using the two-stage Fast PACO approach.

        Parameters
        ----------
        phi0s : np.ndarray, shape (N, 2)
            Array of (x, y) test positions.
        interpolation_order : int
            Spline interpolation order for map sampling (0=nearest, 1=linear).
        cpu : int
            Number of CPU cores for parallel processing (patch extraction,
            when use_gpu=True; the full per-pixel statistics + c/d
            computation otherwise).
        use_gpu : bool
            Batch Stage 1's covariance/shrinkage/inversion + PSF contraction
            on the GPU (see GPUStatisticsBatcher.compute_stage1_streamed)
            instead of computing it per-pixel on CPU. Falls back to the CPU
            path automatically if no CUDA device is available. Stage 1 is
            the O(N_pixels * (T*K^2 + K^3)) statistics stage that dominates
            wall time for a large cube — separate from any GPU use for PSF
            shifting elsewhere in this codebase.
        gpu_memory_fraction : float, optional
            Overrides `self.gpu_memory_fraction` (see PACO.__init__) for
            this call's GPUStatisticsBatcher.
        memory_threshold_gb : float, optional
            Deprecated; use `memory_fraction` (see PACO.__init__), which
            sets the RAM budget for batch sizing. If given, it overrides
            that budget (in GiB) for this call.

        Returns
        -------
        a : np.ndarray, shape (N,)
            Normalization scalars.
        b : np.ndarray, shape (N,)
            Detection statistics.
        """
        if memory_threshold_gb is None:
            memory_threshold_gb = self._memory_budget_bytes() / 1024**3
        else:
            warnings.warn("memory_threshold_gb is deprecated; use memory_fraction",
                          DeprecationWarning, stacklevel=2)

        npx = len(phi0s)
        H, W = self.height, self.width
        a = np.zeros(npx)
        b = np.zeros(npx)
        # phi0s[:, 0] = row, phi0s[:, 1] = col — keep (row, col) order so that
        # get_patch receives (row, col) as it expects (checked against H and W).
        phi0s_yx = phi0s

        self.normalised_psf = normalize_psf(self.psf, verbose=self.verbose,
                                             force_odd=False, model='airy')
        self.psf_mask = create_boolean_circular_mask(self.normalised_psf.shape,
                                                      radius=self.patch_width // 2)
        self.vectorized_psf = self.normalised_psf[self.psf_mask]
        self.patch_y, self.patch_x = np.meshgrid(
            np.arange(-self.width / 2, self.width / 2),
            np.arange(-self.height / 2, self.height / 2),
        )

        # Stage 1 (statistics -> c_map/d_map) MUST always run on the native
        # H*W pixel grid, regardless of what candidate grid `phi0s` itself
        # is: c/d are per-PIXEL quantities computed once on a fixed grid
        # (FLASSEUR_PACO_NOTES Sec. 7: "grilla fija, sin rotacion") and then
        # spatially INTERPOLATED at each candidate in Stage 2 -- they are
        # never recomputed per candidate. Using `phi0s` directly for Stage 1
        # (as an earlier version of this method did) is only correct when
        # phi0s happens to equal the native grid, i.e. oversampling=1;
        # for oversampling>1, PACO.run() passes a LARGER, finer candidate
        # grid, and reshaping Stage 1's per-candidate results into the
        # native (W, H) c_map/d_map shape then raises a ValueError (a
        # crash, not a silent wrong answer -- confirmed empirically) since
        # the sizes no longer match.
        native_x, native_y = np.meshgrid(np.arange(0, H), np.arange(0, W))
        native_phi0s = np.column_stack((native_x.ravel(), native_y.ravel())).astype(phi0s.dtype)
        n_native = len(native_phi0s)

        batch_size = self._calculate_batch_size(memory_threshold_gb, max(npx, n_native))
        indexed_native_phi0s = np.hstack([native_phi0s, np.arange(n_native).reshape(-1, 1)])
        indexed_phi0s = np.hstack([phi0s_yx, np.arange(npx).reshape(-1, 1)])

        ctx = multiprocessing.get_context("fork")

        # --- Stage 1: build c_map and d_map ---
        if self.verbose:
            print("\n--- STAGE 1: Calculating map statistics in batches ---")

        # Populate globals BEFORE forking: workers inherit this via
        # copy-on-write with zero pickling (vs. re-pickling `self.cube` on
        # every chunk when dispatching a bound method).
        _G["cube"] = self.cube
        _G["patch_width"] = self.patch_width
        _G["patch_area"] = self.patch_area_pixels
        _G["vectorized_psf"] = self.vectorized_psf

        effective_use_gpu = use_gpu and GPU_AVAILABLE
        if use_gpu and not GPU_AVAILABLE and self.verbose:
            print("GPU requested but unavailable (no CuPy/CUDA device) — "
                  "falling back to CPU Stage 1.")
        gpu_batcher = None
        if effective_use_gpu:
            from .gpu import GPUStatisticsBatcher
            effective_gpu_fraction = (gpu_memory_fraction if gpu_memory_fraction is not None
                                      else self.gpu_memory_fraction)
            gpu_batcher = GPUStatisticsBatcher(
                gpu_memory_safety_factor=effective_gpu_fraction, verbose=self.verbose)

        pool1 = ctx.Pool(processes=cpu) if cpu > 1 else None
        try:
            all_results_unordered = []
            for i, indexed_batch in enumerate(self._batch_generator(indexed_native_phi0s, batch_size)):
                if effective_use_gpu:
                    batch_results = self._compute_statistics_batch_gpu(
                        indexed_batch, pool1, gpu_batcher)
                else:
                    batch_results = self._compute_statistics_batch(indexed_batch, pool1)
                all_results_unordered.extend(batch_results)
                if self.verbose:
                    print(f"  Statistics Calculation: Processed batch {i + 1}...")
        finally:
            if pool1 is not None:
                pool1.close()
                pool1.join()

        all_results_ordered = sorted(all_results_unordered, key=lambda x: x[0])
        c_vals = [res[1] for res in all_results_ordered]
        d_vals = [res[2] for res in all_results_ordered]

        self.c_map = np.array(c_vals).reshape((W, H)).T
        self.d_map = np.array(d_vals).reshape((W, H, self.num_frames)).transpose(1, 0, 2)

        # --- Stage 2: per-pixel processing via map interpolation ---
        if self.verbose:
            print("\n--- STAGE 2: Processing pixels with interpolation in batches ---")
        start_time = time.time()

        # c_map/d_map only exist after Stage 1, so a second fork boundary is
        # needed to hand them to workers via copy-on-write (still just 1
        # Pool for the *entire* stage, covering every batch, vs. 1 per batch).
        _G["patch_y"] = self.patch_y
        _G["patch_x"] = self.patch_x
        _G["angles"] = self.angles
        _G["width"] = self.width
        _G["height"] = self.height
        _G["c_map"] = self.c_map
        _G["d_map"] = self.d_map
        _G["num_frames"] = self.num_frames

        pool2 = ctx.Pool(processes=cpu) if cpu > 1 else None
        try:
            for i, indexed_batch in enumerate(self._batch_generator(indexed_phi0s, batch_size)):
                process_args = [
                    (int(row[2]), row[:2], interpolation_order) for row in indexed_batch
                ]

                if pool2 is not None:
                    results = pool2.map(_stage2_worker, process_args)
                else:
                    results = [_stage2_worker(arg) for arg in process_args]

                for i_res, a_val, b_val in results:
                    if i_res is not None:
                        a[i_res] = a_val
                        b[i_res] = b_val

                if self.verbose:
                    print(f"  Pixel Processing: Processed batch {i + 1}...")
        finally:
            if pool2 is not None:
                pool2.close()
                pool2.join()

        if self.verbose:
            print(f"\nProcessing took {time.time() - start_time:.2f} seconds")
        return a, b

    # ------------------------------------------------------------------
    # Stage 1 helpers
    # ------------------------------------------------------------------

    def _compute_statistics_batch(self,
                                   indexed_phi0s_batch: np.ndarray,
                                   pool: Optional["multiprocessing.pool.Pool"]) -> list:
        """
        Run statistics computation for one batch and tag results with their
        original flat index for later reassembly. Dispatches to the
        module-level `_stage1_worker`, which fuses patch extraction and the
        (m, Cinv, c, d) terms into a single per-pixel call reading from
        fork-inherited globals (see module docstring).
        """
        phi0s_only = indexed_phi0s_batch[:, :2]
        original_indices = indexed_phi0s_batch[:, 2].astype(int)

        tasks = [tuple(p) for p in phi0s_only]
        if pool is not None:
            stats_results = pool.map(_stage1_worker, tasks)
        else:
            stats_results = [_stage1_worker(p0) for p0 in tasks]

        return [
            (original_indices[i], stats[2], stats[3])
            for i, stats in enumerate(stats_results)
        ]

    def _compute_statistics_batch_gpu(self,
                                       indexed_phi0s_batch: np.ndarray,
                                       pool: Optional["multiprocessing.pool.Pool"],
                                       batcher: "GPUStatisticsBatcher") -> list:
        """
        GPU-batched equivalent of _compute_statistics_batch: patch extraction
        stays on CPU (parallel via `pool` if given — the same
        fork-inherited `_g_get_patch` module function, cheap per item), but
        the whole batch's (m, Cinv, w, c, d) computation is done in one
        GPUStatisticsBatcher.compute_stage1_streamed call instead of one
        _stage1_worker call per pixel.
        """
        phi0s_only = indexed_phi0s_batch[:, :2]
        original_indices = indexed_phi0s_batch[:, 2].astype(int)

        tasks = [(int(p[0]), int(p[1])) for p in phi0s_only]
        if pool is not None:
            raw_patches = pool.map(_g_get_patch, tasks)
        else:
            raw_patches = [_g_get_patch(t) for t in tasks]

        patches_arr = np.array(raw_patches)  # (batch, T, K), same order as tasks
        c_vals, d_vals = batcher.compute_stage1_streamed(
            patches_arr, _G["vectorized_psf"])

        return [
            (original_indices[i], c_vals[i], d_vals[i])
            for i in range(len(original_indices))
        ]

    # ------------------------------------------------------------------
    # Batch utilities
    # ------------------------------------------------------------------

    def _calculate_batch_size(self, memory_threshold_gb: float, npx: int) -> int:
        """Estimate batch size that fits within the memory budget."""
        K = self.patch_area_pixels
        T = self.num_frames
        mem_per_pixel_bytes = (K * K + T) * 8
        mem_per_patch = K * T * 8
        threshold_bytes = memory_threshold_gb * 1024**3
        batch_size = int(threshold_bytes / ((mem_per_pixel_bytes + mem_per_patch) * 1.5))
        if self.verbose:
            print(f"Calculated batch size: {batch_size} pixels")
        return max(1, min(batch_size, npx))

    def _batch_generator(self,
                          data: np.ndarray,
                          batch_size: int) -> Generator[np.ndarray, None, None]:
        """Yield successive slices of data of length batch_size."""
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
