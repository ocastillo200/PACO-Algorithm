"""
Base PACO class shared by all algorithm variants.

Provides cube management, patch extraction, background statistics,
flux estimation, and the reference PACO algorithm (Algorithm 2, [FLA18]).
"""
import multiprocessing
import os
import sys
import time
from typing import Callable, Dict, Generator, List, Literal, Optional, Tuple, Union

import numpy as np
import psutil
from scipy import ndimage
from scipy.ndimage import filters
from vip_hci.fm import normalize_psf
from vip_hci.metrics.detection import detection
from vip_hci.preproc.rescaling import (cube_px_resampling, frame_px_resampling,
                                        frame_shift)

from .gpu import GPU_AVAILABLE
from .math_utils import al, bl, compute_statistics_at_pixel, wfl
from .utils import create_boolean_circular_mask, extract_patch_window, get_rotated_pixel_coords

# ---------------------------------------------------------------------------
# Module-level worker for compute_statistics_parallel.
#
# The original design dispatched two separate pool_map() calls of *bound*
# methods (self.get_patch, then compute_statistics_at_pixel on the results).
# Pickling a bound method pickles the whole instance `self` (including
# self.cube) per chunk, and the intermediate raw-patch list was serialised
# twice (once out of the first Pool, once into the second) — the same
# pattern that, for FastPACOv2/FullPACOv2, measured as a real bottleneck
# once fixed (see those modules' docstrings for the full history and
# measurements). Fix: a single module-level worker, fused (patch extraction
# + statistics in one call), reading the cube from a global set in the
# parent before the fork — so only the tiny per-task position tuple is
# pickled, not the cube.
# ---------------------------------------------------------------------------
_G: dict = {}


def _g_get_patch(px: Tuple[int, int]) -> np.ndarray:
    """Module-level equivalent of PACO.get_patch, reading globals. See
    utils.extract_patch_window for why this is a small windowed extraction
    rather than a full-frame-mask boolean index."""
    return extract_patch_window(_G["cube"], px, _G["patch_width"], _G["patch_area"])


def _stats_worker(p0: tuple) -> tuple:
    """Fused patch extraction + (m, Cinv) computation for one pixel."""
    patch = _g_get_patch((int(p0[0]), int(p0[1])))
    m, Cinv = compute_statistics_at_pixel(patch)
    return patch, m, Cinv


def _read_int(path: str) -> Optional[int]:
    """Integer content of a cgroup file; None if missing or unlimited."""
    try:
        with open(path) as f:
            value = f.read().strip()
    except OSError:
        return None
    if not value.isdigit():  # cgroup v2 writes "max" when unlimited
        return None
    return int(value)


def cgroup_memory_headroom(proc_root: str = "/proc",
                           cgroup_root: str = "/sys/fs/cgroup") -> Optional[int]:
    """
    Bytes this process can still allocate before reaching a Linux cgroup
    memory limit (container, SLURM job, systemd scope), or None if no limit
    applies (no cgroups, e.g. macOS, or unlimited).

    The limit is often set on an ancestor of the process's own cgroup (SLURM
    puts it on the job while tasks run in child cgroups), so every level up
    to the root is checked and the smallest headroom (limit - usage) is
    returned. Usage includes reclaimable page cache, so this errs on the
    safe side. Both cgroup v2 (memory.max / memory.current) and v1
    (memory.limit_in_bytes / memory.usage_in_bytes) are handled. The
    arguments exist so tests can point at a fake hierarchy.
    """
    try:
        with open(os.path.join(proc_root, "self", "cgroup")) as f:
            lines = f.read().splitlines()
    except OSError:
        return None

    candidates = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, rel = parts
        if controllers == "":
            # cgroup v2 unified hierarchy
            base, limit_file, usage_file = cgroup_root, "memory.max", "memory.current"
        elif "memory" in controllers.split(","):
            # cgroup v1 memory controller
            base, limit_file, usage_file = (os.path.join(cgroup_root, "memory"),
                                            "memory.limit_in_bytes",
                                            "memory.usage_in_bytes")
        else:
            continue
        rel = rel.strip("/")
        while True:
            directory = os.path.join(base, rel) if rel else base
            limit = _read_int(os.path.join(directory, limit_file))
            # v1 reports "unlimited" as a huge number (~2^63)
            if limit is not None and limit < 2 ** 60:
                usage = _read_int(os.path.join(directory, usage_file)) or 0
                candidates.append(max(limit - usage, 0))
            if not rel:
                break
            rel = os.path.dirname(rel)
    return min(candidates) if candidates else None


class PACO:
    """
    Base class for the PACO family of algorithms (Flasseur et al. 2018).

    Subclasses override :meth:`PACOCalc` to implement different
    computational strategies (batch, full, GPU-accelerated, etc.).
    :meth:`run` orchestrates the full pipeline and is common to all variants.
    """

    def __init__(self,
                 cube: np.ndarray,
                 angle_list: np.ndarray,
                 psf: np.ndarray,
                 dit_psf: float = 1.0,
                 dit_science: float = 1.0,
                 nd_transmission: float = 1.0,
                 fwhm: float = 4.0,
                 rescaling_factor: float = 1.0,
                 pixscale: float = 0.01225,
                 use_gpu: bool = False,
                 full_output: bool = False,
                 patch_radius_dec: int = 0,
                 flux_normalized: float = 1.0,
                 gpu_memory_fraction: float = 0.8,
                 memory_fraction: float = 0.8,
                 verbose: bool = False) -> None:
        """
        Parameters
        ----------
        cube : np.ndarray, shape (T, H, W)
            Science data cube.
        angle_list : np.ndarray, shape (T,)
            Parallactic angles in degrees.
        psf : np.ndarray
            Unsaturated PSF image (2-D or 3-D; 3-D is collapsed by median).
        dit_psf : float
            Integration time of the PSF observation.
        dit_science : float
            Integration time of the science frames.
        nd_transmission : float
            Neutral-density filter transmission used during PSF acquisition.
        fwhm : float or array-like
            Full width at half maximum of the PSF in pixels.
        rescaling_factor : float
            Spatial rescaling factor applied before processing.
        pixscale : float
            Pixel scale in arcseconds per pixel.
        use_gpu : bool
            Enable GPU acceleration where available.
        full_output : bool
            If True, :meth:`run` returns (snr, flux) instead of snr only.
        patch_radius_dec : int
            Reduction applied to the default patch radius (patch_width = 2*fwhm - patch_radius_dec).
        flux_normalized : float
            Normalisation constant for flux estimates.
        gpu_memory_fraction : float
            Fraction of *currently free* VRAM (re-read on every GPU batch,
            not just once) that GPUStatisticsBatcher may use at a time
            (see src.gpu.GPUBatchDriver). Same role as `memory_fraction`
            but for VRAM instead of host RAM; lower it on a shared GPU
            server where other processes also hold VRAM. Override per call
            via PACOCalc(..., gpu_memory_fraction=...).
        memory_fraction : float
            Fraction of the host RAM that is *available* when a run starts
            (psutil.virtual_memory().available) that the run may use. This
            is the single RAM budget: every large allocation (FullPACOv2's
            shared statistics cache and sweep batches, FastPACOv2's Stage 1
            batches) is sized from it, so the field size changes the run
            time, not whether the run fits. Lower it on a machine shared
            with other jobs. Must be in (0, 1].
        verbose : bool
            Print progress information.
        """
        self.cube = cube
        self.num_frames = cube.shape[0]
        self.width = cube.shape[2]
        self.height = cube.shape[1]

        self.angles = angle_list
        self.rescaling_factor = rescaling_factor
        self.full_output = full_output
        self.pixscale = pixscale
        self.flux_normalized = flux_normalized

        if isinstance(fwhm, (list, np.ndarray)):
            self.fwhm = np.median(fwhm)
        else:
            self.fwhm = fwhm

        if len(psf.shape) > 2:
            psf = np.nanmedian(psf, axis=0)
        self.psf = psf * dit_science / dit_psf / nd_transmission
        self.dit_science = dit_science
        self.dit_psf = dit_psf

        self.patch_width = int(2 * self.fwhm) - patch_radius_dec

        self.verbose = verbose
        self.use_gpu = use_gpu
        self.gpu_memory_fraction = gpu_memory_fraction
        if not 0.0 < memory_fraction <= 1.0:
            raise ValueError(f"memory_fraction must be in (0, 1], got {memory_fraction}")
        self.memory_fraction = memory_fraction

        mask = create_boolean_circular_mask(self.cube[0].shape,
                                            radius=self.patch_width // 2)
        self.patch_area_pixels = int(self.cube[0][mask].ravel().shape[0])

        self.snr = None
        self.flux = None
        self.std = None

        if self.verbose:
            print("---------------------- ")
            print("Summary of PACO setup: \n")
            print(f"Image Cube shape = {self.cube.shape}")
            print("PSF | Area | Rad | Width | ")
            print(f" | {self.patch_area_pixels:01} |"
                  f" {self.patch_width // 2:02} | {self.psf.shape[0]:03} | ")
            print(f"Patch width: {self.patch_width}")
            print("---------------------- \n")
            sys.stdout.flush()

    def _memory_budget_bytes(self) -> int:
        """
        The run's host-RAM budget: `memory_fraction` of the RAM available
        right now. Read once at the start of PACOCalc, so allocations made
        earlier (e.g. the cube itself) are already excluded.

        Inside a container or a batch-scheduler job (e.g. SLURM), the
        process may be confined by a cgroup memory limit well below the
        host's free RAM, and exceeding it gets the job killed; the
        available memory is then the smaller of the two (see
        cgroup_memory_headroom).
        """
        available = psutil.virtual_memory().available
        headroom = cgroup_memory_headroom()
        source = "host"
        if headroom is not None and headroom < available:
            available, source = headroom, "cgroup limit"
        budget = int(available * self.memory_fraction)
        if self.verbose:
            print(f"RAM budget: {budget / 1024**3:.2f} GiB "
                  f"({self.memory_fraction:.0%} of available, {source})")
        return budget

    # ------------------------------------------------------------------
    # Data accessors
    # ------------------------------------------------------------------

    def set_cube(self, cube: np.ndarray) -> None:
        self.cube = cube
        self.num_frames, self.height, self.width = cube.shape

    def set_psf(self, psf: np.ndarray) -> None:
        self.psf = psf

    def set_angles(self, angles: np.ndarray) -> None:
        self.angles = angles

    def rescale_cube_and_psf(self, imlib: str = 'vip-fft',
                              interpolation: str = 'lanczos4',
                              keep_center: bool = True) -> None:
        """Spatially rescale the cube and PSF by :attr:`rescaling_factor`."""
        if self.rescaling_factor == 1:
            return
        cube_px_resampling(self.cube, self.rescaling_factor, imlib=imlib,
                           interpolation=interpolation, keep_center=keep_center,
                           verbose=False)
        self.pixscale = self.pixscale / self.rescaling_factor
        self.fwhm = int(self.fwhm * self.rescaling_factor)
        if self.psf is not None:
            self.psf = frame_px_resampling(self.psf, self.rescaling_factor,
                                           imlib=imlib, interpolation=interpolation,
                                           keep_center=keep_center, verbose=False)
        mask = create_boolean_circular_mask(self.psf.shape, self.fwhm)
        self.patch_area_pixels = self.psf[mask].shape[0]
        self.patch_width = 2 * int(self.fwhm)

    # ------------------------------------------------------------------
    # Patch extraction
    # ------------------------------------------------------------------

    def get_patch(self,
                  px: Tuple[int, int],
                  width: Optional[int] = None,
                  mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract the temporal column of patches centred at pixel px.

        Parameters
        ----------
        px : tuple of int
            (row, col) pixel position.
        width : int, optional
            Patch diameter in pixels. Defaults to :attr:`patch_width`.
        mask : np.ndarray, optional
            Pre-computed circular mask. Computed on-the-fly if not provided.

        Returns
        -------
        np.ndarray, shape (T, patch_area_pixels)
            Flattened patch pixels across time frames. Filled with NaN for
            out-of-bounds positions.
        """
        if width is None:
            width = self.patch_width
        # See utils.extract_patch_window: a windowed extraction, not a
        # full-frame-mask boolean index over the whole cube (measured
        # ~2200x faster on a 1028x1028 cube; identical result either way).
        return extract_patch_window(self.cube, px, width,
                                    self.patch_area_pixels,
                                    full_frame_mask=mask)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self,
            cube: Optional[np.ndarray] = None,
            angle_list: Optional[np.ndarray] = None,
            cpu: int = 1,
            imlib: str = 'vip-fft',
            interpolation: str = 'lanczos4',
            keep_center: bool = True,
            verbose: Optional[bool] = None,
            use_gpu: bool = False,
            full_output: Optional[bool] = False,
            use_subpixel_psf_astrometry: bool = True,
            oversampling: int = 1) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Run PACO on the stored (or provided) cube.

        Parameters
        ----------
        cube : np.ndarray, optional
            Override the cube stored at construction time.
        angle_list : np.ndarray, optional
            Override the parallactic angles.
        cpu : int
            Number of CPU cores for parallel statistics computation.
        imlib : str
            Image library used when rescaling (passed to VIP).
        interpolation : str
            Interpolation scheme used when rescaling (passed to VIP).
        keep_center : bool
            Preserve the image centre during rescaling.
        verbose : bool, optional
            Override the verbosity flag set at construction.
        use_gpu : bool
            Enable GPU acceleration (requires CuPy).
        full_output : bool, optional
            Return (snr_map, flux_map) instead of snr_map only.
        use_subpixel_psf_astrometry : bool
            Shift the PSF template to the exact sub-pixel rotated position.
        oversampling : int
            Detection-grid oversampling factor s (s=1 is native resolution).

        Returns
        -------
        snr : np.ndarray
            Signal-to-noise ratio map b / sqrt(a).
        flux : np.ndarray, optional
            Flux estimate map b / a (only when full_output is True).
        """
        if verbose is not None:
            self.verbose = verbose
        if cube is not None and angle_list is not None:
            self.set_cube(cube)
            self.set_angles(angle_list)
        if full_output is not None:
            self.full_output = full_output

        if self.rescaling_factor != 1:
            self.rescale_cube_and_psf(imlib=imlib, interpolation=interpolation,
                                      keep_center=keep_center)

        x_range = np.arange(0, self.height, 1 / oversampling)
        y_range = np.arange(0, self.width, 1 / oversampling)
        x, y = np.meshgrid(x_range, y_range)
        phi0s = np.column_stack((x.flatten(), y.flatten()))

        if self.verbose:
            print(f"Running PACO with oversampling s={oversampling}. "
                  f"Total test points: {len(phi0s)}")

        a, b = self.PACOCalc(np.array(phi0s), cpu=cpu, use_gpu=use_gpu)

        out_h = len(x_range)
        out_w = len(y_range)
        # phi0s were generated in column-major order (all rows per col); reshape to
        # (W, H) first, then transpose to standard image convention (H, W).
        a = np.reshape(a, (out_w, out_h)).T
        b = np.reshape(b, (out_w, out_h)).T

        snr = b / np.sqrt(a)
        flux = b / a

        self.a_map = a
        self.b_map = b
        self.flux_map = b / a       # b/a consistent with flux variable above
        self.snr = snr
        self.flux = flux
        self.std = 1 / np.sqrt(a)

        if self.full_output:
            return snr, flux
        return snr

    # ------------------------------------------------------------------
    # Reference PACO algorithm (Algorithm 1, Flasseur et al. 2018)
    # ------------------------------------------------------------------

    def PACOCalc(self,
                 phi0s: np.ndarray,
                 use_subpixel_psf_astrometry: bool = True,
                 cpu: int = 1,
                 use_gpu: bool = False,
                 gpu_memory_fraction: Optional[float] = None,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute (a, b) maps for the standard PACO algorithm.

        Pre-computes background statistics for the full image, then sweeps
        over all test positions to accumulate the a and b scalars used to
        form the SNR map (SNR = b / sqrt(a)).

        Subclasses override this method with faster or GPU-accelerated
        implementations. The signature intentionally accepts **kwargs to
        allow drop-in replacement by subclass implementations.

        Parameters
        ----------
        phi0s : np.ndarray, shape (N, 2)
            Array of (x, y) test positions.
        use_subpixel_psf_astrometry : bool
            Shift the PSF model to the precise sub-pixel rotated position.
        cpu : int
            Number of CPU cores for the statistics pre-computation stage
            (also used for CPU-side patch extraction when use_gpu=True).
        use_gpu : bool
            Batch the covariance/shrinkage/inversion stage on the GPU (see
            compute_statistics_gpu) instead of computing it on CPU. Falls
            back to the CPU path automatically if no CUDA device is
            available. This is the stage that dominates wall time for large
            cubes (big T or K), as opposed to the PSF-shift preprocessing
            GPUFrameShifter targets.

        Returns
        -------
        a : np.ndarray, shape (N,)
            Normalization scalars (Equation 15 of [FLA18]).
        b : np.ndarray, shape (N,)
            Detection statistics (Equation 16 of [FLA18]).
        """
        npx = len(phi0s)

        a = np.zeros(npx)
        b = np.zeros(npx)
        # phi0s[:, 0] = row, phi0s[:, 1] = col — keep (row, col) so that
        # get_patch and the statistics arrays are indexed correctly.

        start_time = time.time()
        if use_gpu and GPU_AVAILABLE:
            Cinv, m, patches = self.compute_statistics_gpu(
                phi0s, cpu=cpu, gpu_memory_fraction=gpu_memory_fraction)
        elif cpu == 1:
            Cinv, m, patches = self.compute_statistics(phi0s)
        else:
            Cinv, m, patches = self.compute_statistics_parallel(phi0s, cpu=cpu)
        if self.verbose:
            print(f"Statistics computation took {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        normalised_psf = normalize_psf(self.psf, verbose=self.verbose, force_odd=False)
        psf_mask = create_boolean_circular_mask(normalised_psf.shape, radius=self.patch_width // 2)
        if self.verbose:
            print(f"PSF normalisation took {time.time() - start_time:.2f} seconds")

        x, y = np.meshgrid(np.arange(-self.width / 2, self.width / 2),
                           np.arange(-self.height / 2, self.height / 2))

        if self.verbose:
            print("Running PACO sweep...")

        start_time = time.time()
        for i, p0 in enumerate(phi0s):
            angles_px = get_rotated_pixel_coords(x, y, p0, self.angles)
            # angles_px[:, 0] = col coords, angles_px[:, 1] = row coords
            if (np.any(angles_px[:, 0] < 0) or np.any(angles_px[:, 0] >= self.width) or
                    np.any(angles_px[:, 1] < 0) or np.any(angles_px[:, 1] >= self.height)):
                a[i] = np.nan
                b[i] = np.nan
                continue

            Cinlst, mlst, hlst, patch = [], [], [], []
            for l, ang in enumerate(angles_px):
                # ang[0]=col, ang[1]=row; Cinv/m/patches are indexed [row, col]
                row_i, col_i = int(ang[1]), int(ang[0])
                Cinlst.append(Cinv[row_i, col_i])
                mlst.append(m[row_i, col_i])
                if use_subpixel_psf_astrometry:
                    offax = frame_shift(normalised_psf,
                                        ang[1] - row_i,   # shift_y = row fraction
                                        ang[0] - col_i,   # shift_x = col fraction
                                        imlib='vip-fft',
                                        interpolation='lanczos4',
                                        border_mode='reflect')[psf_mask]
                else:
                    offax = normalised_psf[psf_mask]
                hlst.append(offax)
                patch.append(patches[row_i, col_i, l])

            wlst = wfl(hlst, Cinlst)
            a[i] = al(hlst, wlst)
            b[i] = bl(wlst, patch, mlst)

        if self.verbose:
            print(f"PACO sweep took {time.time() - start_time:.2f} seconds")
        return a, b

    # ------------------------------------------------------------------
    # Statistics pre-computation (serial and parallel)
    # ------------------------------------------------------------------

    def compute_statistics(
        self, phi0s: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute per-pixel background statistics (serial).

        For every position in phi0s, extracts the temporal patch column and
        computes the sample mean and shrinkage-regularised inverse covariance.

        Parameters
        ----------
        phi0s : np.ndarray
            Array of (row, col) pixel positions.

        Returns
        -------
        Cinv : np.ndarray, shape (H, W, K, K)
            Inverse covariance matrices.
        m : np.ndarray, shape (H, W, K)
            Temporal means.
        patch : np.ndarray, shape (H, W, T, K)
            Raw patch columns.
        """
        if self.verbose:
            print("Precomputing Statistics...")

        patch = np.zeros(
            (self.height, self.width, self.num_frames, self.patch_area_pixels))
        m = np.zeros((self.height, self.width, self.patch_area_pixels))
        Cinv = np.zeros(
            (self.height, self.width, self.patch_area_pixels, self.patch_area_pixels))

        for p0 in phi0s:
            apatch = self.get_patch(p0)
            r, c = int(p0[0]), int(p0[1])
            m[r][c], Cinv[r][c] = compute_statistics_at_pixel(apatch)
            patch[r][c] = apatch

        return Cinv, m, patch

    def compute_statistics_parallel(
        self, phi0s: np.ndarray, cpu: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute per-pixel background statistics in parallel.

        Parameters
        ----------
        phi0s : np.ndarray
            Array of (row, col) pixel positions.
        cpu : int
            Number of worker processes.

        Returns
        -------
        Cinv : np.ndarray, shape (H, W, K, K)
            Inverse covariance matrices.
        m : np.ndarray, shape (H, W, K)
            Temporal means.
        patches : np.ndarray, shape (H, W, T, K)
            Raw patch columns.
        """
        if self.verbose:
            print(f"Precomputing Statistics using {cpu} processes...")

        start = time.time()

        # Populate globals BEFORE forking so workers inherit the cube via
        # copy-on-write (zero pickling) instead of it being re-pickled with
        # `self` on every chunk dispatched to a bound method.
        _G["cube"] = self.cube
        _G["patch_width"] = self.patch_width
        _G["patch_area"] = self.patch_area_pixels

        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=cpu) as pool:
            results = pool.map(_stats_worker, [tuple(p0) for p0 in phi0s])

        if self.verbose:
            print(f"Statistics computation took {time.time() - start:.2f} seconds")

        raw_patches = [r[0] for r in results]
        ms, cs = [], []
        for _, m_i, Cinv_i in results:
            if m_i is None or Cinv_i is None:
                ms.append(np.full(self.patch_area_pixels, np.nan))
                cs.append(np.full((self.patch_area_pixels, self.patch_area_pixels),
                                  np.nan))
            else:
                ms.append(m_i)
                cs.append(Cinv_i)

        ms = np.array(ms)
        cs = np.array(cs)
        patches = np.array(raw_patches)

        # Results are in column-major order: for each col (axis-0=W), all rows (axis-1=H).
        # Reshape to (W, H, ...) then swapaxes to (H, W, ...) = standard image convention.
        patches = patches.reshape(
            (self.width, self.height, self.num_frames, self.patch_area_pixels))
        m = ms.reshape((self.width, self.height, self.patch_area_pixels))
        Cinv = cs.reshape(
            (self.width, self.height, self.patch_area_pixels, self.patch_area_pixels))

        patches = np.swapaxes(patches, 0, 1)
        m = np.swapaxes(m, 0, 1)
        Cinv = np.swapaxes(Cinv, 0, 1)

        return Cinv, m, patches

    def compute_statistics_gpu(
        self, phi0s: np.ndarray, cpu: int = 1,
        gpu_memory_fraction: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute per-pixel background statistics with the covariance +
        shrinkage + inversion stage (Equations 7, 12 of [FLA18]) batched on
        the GPU.

        Patch extraction (reading (T, K) values out of the cube for each of
        the H*W positions — cheap per item, embarrassingly parallel) still
        happens on the CPU, using the same fork-inherited-globals pattern as
        compute_statistics_parallel. Extracted patches are then handed to
        GPUStatisticsBatcher in VRAM-sized chunks for the O(T*K^2 + K^3)
        per-pixel math, which is what actually dominates wall time for a
        large cube (big T or K). This CPU-extraction / GPU-math split is the
        hybrid architecture: CPU processes for the cheap-per-item work, GPU
        batching for the expensive-per-item linear algebra.

        Silently falls back to the CPU path if GPU_AVAILABLE is False (no
        CuPy/CUDA device) — callers do not need to branch on GPU
        availability themselves; see PACOCalc's use_gpu flag.

        Parameters
        ----------
        phi0s : np.ndarray
            Array of (row, col) pixel positions.
        cpu : int
            Worker processes for the CPU-side patch-extraction stage.
        gpu_memory_fraction : float, optional
            Overrides `self.gpu_memory_fraction` for this call (see
            PACO.__init__).

        Returns
        -------
        Cinv : np.ndarray, shape (H, W, K, K)
        m : np.ndarray, shape (H, W, K)
        patches : np.ndarray, shape (H, W, T, K)
            Same layout as compute_statistics_parallel.
        """
        if not GPU_AVAILABLE:
            if self.verbose:
                print("GPU requested but unavailable (no CuPy/CUDA device) — "
                      "falling back to CPU statistics computation.")
            return (self.compute_statistics_parallel(phi0s, cpu=cpu) if cpu > 1
                    else self.compute_statistics(phi0s))

        from .gpu import GPUStatisticsBatcher

        if self.verbose:
            print("Precomputing statistics with GPU-batched "
                  "covariance/shrinkage/inversion...")
        start = time.time()

        if cpu > 1:
            _G["cube"] = self.cube
            _G["patch_width"] = self.patch_width
            _G["patch_area"] = self.patch_area_pixels
            ctx = multiprocessing.get_context("fork")
            with ctx.Pool(processes=cpu) as pool:
                raw_patches = pool.map(
                    _g_get_patch, [(int(p0[0]), int(p0[1])) for p0 in phi0s])
        else:
            raw_patches = [self.get_patch(p0) for p0 in phi0s]

        # Same input order as compute_statistics_parallel (pool.map/list
        # comprehension both preserve order) — the column-major reshape
        # below relies on that, exactly as it does there.
        patches_arr = np.array(raw_patches)  # (npx, T, K)

        effective_fraction = (gpu_memory_fraction if gpu_memory_fraction is not None
                              else self.gpu_memory_fraction)
        batcher = GPUStatisticsBatcher(gpu_memory_safety_factor=effective_fraction,
                                       verbose=self.verbose)
        m_flat, Cinv_flat = batcher.compute_stats_streamed(patches_arr)

        if self.verbose:
            print(f"GPU statistics computation took "
                  f"{time.time() - start:.2f} seconds")

        patches = patches_arr.reshape(
            (self.width, self.height, self.num_frames, self.patch_area_pixels))
        m = m_flat.reshape((self.width, self.height, self.patch_area_pixels))
        Cinv = Cinv_flat.reshape(
            (self.width, self.height, self.patch_area_pixels, self.patch_area_pixels))

        patches = np.swapaxes(patches, 0, 1)
        m = np.swapaxes(m, 0, 1)
        Cinv = np.swapaxes(Cinv, 0, 1)

        return Cinv, m, patches

    # ------------------------------------------------------------------
    # Flux estimation (Algorithm 3, Flasseur et al. 2018)
    # ------------------------------------------------------------------

    def estimate_flux_at(self,
                         phi0: Tuple[int, int],
                         initial_est: float = 0.0,
                         eps: float = 0.01,
                         max_iter: int = 50) -> Tuple[float, float, float, list]:
        """
        Iterative unbiased flux estimate for a single point source.

        Implements Algorithm 3 of Flasseur et al. 2018. The background
        statistics are recomputed at each iteration after subtracting the
        current flux estimate so that the companion signal does not bias them.

        Parameters
        ----------
        phi0 : tuple of int
            (x, y) position of the candidate companion.
        initial_est : float
            Starting flux estimate (default 0, i.e. detection threshold).
        eps : float
            Convergence threshold as a fraction of the current estimate.
        max_iter : int
            Maximum number of iterations.

        Returns
        -------
        final_flux : float
            Normalised flux estimate.
        final_std_norm : float
            Normalised 1-sigma uncertainty.
        norm : float
            Normalisation constant used.
        convergence_history : list of float
            Flux estimate at each iteration.
        """
        if self.verbose:
            print(f"\nEstimating unbiased flux for source at {phi0}...")

        norm = self.flux_normalized
        normalised_psf = self.psf
        psf_mask = create_boolean_circular_mask(normalised_psf.shape,
                                                radius=self.patch_width // 2)

        grid_x, grid_y = np.meshgrid(np.arange(-self.width / 2, self.width / 2),
                                      np.arange(-self.height / 2, self.height / 2))

        p0_yx = (phi0[1], phi0[0])
        angles_px = get_rotated_pixel_coords(grid_x, grid_y, p0_yx, self.angles)

        h_on, r_track_original = [], []
        for l, ang in enumerate(angles_px):
            ang_int = ang.astype(int)
            ang_subpix = ang - ang_int
            psf_shifted = frame_shift(normalised_psf,
                                      shift_y=ang_subpix[1],   # row fraction
                                      shift_x=ang_subpix[0])[psf_mask]  # col fraction
            h_on.append(psf_shifted)
            patch_col = self.get_patch((ang_int[1], ang_int[0]))
            r_track_original.append(
                patch_col[l] if patch_col is not None
                else np.full(self.patch_area_pixels, np.nan)
            )

        h_on = np.array(h_on)
        r_track_original = np.array(r_track_original)

        ahat = initial_est
        aprev = np.inf
        iter_count = 0
        convergence_history = []

        while np.abs(ahat - aprev) > ahat * eps and iter_count < max_iter:
            aprev = ahat
            iter_count += 1
            a_total, b_total = 0.0, 0.0
            stats_cache = {}

            for l, ang in enumerate(angles_px):
                theta_l = tuple(ang.astype(int))
                if theta_l in stats_cache:
                    m_u, Cinv_u = stats_cache[theta_l]
                else:
                    observed_patches = self.get_patch((theta_l[1], theta_l[0]))
                    if observed_patches is None or np.all(np.isnan(observed_patches)):
                        stats_cache[theta_l] = (None, None)
                        continue

                    contaminant_signal = np.zeros_like(observed_patches)
                    for l_prime, ang_prime in enumerate(angles_px):
                        shift_y = ang_prime[0] - theta_l[0]
                        shift_x = ang_prime[1] - theta_l[1]
                        contaminant_psf = frame_shift(normalised_psf,
                                                      shift_x, shift_y)[psf_mask]
                        contaminant_signal[l_prime, :] = contaminant_psf

                    unbiased_patches = observed_patches - ahat * contaminant_signal
                    m_u, Cinv_u = compute_statistics_at_pixel(unbiased_patches)
                    stats_cache[theta_l] = (m_u, Cinv_u)

                current_r = r_track_original[l]
                current_h = h_on[l]

                if m_u is None or Cinv_u is None or np.any(np.isnan(current_r)):
                    continue

                w_l = Cinv_u @ current_h
                a_l = current_h.T @ w_l
                b_l = w_l.T @ (current_r - m_u)
                a_total += a_l
                b_total += b_l

            ahat = max(0.0, b_total) / a_total if a_total > 1e-8 else 0.0
            convergence_history.append(ahat / norm)

            if self.verbose:
                print(f"  Iter {iter_count}: a={a_total:.2f}, b={b_total:.2f}, "
                      f"flux_est={ahat / norm:.2f}")

        final_std = 1 / np.sqrt(a_total) if a_total > 1e-8 else np.inf
        final_flux = ahat / norm
        final_std_norm = final_std / norm

        if self.verbose:
            print(f"Final: flux = {final_flux:.2f} ± {final_std_norm:.2f}")

        return final_flux, final_std_norm, norm, convergence_history

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def subpixel_threshold_detect(self,
                                   snr_map: np.ndarray,
                                   threshold: float,
                                   mode: str = 'lpeaks',
                                   bkg_sigma: float = 5.0,
                                   matched_filter: bool = False,
                                   mask: bool = True,
                                   full_output: bool = False,
                                   cpu: int = 1) -> np.ndarray:
        """
        Detect sources above a threshold using VIP blob-finding algorithms.

        Wraps :func:`vip_hci.metrics.detection.detection`. Refer to that
        function's documentation for a full description of each parameter.
        Note that this wrapper returns (x, y) instead of (y, x).

        Parameters
        ----------
        snr_map : np.ndarray, 2-D
            Input SNR frame.
        threshold : float
            Minimum SNR for a detection.
        mode : str
            Detection algorithm: 'lpeaks', 'log', 'dog', 'snrmap', or 'snrmapf'.
        bkg_sigma : float
            Background sigma clipping level (used by lpeaks, dog, log).
        matched_filter : bool
            Cross-correlate with the PSF before peak finding.
        mask : bool
            Mask the inner 2*FWHM region of the image.
        full_output : bool
            Return a full detection table instead of just coordinates.
        cpu : int
            Number of processes for SNR map computation.

        Returns
        -------
        peaks : np.ndarray
            (x, y) coordinates of detections above the threshold.
        """
        peaks = detection(snr_map,
                          fwhm=self.fwhm,
                          psf=self.psf / np.nanmax(self.psf),
                          mode=mode,
                          bkg_sigma=bkg_sigma,
                          matched_filter=matched_filter,
                          mask=mask,
                          snr_thresh=threshold,
                          nproc=cpu,
                          plot=False,
                          debug=False,
                          full_output=full_output,
                          verbose=self.verbose)
        if full_output:
            return peaks.T
        return peaks

    def pixel_threshold_detection(self,
                                   snr_map: np.ndarray,
                                   threshold: float) -> np.ndarray:
        """
        Return pixel coordinates of local maxima above a given SNR threshold.

        Parameters
        ----------
        snr_map : np.ndarray, 2-D
            SNR map produced by :meth:`run`.
        threshold : float
            Minimum SNR value for detection.

        Returns
        -------
        np.ndarray, shape (N, 2)
            Array of (x, y) pixel coordinates for each detection.
        """
        data_max = filters.maximum_filter(snr_map, size=self.fwhm)
        maxima = snr_map == data_max
        maxima[data_max <= threshold] = 0

        labeled, _ = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        x, y = [], []
        for dy, dx in slices:
            x.append((dx.start + dx.stop - 1) / 2)
            y.append((dy.start + dy.stop - 1) / 2)
        return np.array(list(zip(x, y)))
