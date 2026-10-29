"""
GPU backend detection and GPU-accelerated frame shifting / statistics
utilities.
"""
import time
import warnings
from typing import Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import tqdm

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp  # type: ignore[no-redef]
    GPU_AVAILABLE = False


class GPUBatchDriver:
    """
    Unified VRAM-budgeted chunking driver for batched GPU work.

    Replaces the batch-size arithmetic that used to be duplicated,
    slightly differently, in both ``GPUFrameShifter`` and
    ``GPUStatisticsBatcher``: one formula for turning
    (bytes-per-item, safety factor, free VRAM) into a batch size, shared
    by both classes.

    Two robustness gaps this closes relative to the old per-class
    formulas:

    1. ``estimate_batch_size`` re-reads ``cp.cuda.Device().mem_info``
       on every call instead of relying on a value cached once at
       construction -- on a shared server, free VRAM at the start of a
       multi-hour run is not representative of free VRAM partway
       through (another process may have claimed or released memory).
    2. ``run`` retries a chunk that raises
       ``cupy.cuda.memory.OutOfMemoryError`` by halving its size (up to
       `max_retries` times) before giving up on the GPU for that chunk
       specifically. If the chunk still doesn't fit after retries, or
       any other CUDA error occurs, only that chunk falls back to the
       caller-supplied CPU path (with a warning) -- a transient VRAM
       pressure spike or an unexpected device error can no longer lose
       an entire multi-hour run.
    """

    def __init__(self, gpu_memory_safety_factor: float = 0.8,
                 max_retries: int = 3, verbose: bool = False):
        self.gpu_memory_safety_factor = gpu_memory_safety_factor
        self.max_retries = max_retries
        self.verbose = verbose

    def free_usable_bytes(self) -> float:
        """Currently free VRAM (fresh ``mem_info`` read, not cached) times
        the safety factor. Exposed separately from `estimate_batch_size`
        so a caller with its own fixed reservation on top (e.g.
        GPUFrameShifter reserving space for a PSF already resident) can
        subtract that before dividing by its own per-item cost."""
        free_bytes, _total_bytes = cp.cuda.Device().mem_info
        return free_bytes * self.gpu_memory_safety_factor

    def estimate_batch_size(self, bytes_per_item: float, n_items: int) -> int:
        """Batch size fitting `bytes_per_item` into a safety-factor
        fraction of CURRENTLY free VRAM (not a cached snapshot)."""
        if n_items <= 0:
            return 1
        if bytes_per_item <= 0:
            return n_items
        usable = self.free_usable_bytes()
        batch_size = max(1, int(usable / bytes_per_item))
        return min(batch_size, n_items)

    def run(self, n_items: int, bytes_per_item: float,
            gpu_fn: Callable[[int, int], None],
            cpu_fn: Callable[[int, int], None]) -> None:
        """
        Process `n_items` in VRAM-budgeted chunks, calling
        `gpu_fn(start, end)` for each -- it must write its results for
        items [start, end) into the caller's own output arrays (closed
        over), and raise on failure. On
        ``cupy.cuda.memory.OutOfMemoryError`` the chunk is halved and
        retried (up to `max_retries` times); if it still fails, or any
        other exception is raised, `cpu_fn(start, end)` (same contract,
        on CPU) is called for that chunk instead, with a warning -- the
        rest of the run is unaffected.
        """
        start = 0
        while start < n_items:
            chunk = self.estimate_batch_size(bytes_per_item, n_items - start)
            end = start + chunk
            attempt = 0
            succeeded = False
            while attempt <= self.max_retries:
                try:
                    gpu_fn(start, end)
                    succeeded = True
                    break
                except cp.cuda.memory.OutOfMemoryError:
                    cp.get_default_memory_pool().free_all_blocks()
                    shrunk = max(1, (end - start) // 2)
                    if shrunk == (end - start):
                        break  # already down to 1 item -- no smaller chunk to try
                    if self.verbose:
                        print(f"  [GPUBatchDriver] OOM on [{start}:{end}), "
                              f"retrying with chunk size {shrunk} "
                              f"(attempt {attempt + 1}/{self.max_retries})")
                    end = start + shrunk
                    attempt += 1
                except Exception as exc:
                    warnings.warn(
                        f"[GPUBatchDriver] GPU chunk [{start}:{end}) failed "
                        f"({exc!r}); falling back to CPU for this chunk only.")
                    break
            if not succeeded:
                if attempt > self.max_retries or (attempt > 0 and not succeeded):
                    warnings.warn(
                        f"[GPUBatchDriver] chunk [{start}:{end}) would not "
                        f"fit on GPU after {attempt} OOM retries; falling "
                        f"back to CPU for this chunk only.")
                cpu_fn(start, end)
            start = end


class GPUFrameShifter:
    """GPU-accelerated frame shifting with batch processing and memory optimization."""

    def __init__(self, psf: np.ndarray, psf_mask: np.ndarray, verbose: bool = False):
        """
        Parameters
        ----------
        psf : np.ndarray
            Point spread function array.
        psf_mask : np.ndarray
            Boolean mask selecting the PSF support region.
        verbose : bool
            Print progress information.
        """
        self.psf = psf
        self.psf_mask = psf_mask
        self.verbose = verbose
        self.device = cp.cuda.Device()
        self.stream = cp.cuda.Stream()

        self.gpu_memory_total = self.device.mem_info[1]
        self.gpu_memory_safety_factor = 0.8
        self._batch_driver = GPUBatchDriver(
            gpu_memory_safety_factor=self.gpu_memory_safety_factor,
            verbose=verbose)

        self.gpu_psf = cp.asarray(psf)
        self.gpu_psf_mask = cp.asarray(psf_mask)

        self.timing_stats = {
            'memory_estimation': 0.0,
            'batch_preparation': 0.0,
            'gpu_transfer_in': 0.0,
            'gpu_computation': 0.0,
            'gpu_transfer_out': 0.0,
            'total_preprocessing': 0.0,
        }

        if verbose:
            print(f"GPU Memory: {self.gpu_memory_total / 1024**3:.2f} GB")

    def estimate_optimal_batch_size(self, image_shape: Tuple[int, int],
                                    max_shifts: int) -> int:
        """
        Estimate optimal batch size based on available GPU memory.

        Parameters
        ----------
        image_shape : tuple of int
            Shape of the input images (height, width).
        max_shifts : int
            Maximum number of shifts to process.

        Returns
        -------
        int
            Optimal batch size.
        """
        t_start = time.perf_counter()

        h, w = image_shape
        bytes_per_float64 = 8

        memory_per_shift = (
            h * w * bytes_per_float64 +
            (h + 40) * (w + 40) * bytes_per_float64 * 2 +
            (h + 40) * (w + 40) * bytes_per_float64 * 4 +
            h * w * bytes_per_float64
        ) * 1.5

        # Fresh mem_info read (via the shared driver) rather than the
        # value cached at construction -- see GPUBatchDriver's docstring.
        usable_memory = self._batch_driver.free_usable_bytes() - self.gpu_psf.nbytes * 2

        batch_size = max(1, int(usable_memory / memory_per_shift))
        batch_size = min(batch_size, max_shifts)

        self.timing_stats['memory_estimation'] += time.perf_counter() - t_start

        if self.verbose:
            print(f"  Memory per shift: {memory_per_shift / 1024**2:.2f} MB")
            print(f"  Available memory: {usable_memory / 1024**3:.2f} GB")
            print(f"  Optimal batch size: {batch_size}")

        return batch_size

    def _gpu_vip_fft_shift_batch(self, images: "cp.ndarray",
                                  shifts: "cp.ndarray") -> "cp.ndarray":
        """
        GPU implementation of VIP-FFT frame shifting for a batch of images.

        Parameters
        ----------
        images : cupy.ndarray
            Batch of images (batch_size, height, width).
        shifts : cupy.ndarray
            Shift values (batch_size, 2) where each row is [shift_y, shift_x].

        Returns
        -------
        cupy.ndarray
            Batch of shifted images.
        """
        batch_size, ny_ori, nx_ori = images.shape

        max_shift = cp.ceil(cp.amax(cp.abs(shifts)))
        npad = int(max_shift.item())

        new_y = int(ny_ori + 2 * npad)
        new_x = int(nx_ori + 2 * npad)

        padded_images = cp.zeros((batch_size, new_y, new_x), dtype=images.dtype)
        y_start, y_end = npad, npad + ny_ori
        x_start, x_end = npad, npad + nx_ori
        padded_images[:, y_start:y_end, x_start:x_end] = images

        npix = max(new_y, new_x)
        if new_y != new_x:
            square_images = cp.zeros((batch_size, npix, npix), dtype=images.dtype)
            if new_y > new_x:
                x_offset = (new_y - new_x) // 2
                square_images[:, :, x_offset:x_offset + new_x] = padded_images
                x_start += x_offset
            else:
                y_offset = (new_x - new_y) // 2
                square_images[:, y_offset:y_offset + new_y, :] = padded_images
                y_start += y_offset
            padded_images = square_images

        if npix % 2:
            npix += 1
            even_images = cp.zeros((batch_size, npix, npix), dtype=images.dtype)

            y_offsets = cp.where(shifts[:, 0] > 0, 0, 1)
            x_offsets = cp.where(shifts[:, 1] > 0, 0, 1)

            mask00 = (y_offsets == 0) & (x_offsets == 0)
            mask01 = (y_offsets == 0) & (x_offsets == 1)
            mask10 = (y_offsets == 1) & (x_offsets == 0)
            mask11 = (y_offsets == 1) & (x_offsets == 1)

            if cp.any(mask00):
                even_images[mask00, 0:npix-1, 0:npix-1] = padded_images[mask00]
            if cp.any(mask01):
                even_images[mask01, 0:npix-1, 1:npix] = padded_images[mask01]
            if cp.any(mask10):
                even_images[mask10, 1:npix, 0:npix-1] = padded_images[mask10]
            if cp.any(mask11):
                even_images[mask11, 1:npix, 1:npix] = padded_images[mask11]

            padded_images = even_images

            batch_y_start = cp.full(batch_size, y_start, dtype=cp.int64) + y_offsets
            batch_x_start = cp.full(batch_size, x_start, dtype=cp.int64) + x_offsets
        else:
            batch_y_start = cp.full(batch_size, y_start, dtype=cp.int64)
            batch_x_start = cp.full(batch_size, x_start, dtype=cp.int64)

        ramp = cp.outer(cp.ones(npix), cp.arange(npix) - npix / 2)
        shift_x_batch = shifts[:, 1][:, None, None]
        shift_y_batch = shifts[:, 0][:, None, None]

        tilt_batch = (-2 * cp.pi / npix) * (
            shift_x_batch * ramp[None, :, :] + shift_y_batch * ramp.T[None, :, :]
        )
        fact_batch = cp.fft.fftshift(
            cp.cos(tilt_batch) + 1j * cp.sin(tilt_batch), axes=(1, 2)
        )

        images_ft = cp.fft.fft2(padded_images)
        shifted_images = cp.fft.ifft2(images_ft * fact_batch).real

        b_idx, h_idx, w_idx = cp.indices((batch_size, ny_ori, nx_ori), dtype=cp.int64)
        h_idx += batch_y_start[:, None, None]
        w_idx += batch_x_start[:, None, None]

        return shifted_images[b_idx, h_idx, w_idx]

    def precompute_frame_shifts_batch(self, shift_coords: List[Tuple[float, float]],
                                      valid_mask: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Precompute frame shifts for all valid coordinates using GPU batching.

        Parameters
        ----------
        shift_coords : list of (float, float)
            List of (shift_y, shift_x) tuples.
        valid_mask : np.ndarray
            Boolean mask indicating which shifts are valid.

        Returns
        -------
        dict
            Mapping from original index to shifted PSF (flattened under mask).
        """
        total_start = time.perf_counter()

        valid_indices = np.where(valid_mask)[0]
        valid_shifts = [shift_coords[i] for i in valid_indices]

        if not valid_shifts:
            self.timing_stats['total_preprocessing'] = time.perf_counter() - total_start
            return {}

        batch_size = self.estimate_optimal_batch_size(self.psf.shape, len(valid_shifts))
        results = {}

        for batch_start in range(0, len(valid_shifts), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_shifts))
            current_batch_size = batch_end - batch_start

            t_prep = time.perf_counter()
            batch_shifts = valid_shifts[batch_start:batch_end]
            batch_indices = valid_indices[batch_start:batch_end]
            psf_batch = np.tile(self.psf[None, :, :], (current_batch_size, 1, 1))
            shifts_array = np.array(batch_shifts, dtype=psf_batch.dtype)
            self.timing_stats['batch_preparation'] += time.perf_counter() - t_prep

            t_transfer_in = time.perf_counter()
            with self.stream:
                gpu_psf_batch = cp.asarray(psf_batch)
                gpu_shifts = cp.asarray(shifts_array)
            self.stream.synchronize()
            self.timing_stats['gpu_transfer_in'] += time.perf_counter() - t_transfer_in

            t_compute = time.perf_counter()
            with self.stream:
                shifted_batch = self._gpu_vip_fft_shift_batch(gpu_psf_batch, gpu_shifts)
            self.stream.synchronize()
            self.timing_stats['gpu_computation'] += time.perf_counter() - t_compute

            t_transfer_out = time.perf_counter()
            cpu_shifted_batch = cp.asnumpy(shifted_batch)
            self.timing_stats['gpu_transfer_out'] += time.perf_counter() - t_transfer_out

            for i, orig_idx in enumerate(batch_indices):
                results[orig_idx] = cpu_shifted_batch[i][self.psf_mask]

            del gpu_psf_batch, gpu_shifts, shifted_batch
            cp.get_default_memory_pool().free_all_blocks()

            if self.verbose and len(valid_shifts) > batch_size:
                print(f"  Processed batch {batch_start // batch_size + 1}/"
                      f"{(len(valid_shifts) - 1) // batch_size + 1}")

        self.timing_stats['total_preprocessing'] = time.perf_counter() - total_start
        return results

    def stream_precomputed_shifts(
        self,
        shift_coords: List[Tuple[float, float]],
        valid_mask: np.ndarray,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator that yields (index, shifted_psf) pairs batch-by-batch to save memory.

        Parameters
        ----------
        shift_coords : list of (float, float)
            (shift_y, shift_x) pairs for every candidate shift.
        valid_mask : np.ndarray
            Boolean mask indicating which entries in shift_coords are valid.

        Yields
        ------
        tuple of (int, np.ndarray)
            Original shift index and the corresponding flattened shifted PSF.
        """
        total_start = time.perf_counter()
        valid_indices = np.where(valid_mask)[0]
        valid_shifts = [shift_coords[i] for i in valid_indices]

        if not valid_shifts:
            self.timing_stats['total_preprocessing'] = time.perf_counter() - total_start
            return

        batch_size = self.estimate_optimal_batch_size(self.psf.shape, len(valid_shifts))
        stream1, stream2 = cp.cuda.Stream(), cp.cuda.Stream()
        streams = [stream1, stream2]
        gpu_shifted_buffers = [None, None]
        batch_indices_pipeline = [None, None]
        num_batches = (len(valid_shifts) + batch_size - 1) // batch_size

        for i in range(num_batches):
            buf = i % 2
            current_stream = streams[buf]

            if gpu_shifted_buffers[buf] is not None:
                current_stream.synchronize()
                cpu_shifted_batch = cp.asnumpy(gpu_shifted_buffers[buf])
                for j, orig_idx in enumerate(batch_indices_pipeline[buf]):
                    yield orig_idx, cpu_shifted_batch[j][self.psf_mask]
                gpu_shifted_buffers[buf] = None
                cp.get_default_memory_pool().free_all_blocks()

            batch_start = i * batch_size
            batch_end = min(batch_start + batch_size, len(valid_shifts))
            current_batch_shifts = valid_shifts[batch_start:batch_end]
            batch_indices_pipeline[buf] = valid_indices[batch_start:batch_end]

            psf_batch = np.tile(self.psf[None, :, :], (len(current_batch_shifts), 1, 1))
            shifts_array = np.array(current_batch_shifts, dtype=psf_batch.dtype)

            with current_stream:
                gpu_psf_batch = cp.asarray(psf_batch)
                gpu_shifts = cp.asarray(shifts_array)
                gpu_shifted_buffers[buf] = self._gpu_vip_fft_shift_batch(
                    gpu_psf_batch, gpu_shifts
                )

        for buf in range(len(streams)):
            if gpu_shifted_buffers[buf] is not None:
                streams[buf].synchronize()
                cpu_shifted_batch = cp.asnumpy(gpu_shifted_buffers[buf])
                for j, orig_idx in enumerate(batch_indices_pipeline[buf]):
                    yield orig_idx, cpu_shifted_batch[j][self.psf_mask]

        self.timing_stats['total_preprocessing'] = time.perf_counter() - total_start

    def print_timing_report(self) -> None:
        """Print a detailed timing breakdown for GPU operations."""
        print("\n--- GPU Frame Shift Profiling Report ---")
        total_time = self.timing_stats['total_preprocessing']
        if total_time == 0:
            print("No GPU timing data collected.")
            return

        print(f"Total preprocessing time: {total_time:.4f} seconds\n")
        print(f"{'Operation':<25} | {'Time (s)':<15} | {'% of Total':<10}")
        print("-" * 55)

        for key, time_spent in self.timing_stats.items():
            if key != 'total_preprocessing':
                pct = (time_spent / total_time) * 100 if total_time > 0 else 0
                print(f"{key:<25} | {time_spent:<15.4f} | {pct:<10.2f}%")

        transfer_time = (self.timing_stats['gpu_transfer_in'] +
                         self.timing_stats['gpu_transfer_out'])
        compute_time = self.timing_stats['gpu_computation']
        if transfer_time > 0:
            efficiency = compute_time / (compute_time + transfer_time) * 100
            print(f"\nGPU Compute Efficiency: {efficiency:.2f}%")


class GPUStatisticsBatcher:
    """
    Batches PACO's per-pixel background-statistics computation (Eqs 7 & 12
    of [FLA18]: temporal mean, shrinkage-regularised inverse covariance) onto
    the GPU.

    Why this stage specifically: GPUFrameShifter accelerates the FFT-based
    PSF shifting, whose cost is O(N_pixels) — dominant for a small cube. The
    *statistics* cost is O(N_pixels * (T*K^2 + K^3)) (covariance + K x K
    inversion per pixel), which grows faster in both T (frame count) and K
    (patch area) — for a large cube this, not the PSF shifts, dominates.
    Unlike per-pixel CPU workers, a single (N, T, K) patch batch lets
    covariance and inversion run as a handful of large batched BLAS/cuBLAS
    calls (see math_utils.compute_statistics_batch) instead of N small
    per-pixel Python calls dominated by interpreter/launch overhead — that
    batching, not raw device throughput on one (T, K) problem, is what makes
    a GPU worth using here.

    VRAM budgeting mirrors GPUFrameShifter: a safety-factor fraction of
    total device memory, chunked so a single transfer+compute step never
    tries to hold more than that on the device at once. This is a *separate*
    budget from whatever host-RAM batch sizing the caller already does
    (e.g. FastPACOv2._calculate_batch_size, FullPACOv2._estimate_cache_capacity)
    — a CPU-side batch of pixels can be, and often should be, larger than
    what fits on the GPU at once; this class re-chunks internally rather
    than assuming the two budgets coincide.

    GPU failures are handled per-chunk, not per-run (see GPUBatchDriver):
    a chunk that doesn't fit in currently-free VRAM is retried at half
    size, and a chunk that still fails (OOM after retries, or any other
    CUDA error) falls back to the CPU path (math_utils.compute_statistics_batch
    with use_gpu=False) for that chunk only, with a warning -- a transient
    VRAM pressure spike on a shared server can no longer lose an entire
    multi-hour run.

    Exercised end-to-end (including the CPU-fallback path) against the
    fake-GPU shim in benchmarks/fake_cupy.py, which lets the exact same
    code run on a machine with no CUDA device by delegating every array
    op to NumPy -- see benchmarks/bench_gpu_fake_smoke.py. That catches
    API-shape bugs (wrong axis, wrong dtype, a CuPy-only call that
    doesn't exist) but not real GPU throughput or real OOM behavior; see
    benchmarks/bench_gpu_stats.py for the correctness+speed check to run
    on a server with a real CUDA device.
    """

    def __init__(self, gpu_memory_safety_factor: float = 0.8,
                 max_retries: int = 3, verbose: bool = False):
        self.verbose = verbose
        self.gpu_memory_safety_factor = gpu_memory_safety_factor
        self.device = cp.cuda.Device()
        self.gpu_memory_total = self.device.mem_info[1]
        self._batch_driver = GPUBatchDriver(
            gpu_memory_safety_factor=gpu_memory_safety_factor,
            max_retries=max_retries, verbose=verbose)
        if verbose:
            print(f"GPU Memory: {self.gpu_memory_total / 1024**3:.2f} GB "
                  f"(statistics batcher)")

    @staticmethod
    def _bytes_per_item(T: int, K: int) -> float:
        """
        Accounts for: the input patch batch (T*K per item), the working
        covariance / shrinkage-target / combined-covariance buffers (3*K*K,
        all live briefly per batch inside compute_statistics_batch), and the
        output (mean K + inverse-covariance K*K) — all float64. The 1.5x
        fudge factor mirrors GPUFrameShifter's, for the same reason:
        cuBLAS/cuSOLVER internal working buffers for batched einsum/inv
        aren't fully accounted for by the raw array sizes above.
        """
        return (T * K + 3 * K * K + K + K * K) * 8 * 1.5

    def estimate_optimal_batch_size(self, T: int, K: int, n_items: int) -> int:
        """VRAM-budget-based batch size for a (batch, T, K) patch stack,
        against currently-free VRAM (see GPUBatchDriver.free_usable_bytes)."""
        bytes_per_item = self._bytes_per_item(T, K)
        batch_size = self._batch_driver.estimate_batch_size(bytes_per_item, n_items)
        if self.verbose:
            usable = self._batch_driver.free_usable_bytes()
            print(f"  [GPUStatisticsBatcher] bytes/item={bytes_per_item / 1024:.1f} KiB, "
                  f"usable={usable / 1024**3:.2f} GiB -> batch_size={batch_size}")
        return batch_size

    def compute_stats_streamed(self, patches: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute (m, Cinv) for a full (N, T, K) host-side patch stack,
        chunked to fit the VRAM budget.

        Used by PACO.compute_statistics_gpu (the base/reference Algorithm-1
        statistics stage), which needs the full m/Cinv maps for later
        per-trajectory lookups — unlike FastPACO's Stage 1 (see
        compute_stage1_streamed), which only needs a scalar+vector
        contraction of these and so avoids transferring the full Cinv back.

        Parameters
        ----------
        patches : np.ndarray, shape (N, T, K)

        Returns
        -------
        m : np.ndarray, shape (N, K)
        Cinv : np.ndarray, shape (N, K, K)
        """
        from .math_utils import compute_statistics_batch

        N, T, K = patches.shape
        m_out = np.full((N, K), np.nan, dtype=np.float64)
        Cinv_out = np.full((N, K, K), np.nan, dtype=np.float64)
        if N == 0:
            return m_out, Cinv_out

        bytes_per_item = self._bytes_per_item(T, K)
        pbar = tqdm.tqdm(total=N, desc="  - GPU statistics", leave=False,
                         disable=not self.verbose)

        def gpu_fn(start: int, end: int) -> None:
            gpu_patches = cp.asarray(patches[start:end])
            m_gpu, Cinv_gpu = compute_statistics_batch(gpu_patches, use_gpu=True)
            m_out[start:end] = cp.asnumpy(m_gpu)
            Cinv_out[start:end] = cp.asnumpy(Cinv_gpu)
            del gpu_patches, m_gpu, Cinv_gpu
            cp.get_default_memory_pool().free_all_blocks()
            pbar.update(end - start)

        def cpu_fn(start: int, end: int) -> None:
            # Per-chunk fallback (see GPUBatchDriver): the same batched
            # math, just on NumPy -- not a per-pixel loop, so a fallback
            # chunk is not catastrophically slower than its GPU twin.
            m_cpu, Cinv_cpu = compute_statistics_batch(patches[start:end], use_gpu=False)
            m_out[start:end] = m_cpu
            Cinv_out[start:end] = Cinv_cpu
            pbar.update(end - start)

        try:
            self._batch_driver.run(N, bytes_per_item, gpu_fn, cpu_fn)
        finally:
            pbar.close()
        return m_out, Cinv_out

    def compute_stage1_streamed(self, patches: np.ndarray,
                                psf_vectorized: np.ndarray
                                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        FastPACO Stage-1-specific batched pipeline: computes (m, Cinv) *and*
        immediately contracts them with the (fixed) vectorised PSF template
        to produce the scalar c and temporal vector d that Stage 1 actually
        keeps (see fast_paco._stage1_worker for the per-pixel equivalent:
        w = Cinv @ psf, c = psf . w, d = (patch - m) @ w).

        Doing this contraction on-device before transferring back is the
        point: it shrinks the per-pixel payload from a K x K matrix (Cinv)
        to a scalar (c) plus a T-vector (d) — for a large K this is the
        difference between transferring O(N*K^2) and O(N*T) floats back over
        PCIe, and FastPACO never needs the raw Cinv/m after Stage 1 anyway
        (only c_map and d_map survive into Stage 2).

        Parameters
        ----------
        patches : np.ndarray, shape (N, T, K)
        psf_vectorized : np.ndarray, shape (K,)
            PSF template vector under the same circular mask as the patches.

        Returns
        -------
        c : np.ndarray, shape (N,)
        d : np.ndarray, shape (N, T)
            NaN for pixels whose patch was invalid — falls out automatically
            from compute_statistics_batch's NaN (m, Cinv) for those rows
            propagating through the einsum contractions below, no separate
            masking needed.
        """
        from .math_utils import compute_statistics_batch

        N, T, K = patches.shape
        c_out = np.full(N, np.nan, dtype=np.float64)
        d_out = np.full((N, T), np.nan, dtype=np.float64)
        if N == 0:
            return c_out, d_out

        bytes_per_item = self._bytes_per_item(T, K)
        gpu_psf = cp.asarray(psf_vectorized)
        pbar = tqdm.tqdm(total=N, desc="  - GPU stage-1 statistics", leave=False,
                         disable=not self.verbose)

        def gpu_fn(start: int, end: int) -> None:
            gpu_patches = cp.asarray(patches[start:end])
            m_gpu, Cinv_gpu = compute_statistics_batch(gpu_patches, use_gpu=True)

            w_gpu = cp.einsum('nkl,l->nk', Cinv_gpu, gpu_psf)
            c_gpu = cp.einsum('nk,k->n', w_gpu, gpu_psf)
            d_gpu = cp.einsum('ntk,nk->nt', gpu_patches - m_gpu[:, None, :], w_gpu)

            c_out[start:end] = cp.asnumpy(c_gpu)
            d_out[start:end] = cp.asnumpy(d_gpu)
            del gpu_patches, m_gpu, Cinv_gpu, w_gpu, c_gpu, d_gpu
            cp.get_default_memory_pool().free_all_blocks()
            pbar.update(end - start)

        def cpu_fn(start: int, end: int) -> None:
            cpu_patches = patches[start:end]
            m_cpu, Cinv_cpu = compute_statistics_batch(cpu_patches, use_gpu=False)
            w_cpu = np.einsum('nkl,l->nk', Cinv_cpu, psf_vectorized)
            c_cpu = np.einsum('nk,k->n', w_cpu, psf_vectorized)
            d_cpu = np.einsum('ntk,nk->nt', cpu_patches - m_cpu[:, None, :], w_cpu)
            c_out[start:end] = c_cpu
            d_out[start:end] = d_cpu
            pbar.update(end - start)

        try:
            self._batch_driver.run(N, bytes_per_item, gpu_fn, cpu_fn)
        finally:
            pbar.close()
        return c_out, d_out
