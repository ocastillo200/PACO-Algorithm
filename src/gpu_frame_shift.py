import time
from typing import Dict, Generator, List, Tuple
import numpy as np

try:
    import cupy as cp
except ImportError:
    raise ImportError("CuPy is required for GPUFrameShifter but is not installed or available.")

class GPUFrameShifter:
    """
    GPU-accelerated frame shifting with batch processing and memory optimization.
    """
    
    def __init__(self, psf: np.ndarray, psf_mask: np.ndarray, verbose: bool = False):
        """
        Initialize GPU frame shifter.
        
        Args:
            psf: Point spread function array
            psf_mask: Boolean mask for PSF template
            verbose: Enable verbose output
        """
        self.psf = psf
        self.psf_mask = psf_mask
        self.verbose = verbose
        self.device = cp.cuda.Device()
        self.stream = cp.cuda.Stream()
        
        # GPU memory info
        self.gpu_memory_total = self.device.mem_info[1]
        self.gpu_memory_safety_factor = 0.8  # Use 80% of available memory
        
        # Transfer PSF to GPU once
        self.gpu_psf = cp.asarray(psf)
        self.gpu_psf_mask = cp.asarray(psf_mask)
        
        # Timing statistics
        self.timing_stats = {
            'memory_estimation': 0.0,
            'batch_preparation': 0.0,
            'gpu_transfer_in': 0.0,
            'gpu_computation': 0.0,
            'gpu_transfer_out': 0.0,
            'total_preprocessing': 0.0
        }
        
        if verbose:
            print(f"GPU Memory: {self.gpu_memory_total / 1024**3:.2f} GB")
    
    def estimate_optimal_batch_size(self, image_shape: Tuple[int, int], 
                                   max_shifts: int) -> int:
        """
        Estimate optimal batch size based on GPU memory constraints.
        
        Args:
            image_shape: Shape of input images (height, width)
            max_shifts: Maximum number of shifts to process
            
        Returns:
            Optimal batch size
        """
        t_start = time.perf_counter()
        
        # Memory required per shift operation
        h, w = image_shape
        bytes_per_float64 = 8
        
        # Memory for: original image, padded image, FFT arrays, result
        memory_per_shift = (
            h * w * bytes_per_float64 +  # original image
            (h + 40) * (w + 40) * bytes_per_float64 * 2 +  # padded image (real + complex)
            (h + 40) * (w + 40) * bytes_per_float64 * 4 +  # FFT intermediate arrays
            h * w * bytes_per_float64  # result
        )
        
        # Add overhead for other GPU operations
        memory_per_shift *= 1.5
        
        # Available memory (considering safety factor)
        available_memory = self.gpu_memory_total * self.gpu_memory_safety_factor
        
        # Reserve memory for PSF and other constants
        reserved_memory = self.gpu_psf.nbytes * 2
        usable_memory = available_memory - reserved_memory
        
        # Calculate batch size
        batch_size = max(1, int(usable_memory / memory_per_shift))
        batch_size = min(batch_size, max_shifts)
        
        self.timing_stats['memory_estimation'] += time.perf_counter() - t_start
        
        if self.verbose:
            print(f"\n Memory per shift: {memory_per_shift / 1024**2:.2f} MB")
            print(f"Available memory: {usable_memory / 1024**3:.2f} GB")
            print(f"Optimal batch size: {batch_size} \n")
        
        return batch_size
    
    def _gpu_vip_fft_shift_batch(self, images: cp.ndarray, shifts: cp.ndarray) -> cp.ndarray:
        """
        GPU implementation of VIP-FFT frame shifting for a batch of images.
        
        Args:
            images: Batch of images (batch_size, height, width)
            shifts: Shift values (batch_size, 2) where each row is [shift_y, shift_x]
            
        Returns:
            Batch of shifted images
        """
        batch_size, ny_ori, nx_ori = images.shape
        
        # Find maximum shift to determine padding
        max_shift = cp.ceil(cp.amax(cp.abs(shifts)))
        npad = int(max_shift.item())
        
        # Prepare padded images
        new_y = int(ny_ori + 2 * npad)
        new_x = int(nx_ori + 2 * npad)
        
        # Batch padding
        padded_images = cp.zeros((batch_size, new_y, new_x), dtype=images.dtype)
        y_start = npad
        y_end = npad + ny_ori
        x_start = npad
        x_end = npad + nx_ori
        padded_images[:, y_start:y_end, x_start:x_end] = images
        
        # Make square if needed
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
        
        # Make even if needed 
        if npix % 2:
            npix += 1
            even_images = cp.zeros((batch_size, npix, npix), dtype=images.dtype)
            
            # Determine 0 or 1 pixel offset based on shift direction
            y_offsets = cp.where(shifts[:, 0] > 0, 0, 1)
            x_offsets = cp.where(shifts[:, 1] > 0, 0, 1)

            # Create boolean masks for each of the four offset cases
            mask00 = (y_offsets == 0) & (x_offsets == 0)
            mask01 = (y_offsets == 0) & (x_offsets == 1)
            mask10 = (y_offsets == 1) & (x_offsets == 0)
            mask11 = (y_offsets == 1) & (x_offsets == 1)

            # Apply vectorized copy for each case
            if cp.any(mask00):
                even_images[mask00, 0:npix-1, 0:npix-1] = padded_images[mask00]
            if cp.any(mask01):
                even_images[mask01, 0:npix-1, 1:npix] = padded_images[mask01]
            if cp.any(mask10):
                even_images[mask10, 1:npix, 0:npix-1] = padded_images[mask10]
            if cp.any(mask11):
                even_images[mask11, 1:npix, 1:npix] = padded_images[mask11]
                
            padded_images = even_images
            
            # Vectorially update start coordinates for the final crop
            batch_y_start = cp.full(batch_size, y_start, dtype=cp.int64) + y_offsets
            batch_x_start = cp.full(batch_size, x_start, dtype=cp.int64) + x_offsets
        else:
            # If no padding occurred, create the coordinate arrays without offsets
            batch_y_start = cp.full(batch_size, y_start, dtype=cp.int64)
            batch_x_start = cp.full(batch_size, x_start, dtype=cp.int64)

        # Create tilt arrays for batch
        ramp = cp.outer(cp.ones(npix), cp.arange(npix) - npix / 2)
        
        # Vectorized tilt computation
        shift_x_batch = shifts[:, 1][:, None, None]  # (batch_size, 1, 1)
        shift_y_batch = shifts[:, 0][:, None, None]  # (batch_size, 1, 1)
        
        tilt_batch = (-2 * cp.pi / npix) * (shift_x_batch * ramp[None, :, :] + 
                                        shift_y_batch * ramp.T[None, :, :])
        
        # Compute phase factors
        fact_batch = cp.fft.fftshift(cp.cos(tilt_batch) + 1j * cp.sin(tilt_batch), axes=(1, 2))
        
        # Batch FFT
        images_ft = cp.fft.fft2(padded_images)
        
        # Apply phase shifts
        shifted_ft = images_ft * fact_batch
        
        # Batch IFFT
        shifted_images = cp.fft.ifft2(shifted_ft).real
        
        # Create index grids for the destination shape (batch, height, width)
        b_idx, h_idx, w_idx = cp.indices((batch_size, ny_ori, nx_ori), dtype=cp.int64)

        # Add the per-image start offsets to the grids
        h_idx += batch_y_start[:, None, None]
        w_idx += batch_x_start[:, None, None]

        # Use advanced indexing to gather the pixels from the shifted images
        result = shifted_images[b_idx, h_idx, w_idx]
        
        return result
    
    def precompute_frame_shifts_batch(self, shift_coords: List[Tuple[float, float]], 
                                     valid_mask: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Precompute frame shifts for all valid coordinates using GPU batching.
        
        Args:
            shift_coords: List of (shift_y, shift_x) tuples
            valid_mask: Boolean mask indicating which shifts are valid
            
        Returns:
            Dictionary mapping original index to shifted PSF
        """
        total_start = time.perf_counter()
        
        # Filter valid shifts
        valid_indices = np.where(valid_mask)[0]
        valid_shifts = [shift_coords[i] for i in valid_indices]
        
        if not valid_shifts:
            self.timing_stats['total_preprocessing'] = time.perf_counter() - total_start
            return {}
        
        # Estimate batch size
        batch_size = self.estimate_optimal_batch_size(self.psf.shape, len(valid_shifts))
        
        results = {}
        
        # Process in batches
        for batch_start in range(0, len(valid_shifts), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_shifts))
            current_batch_size = batch_end - batch_start
            
            # Prepare batch
            t_prep = time.perf_counter()
            batch_shifts = valid_shifts[batch_start:batch_end]
            batch_indices = valid_indices[batch_start:batch_end]
            
            # Create batch of PSF images
            psf_batch = np.tile(self.psf[None, :, :], (current_batch_size, 1, 1))
            shifts_array = np.array(batch_shifts, dtype=psf_batch.dtype)
            
            self.timing_stats['batch_preparation'] += time.perf_counter() - t_prep
            
            # Transfer to GPU
            t_transfer_in = time.perf_counter()
            with self.stream:
                gpu_psf_batch = cp.asarray(psf_batch)
                gpu_shifts = cp.asarray(shifts_array)
            self.stream.synchronize()
            self.timing_stats['gpu_transfer_in'] += time.perf_counter() - t_transfer_in
            
            # GPU computation
            t_compute = time.perf_counter()
            with self.stream:
                shifted_batch = self._gpu_vip_fft_shift_batch(gpu_psf_batch, gpu_shifts)
            self.stream.synchronize()
            self.timing_stats['gpu_computation'] += time.perf_counter() - t_compute
            
            # Transfer back to CPU
            t_transfer_out = time.perf_counter()
            cpu_shifted_batch = cp.asnumpy(shifted_batch)
            self.timing_stats['gpu_transfer_out'] += time.perf_counter() - t_transfer_out
            
            # Store results with mask applied
            for i, orig_idx in enumerate(batch_indices):
                results[orig_idx] = cpu_shifted_batch[i][self.psf_mask]
            
            # Memory cleanup
            del gpu_psf_batch, gpu_shifts, shifted_batch
            cp.get_default_memory_pool().free_all_blocks()
            
            if self.verbose and len(valid_shifts) > batch_size:
                print(f"Processed batch {batch_start//batch_size + 1}/{(len(valid_shifts)-1)//batch_size + 1}")
        
        self.timing_stats['total_preprocessing'] = time.perf_counter() - total_start
        return results

    def stream_precomputed_shifts(self, shift_coords: List[Tuple[float, float]],
                                  valid_mask: np.ndarray) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Generator that computes and yields frame shifts batch by batch to save memory."""
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
            buffer_idx = i % 2
            current_stream = streams[buffer_idx]
            if gpu_shifted_buffers[buffer_idx] is not None:
                current_stream.synchronize()
                cpu_shifted_batch = cp.asnumpy(gpu_shifted_buffers[buffer_idx])
                for j, orig_idx in enumerate(batch_indices_pipeline[buffer_idx]):
                    yield orig_idx, cpu_shifted_batch[j][self.psf_mask]
                gpu_shifted_buffers[buffer_idx] = None
                cp.get_default_memory_pool().free_all_blocks()
            batch_start = i * batch_size
            batch_end = min(batch_start + batch_size, len(valid_shifts))
            current_batch_shifts = valid_shifts[batch_start:batch_end]
            batch_indices_pipeline[buffer_idx] = valid_indices[batch_start:batch_end]
            psf_batch = np.tile(self.psf[None, :, :], (len(current_batch_shifts), 1, 1))
            shifts_array = np.array(current_batch_shifts, dtype=psf_batch.dtype)
            with current_stream:
                gpu_psf_batch = cp.asarray(psf_batch)
                gpu_shifts = cp.asarray(shifts_array)
                gpu_shifted_buffers[buffer_idx] = self._gpu_vip_fft_shift_batch(gpu_psf_batch, gpu_shifts)
        for buffer_idx in range(len(streams)):
            if gpu_shifted_buffers[buffer_idx] is not None:
                streams[buffer_idx].synchronize()
                cpu_shifted_batch = cp.asnumpy(gpu_shifted_buffers[buffer_idx])
                for j, orig_idx in enumerate(batch_indices_pipeline[buffer_idx]):
                    yield orig_idx, cpu_shifted_batch[j][self.psf_mask]
        self.timing_stats['total_preprocessing'] = time.perf_counter() - total_start
    
    def print_timing_report(self):
        """Print detailed timing report for GPU operations."""
        print("\n--- GPU Frame Shift Profiling Report ---")
        
        total_time = self.timing_stats['total_preprocessing']
        if total_time == 0:
            print("No GPU timing data collected.")
            return
            
        print(f"Total preprocessing time: {total_time:.4f} seconds")
        print()
        
        print("Breakdown of GPU operations:")
        print(f"{'Operation':<25} | {'Time (s)':<15} | {'% of Total':<10}")
        print("-" * 55)
        
        for key, time_spent in self.timing_stats.items():
            if key != 'total_preprocessing':
                percentage = (time_spent / total_time) * 100 if total_time > 0 else 0
                print(f"{key:<25} | {time_spent:<15.4f} | {percentage:<10.2f}%")
        
        # Memory transfer efficiency
        transfer_time = (self.timing_stats['gpu_transfer_in'] + 
                        self.timing_stats['gpu_transfer_out'])
        compute_time = self.timing_stats['gpu_computation']
        
        if transfer_time > 0:
            efficiency = compute_time / (compute_time + transfer_time) * 100
            print(f"\nGPU Compute Efficiency: {efficiency:.2f}%")
            print(f"(Compute time / (Compute + Transfer time))")