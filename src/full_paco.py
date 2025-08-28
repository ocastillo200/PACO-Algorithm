import multiprocessing

from tqdm import tqdm
from gpu_frame_shift import GPUFrameShifter
from main_paco import PACO
from vip_hci.preproc.rescaling import frame_shift
import numpy as np
import psutil
from vip_hci.config.utils_conf import pool_map, iterable
from main_paco import PACO
from math_functions import compute_statistics_at_pixel
from utils import create_boolean_circular_mask, get_rotated_pixel_coords

try:
    import cupy 
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class FullPACO(PACO):
    """
    GPU-accelerated FullPACO implementation with proper CUDA/multiprocessing isolation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_shifter = None
        self.precomputed_shifts = {}
        
    def PACOCalc(self,
                 phi0s: np.ndarray,
                 use_subpixel_psf_astrometry: bool = True,
                 use_gpu: bool = True,
                 cpu: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Enhanced PACOCalc with proper GPU/multiprocessing isolation.
        """
        npx = len(phi0s)
        dimx = self.width / 2

        a = np.zeros(npx)
        b = np.zeros(npx)

        self.psf_template_mask = create_boolean_circular_mask(
            self.psf.shape, radius=self.fwhm)

        self.centered_grid_x_values, self.centered_grid_y_values = np.meshgrid(
            np.arange(-dimx, dimx), np.arange(-dimx, dimx)
        )

        self.current_use_subpixel_psf_astrometry = use_subpixel_psf_astrometry
        self.use_gpu = use_gpu and GPU_AVAILABLE

        if cpu > 1:
            manager = multiprocessing.Manager()
            self.cached_pixel_data = manager.dict()

        else:
            self.cached_pixel_data = {}

        pixel_batch_size = self._estimate_cpu_pixel_batch_size()

        for i in range(0, npx, pixel_batch_size):
            pixel_batch_slice = slice(i, min(i + pixel_batch_size, npx))
            pixel_batch = phi0s[pixel_batch_slice]
            
            # 1. GPU preprocessing 
            if self.use_gpu and use_subpixel_psf_astrometry:
                self._precompute_shifts_gpu(pixel_batch, original_start_index=i)

            precomputed_subset = {
                pixel_idx: self.precomputed_shifts[pixel_idx]
                for pixel_idx in range(i, i + len(pixel_batch))
                if pixel_idx in self.precomputed_shifts
            }

            # 2. Prepare args including only pixel-level data
            process_args = [
                (i + j, p0, precomputed_subset.get(i + j, {}))
                for j, p0 in enumerate(pixel_batch)
            ]

            if cpu > 1: 
                results = pool_map(cpu, self._process_pixel_for_calc, iterable(process_args), progressbar_single=self.verbose)
            else:
                results = [self._process_pixel_for_calc(args) for args in tqdm.tqdm(process_args, desc="  - CPU calculations", leave=False, disable=not self.verbose)]

            # 3. Store results from the batch
            for res_i, res_a, res_b in results:
                if res_i is not None:
                    a[res_i] = res_a
                    b[res_i] = res_b

        if self.verbose:
            print(f"\Full PACO is done!")

        return a, b
    
    def _precompute_shifts_gpu(self, pixel_batch: np.ndarray, original_start_index: int):
        """
        GPU preprocessing with complete CUDA isolation from multiprocessing.
        """
        
        self.precomputed_shifts = {}
        
        # Create GPU shifter in isolated scope
        gpu_shifter = None
        try:
            gpu_shifter = GPUFrameShifter(
                self.psf, self.psf_template_mask, verbose=self.verbose
            )
            
            # Prepare shift requests
            all_shifts = []
            shift_to_pixel = {}
            valid_mask = []
            shift_idx_counter = 0

            for j, p0 in enumerate(pixel_batch):
                original_pixel_idx = original_start_index + j
                p0_indices = (p0[1], p0[0])
                is_pixel_valid = False
                
                try:
                    angles_px = get_rotated_pixel_coords(
                        self.centered_grid_x_values, self.centered_grid_y_values, 
                        p0_indices, self.angles
                    )
                    y_coords, x_coords = angles_px[:, 0], angles_px[:, 1]
                    if not (np.any(y_coords < 0) or np.any(y_coords >= self.height) or
                            np.any(x_coords < 0) or np.any(x_coords >= self.width)):
                        is_pixel_valid = True
                except Exception:
                    angles_px = np.zeros((len(self.angles), 2))
                
                for l, ang in enumerate(angles_px):
                    if is_pixel_valid:
                        yi_rot, xi_rot = int(ang[0]), int(ang[1])
                        subpix_dy, subpix_dx = ang[0] - yi_rot, ang[1] - xi_rot
                        all_shifts.append((subpix_dx, subpix_dy))
                        shift_to_pixel[shift_idx_counter] = (original_pixel_idx, l)
                        valid_mask.append(True)
                    else:
                        all_shifts.append((0.0, 0.0))
                        valid_mask.append(False)
                    shift_idx_counter += 1
            
            # Process GPU shifts and immediately convert to CPU
            valid_mask_array = np.array(valid_mask)
            shift_generator = gpu_shifter.stream_precomputed_shifts(all_shifts, valid_mask_array)

            num_valid_shifts = np.sum(valid_mask_array)
            pbar_desc = "  - GPU preprocessing"
            pbar = tqdm.tqdm(shift_generator, total=num_valid_shifts, desc=pbar_desc, 
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
                print(f"GPU preprocessing: {num_processed} shifts computed and transferred to CPU.")
                
        finally:
            # Clean up GPU resources immediately and completely
            if gpu_shifter is not None:
                del gpu_shifter
            # Force CUDA cleanup
            try:
                import cupy
                cupy.get_default_memory_pool().free_all_blocks()
                cupy.fuse(mode='never')  # Disable kernel fusion to avoid lingering CUDA state
            except:
                pass
    
    def _estimate_cpu_pixel_batch_size(self) -> int:
        """
        Dynamically estimates the number of pixels that can be processed at once.
        """
        try:
            available_ram = psutil.virtual_memory().available
            usable_ram = available_ram * 0.65
            
            float_size = np.dtype(np.float64).itemsize
            num_angles = len(self.angles)
            pixels_in_psf_mask = np.sum(self.psf_template_mask)
            memory_per_pixel = num_angles * pixels_in_psf_mask * float_size * 1.2
            
            if memory_per_pixel == 0:
                return 500 
            
            batch_size = int(usable_ram / memory_per_pixel)
            
            if self.verbose:
                print(f"Available RAM: {available_ram / 1024**3:.2f} GB")
                print(f"Memory per pixel: {memory_per_pixel / 1024**2:.2f} MB")
                print(f"Batch size: {max(1, batch_size)}")
            
            return max(1, batch_size)
            
        except ImportError:
            if self.verbose:
                print("psutil not found. Using default batch size of 500.")
            return 500
        except Exception as e:
            if self.verbose:
                print(f"Error estimating batch size: {e}. Using default of 500.")
            return 500
    
    def _process_pixel_for_calc(self, args_tuple: tuple):
        i, p0, precomputed_pixel_data = args_tuple

        try:
            p0_indices_for_grid = (p0[1], p0[0])
            angles_px = get_rotated_pixel_coords(
                self.centered_grid_x_values,
                self.centered_grid_y_values,
                p0_indices_for_grid,
                self.angles
            )

            y_coords_rot = angles_px[:, 0]
            x_coords_rot = angles_px[:, 1]
            if (np.any(y_coords_rot < 0) or np.any(y_coords_rot >= self.height) or
                np.any(x_coords_rot < 0) or np.any(x_coords_rot >= self.width)):
                return i, np.nan, np.nan

            patches, mlst, clst, hlst = [], [], [], []

            for l, ang in enumerate(angles_px):
                yi_rot, xi_rot = int(ang[0]), int(ang[1])

                if not (0 <= yi_rot < self.height and 0 <= xi_rot < self.width):
                    return i, np.nan, np.nan

                cache_key = (yi_rot, xi_rot)
                if cache_key not in self.cached_pixel_data:
                    apatch = self.get_patch((xi_rot, yi_rot))
                    m, Cinv = compute_statistics_at_pixel(apatch)
                    self.cached_pixel_data[cache_key] = (apatch, m, Cinv)

                apatch_cached, m_cached, Cinv_cached = self.cached_pixel_data[cache_key]
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
                else:
                    if self.current_use_subpixel_psf_astrometry:
                        subpix_dx = ang[1] - xi_rot
                        subpix_dy = ang[0] - yi_rot
                        offax = frame_shift(
                            self.psf,
                            subpix_dx, subpix_dy,
                            imlib='vip-fft', interpolation='lanczos4',
                            border_mode='reflect'
                        )[self.psf_template_mask]
                    else:
                        offax = self.psf[self.psf_template_mask]

                hlst.append(offax)

            if not hlst:
                return i, np.nan, np.nan

            h_arr = np.array(hlst)
            Cinv_arr = np.array(clst)
            r_arr = np.array(patches)
            m_arr = np.array(mlst)

            if (h_arr.size == 0 or np.any(np.isnan(h_arr)) or 
                np.any(np.isnan(Cinv_arr)) or np.any(np.isnan(r_arr)) or
                np.any(np.isnan(m_arr))):
                return i, np.nan, np.nan

            w_arr = self.wfl(h_arr, Cinv_arr, use_gpu=False)
            a_val = self.al(h_arr, w_arr, use_gpu=False)
            b_val = self.bl(w_arr, r_arr, m_arr, use_gpu=False)

            return i, a_val, b_val

        except Exception as e:
            if self.verbose:
                print(f"Error processing pixel {i}: {e}")
            return i, np.nan, np.nan