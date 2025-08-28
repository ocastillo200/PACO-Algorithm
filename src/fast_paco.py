from typing import Generator, Optional, Tuple
import numpy as np
import psutil
from scipy.ndimage import map_coordinates
from vip_hci.config.utils_conf import pool_map, iterable
from main_paco import PACO
from math_functions import compute_statistics_at_pixel
from utils import create_boolean_circular_mask, get_rotated_pixel_coords

class FastPACO(PACO):
    def PACOCalc(self,
                 phi0s: np.ndarray,
                 interpolation_order: Optional[int] = 1,
                 cpu: Optional[int] = 1,
                 memory_threshold_gb: Optional[float] = None,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        
        if memory_threshold_gb is None:
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            memory_threshold_gb = available_gb * 0.9
            if self.verbose:
                print(f"System has {available_gb:.2f} GiB RAM available. Setting threshold to {memory_threshold_gb:.2f} GiB.")

        npx = len(phi0s)
        H, W = self.height, self.width
        a = np.zeros(npx)
        b = np.zeros(npx)
        phi0s_yx = np.array([phi0s[:, 1], phi0s[:, 0]]).T

        self.normalised_psf = self.psf
        self.psf_mask = create_boolean_circular_mask(self.normalised_psf.shape, radius=self.fwhm)
        self.vectorized_psf = self.normalised_psf[self.psf_mask]
        self.patch_y, self.patch_x = np.meshgrid(np.arange(-self.width/2, self.width/2), np.arange(-self.width/2, self.width/2))

            
        batch_size = self._calculate_batch_size(memory_threshold_gb, npx)
        indexed_phi0s = np.hstack([phi0s_yx, np.arange(npx).reshape(-1, 1)])
        batch_gen_stage1 = self._batch_generator(indexed_phi0s, batch_size)


        all_results_unordered = []
        for i, indexed_batch in enumerate(batch_gen_stage1):
            batch_results = self._compute_statistics_batch(indexed_batch, cpu)
            all_results_unordered.extend(batch_results)
            if self.verbose:
                print(f"  Statistics Calculation: Processed batch {i+1}...")

        all_results_ordered = sorted(all_results_unordered, key=lambda x: x[0])
        
        c_vals = [res[1] for res in all_results_ordered]
        d_vals = [res[2] for res in all_results_ordered]

        self.c_map = np.array(c_vals).reshape((W, H)).T
        self.d_map = np.array(d_vals).reshape((W, H, self.num_frames)).transpose(1, 0, 2)

        
        
        batch_gen_stage2 = self._batch_generator(indexed_phi0s, batch_size)

        for i, indexed_batch in enumerate(batch_gen_stage2):
            process_args = [(row[2].astype(int), row[:2], interpolation_order) for row in indexed_batch]
            
            if cpu > 1:
                results = pool_map(cpu, self.process_pixel, iterable(process_args))
            else:
                results = [self.process_pixel(arg) for arg in process_args]

            for i_res, a_val, b_val in results:
                if i_res is not None:
                    a[i_res] = a_val
                    b[i_res] = b_val
        
        if self.verbose:
            print(f"\nFast PACO is done!")
        return a, b
        
    def _compute_statistics_batch(self, indexed_phi0s_batch: np.ndarray, cpu: int) -> list:
        """Computes stats for a batch and returns them tagged with their original index."""
        phi0s_only_batch = indexed_phi0s_batch[:, :2]
        original_indices = indexed_phi0s_batch[:, 2].astype(int)

        p_data = pool_map(cpu, self.get_patch, iterable(phi0s_only_batch))
        patches = list(p_data)
        
        all_args = [(p, self.vectorized_psf) for p in patches]
        stats_results = pool_map(cpu, self._compute_all_terms_for_pixel, iterable(all_args))
        
        return [
            (original_indices[i], stats[2], stats[3]) 
            for i, stats in enumerate(stats_results)
        ]

    def process_pixel(self, args):
        """
        This is the ORIGINAL process_pixel logic. It is now used correctly
        as it relies on the pre-computed self.c_map and self.d_map.
        """
        i, p0, interpolation_order = args
        try:
            angles_px = get_rotated_pixel_coords(self.patch_y, self.patch_x, p0, self.angles)

            if np.any(angles_px < 0) or np.any(angles_px[:, 0] >= self.height) or np.any(angles_px[:, 1] >= self.width):
                return i, np.nan, np.nan

            coords_spatial = angles_px.T
            c_interpolated = map_coordinates(self.c_map, coords_spatial, order=interpolation_order, mode='nearest')
            a_val = np.sum(c_interpolated)

            time_coords = np.arange(self.num_frames)
            coords_d = np.vstack([coords_spatial, time_coords])
            d_interpolated = map_coordinates(self.d_map, coords_d, order=interpolation_order, mode='nearest')
            b_val = np.sum(d_interpolated)
            
            if np.isnan(a_val) or np.isnan(b_val):
                return i, np.nan, np.nan
            return i, a_val, b_val
        except Exception as e:
            if self.verbose:
                print(f"Error processing pixel {i} at {p0}: {e}")
            return i, np.nan, np.nan
    
    def _calculate_batch_size(self, memory_threshold_gb: float, npx: int) -> int:
        K = self.patch_area_pixels
        T = self.num_frames
        mem_per_pixel_bytes = (K * K + T) * 8 
        threshold_bytes = memory_threshold_gb * (1024**3)
        mem_per_patch = K * T * 8
        batch_size = int(threshold_bytes / ((mem_per_pixel_bytes + mem_per_patch) * 1.5))
        if self.verbose:
            print(f"Calculated batch size: {batch_size} pixels")
        return max(1, min(batch_size, npx))

    def _batch_generator(self, data: np.ndarray, batch_size: int) -> Generator[np.ndarray, None, None]:
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]       

    @staticmethod
    def _compute_all_terms_for_pixel(args):
        patch, psf_vectorized = args
        T = patch.shape[0]
        K = patch.shape[1]
        m, Cinv = compute_statistics_at_pixel(patch)
        if (m is None) or (Cinv is None) or np.any(np.isnan(patch)):
            return (None, None, np.nan, np.full(T, np.nan))
        w = Cinv @ psf_vectorized
        c = float(psf_vectorized.T @ w)
        d_vals = (patch - m) @ w
        return (m, Cinv, c, d_vals)