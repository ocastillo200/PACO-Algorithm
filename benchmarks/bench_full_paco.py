#!/usr/bin/env python3
"""
Benchmark + correctness check: FullPACOv2 original design (fresh pool_map()
Pool per pixel batch, cross-process Manager().dict() cache storing full
arrays) vs the current shared-memory design (src.full_paco.FullPACOv2).

Usage
-----
    python3 benchmarks/bench_full_paco.py
    python3 benchmarks/bench_full_paco.py --nproc 2,4,8,16,32
    python3 benchmarks/bench_full_paco.py --data synthetic --shape 4800,200,200 --npix 3000
    python3 benchmarks/bench_full_paco.py --csv results_full_paco.csv

Always verifies a/b outputs are numerically identical (within float
tolerance) between OLD and NEW before trusting any timing number — a
faster-but-wrong result is worthless.

FullPACOv2's whole reason to exist is handling cases where precomputing
statistics for *every* pixel is too much memory (e.g. NIRC2-scale K); use
--shape/--fwhm to explore that regime (larger K = bigger patches = the
scenario this class targets) as well as large-T scalability (LMIRCam-scale).
"""
import multiprocessing
import time

import numpy as np

from _common import (arrays_equal, make_arg_parser, make_test_positions,
                     load_dataset_from_args, print_system_info, ResultTable)

from src.paco_base import PACO
from src.full_paco import FullPACOv2
from src.math_utils import al, bl, compute_statistics_at_pixel, wfl
from src.utils import create_boolean_circular_mask, get_rotated_pixel_coords


# ═════════════════════════════════════════════════════════════════════════
# Original FullPACOv2 design, preserved here for comparison since it no
# longer exists in src/full_paco.py (see that file's module docstring for
# the full history of intermediate designs and why each one was replaced).
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

        for i in range(0, npx, pixel_batch_size):
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


def main():
    parser = make_arg_parser(__doc__)
    args = parser.parse_args()
    print_system_info()

    cube, pa, psf = load_dataset_from_args(args)
    fwhm = args.fwhm
    T, H, W = cube.shape
    npix = args.npix if args.npix is not None else min(2000, H * W)
    phi0s = make_test_positions(H, W, npix, seed=args.seed, sampling=args.sampling)

    print(f"Dataset: cube={cube.shape}, fwhm={fwhm}, N test positions={len(phi0s)}\n")

    from _common import parse_nproc_list
    import os
    nprocs = parse_nproc_list(args.nproc, os.cpu_count())
    print(f"nproc sweep: {nprocs}\n")

    table = ResultTable(["nproc", "old_s", "new_s", "speedup", "verdict", "correct"])

    for nproc in nprocs:
        old = FullPACOv2_Original(cube, pa, psf, fwhm=fwhm, verbose=False)
        new = FullPACOv2(cube, pa, psf, fwhm=fwhm, verbose=False)

        t0 = time.perf_counter()
        a_old, b_old = old.PACOCalc(phi0s, cpu=nproc)
        t_old = time.perf_counter() - t0

        t0 = time.perf_counter()
        a_new, b_new = new.PACOCalc(phi0s, cpu=nproc)
        t_new = time.perf_counter() - t0

        ok, detail = arrays_equal(a_old, b_old, a_new, b_new)
        speedup = t_old / t_new if t_new > 0 else float("nan")
        verdict = "GAIN" if speedup > 1.05 else ("TIE" if speedup > 0.95 else "REGRESSION")
        print(f"nproc={nproc}: OLD={t_old:.2f}s NEW={t_new:.2f}s "
              f"speedup={speedup:.2f}x [{verdict}] correct={ok}{detail}")

        table.add(nproc=nproc, old_s=f"{t_old:.2f}", new_s=f"{t_new:.2f}",
                  speedup=f"{speedup:.2f}x", verdict=verdict,
                  correct=("OK" if ok else "MISMATCH"))

    print()
    table.print_table()
    if args.csv:
        table.write_csv(args.csv)


if __name__ == "__main__":
    main()
