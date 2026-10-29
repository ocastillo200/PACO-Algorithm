#!/usr/bin/env python3
"""
Benchmark + correctness check: FastPACOv2 original design (bound-method
dispatch to pool_map(), fresh Pool per batch per stage) vs the current
module-level-function + fork-inherited-globals design
(src.fast_paco.FastPACOv2).

Usage
-----
    python3 benchmarks/bench_fast_paco.py
    python3 benchmarks/bench_fast_paco.py --nproc 2,4,8,16,32
    python3 benchmarks/bench_fast_paco.py --npix 25600      # full native grid
    python3 benchmarks/bench_fast_paco.py --data synthetic --shape 4800,200,200

--force-batch-size lets you simulate the many-small-batches regime that a
large cube's memory-based batch estimator would naturally trigger, without
needing that much RAM/data on hand — on the dev machine this was necessary
to even see the effect (natural batch size covered the whole sph3 field in
one go). On a server with real large data this shouldn't be needed.
"""
import time

import numpy as np
from vip_hci.config.utils_conf import iterable, pool_map
from vip_hci.fm import normalize_psf

from _common import (arrays_equal, make_arg_parser, make_test_positions,
                     load_dataset_from_args, print_system_info, ResultTable)

from src.paco_base import PACO
from src.fast_paco import FastPACOv2
from src.math_utils import compute_statistics_at_pixel
from src.utils import create_boolean_circular_mask, get_rotated_pixel_coords


# ═════════════════════════════════════════════════════════════════════════
# Original FastPACOv2 design, preserved here for comparison (no longer in
# src/fast_paco.py — see that file's module docstring for why).
# ═════════════════════════════════════════════════════════════════════════
class FastPACOv2_Original(PACO):
    def PACOCalc(self, phi0s, interpolation_order=1, cpu=1,
                 memory_threshold_gb=None, **kwargs):
        import psutil
        if memory_threshold_gb is None:
            memory_threshold_gb = psutil.virtual_memory().available / 1024**3 * 0.9
        npx = len(phi0s)
        H, W = self.height, self.width
        a = np.zeros(npx); b = np.zeros(npx)
        phi0s_yx = phi0s

        self.normalised_psf = normalize_psf(self.psf, verbose=False,
                                            force_odd=False, model='airy')
        self.psf_mask = create_boolean_circular_mask(
            self.normalised_psf.shape, radius=self.patch_width // 2)
        self.vectorized_psf = self.normalised_psf[self.psf_mask]
        self.patch_y, self.patch_x = np.meshgrid(
            np.arange(-self.width / 2, self.width / 2),
            np.arange(-self.height / 2, self.height / 2))

        batch_size = self._calculate_batch_size(memory_threshold_gb, npx)
        indexed_phi0s = np.hstack([phi0s_yx, np.arange(npx).reshape(-1, 1)])

        all_results_unordered = []
        for indexed_batch in self._batch_generator(indexed_phi0s, batch_size):
            batch_results = self._compute_statistics_batch(indexed_batch, cpu)
            all_results_unordered.extend(batch_results)

        all_results_ordered = sorted(all_results_unordered, key=lambda x: x[0])
        c_vals = [res[1] for res in all_results_ordered]
        d_vals = [res[2] for res in all_results_ordered]
        self.c_map = np.array(c_vals).reshape((W, H)).T
        self.d_map = np.array(d_vals).reshape((W, H, self.num_frames)).transpose(1, 0, 2)

        for indexed_batch in self._batch_generator(indexed_phi0s, batch_size):
            process_args = [(int(row[2]), row[:2], interpolation_order)
                            for row in indexed_batch]
            if cpu > 1:
                results = pool_map(cpu, self.process_pixel, iterable(process_args))
            else:
                results = [self.process_pixel(arg) for arg in process_args]
            for i_res, a_val, b_val in results:
                if i_res is not None:
                    a[i_res] = a_val; b[i_res] = b_val
        return a, b

    def _compute_statistics_batch(self, indexed_phi0s_batch, cpu):
        phi0s_only = indexed_phi0s_batch[:, :2]
        original_indices = indexed_phi0s_batch[:, 2].astype(int)
        raw_patches = list(pool_map(cpu, self.get_patch, iterable(phi0s_only)))
        all_args = [(p, self.vectorized_psf) for p in raw_patches]
        stats_results = pool_map(cpu, self._compute_all_terms_for_pixel,
                                 iterable(all_args))
        return [(original_indices[i], stats[2], stats[3])
                for i, stats in enumerate(stats_results)]

    @staticmethod
    def _compute_all_terms_for_pixel(args):
        patch, psf_vectorized = args
        T = patch.shape[0]
        m, Cinv = compute_statistics_at_pixel(patch)
        if (m is None) or (Cinv is None) or np.any(np.isnan(patch)):
            return (None, None, np.nan, np.full(T, np.nan))
        w = Cinv @ psf_vectorized
        c = float(psf_vectorized.T @ w)
        d_vals = (patch - m) @ w
        return (m, Cinv, c, d_vals)

    def process_pixel(self, args):
        from scipy.ndimage import map_coordinates
        i, p0, interpolation_order = args
        try:
            angles_px = get_rotated_pixel_coords(self.patch_y, self.patch_x,
                                                 p0, self.angles)
            if (np.any(angles_px < 0) or
                    np.any(angles_px[:, 0] >= self.width) or
                    np.any(angles_px[:, 1] >= self.height)):
                return i, np.nan, np.nan
            coords_spatial = np.array([angles_px[:, 1], angles_px[:, 0]])
            c_interpolated = map_coordinates(self.c_map, coords_spatial,
                                             order=interpolation_order, mode='nearest')
            a_val = np.sum(c_interpolated)
            time_coords = np.arange(self.num_frames)
            coords_d = np.vstack([coords_spatial, time_coords])
            d_interpolated = map_coordinates(self.d_map, coords_d,
                                             order=interpolation_order, mode='nearest')
            b_val = np.sum(d_interpolated)
            if np.isnan(a_val) or np.isnan(b_val):
                return i, np.nan, np.nan
            return i, a_val, b_val
        except Exception:
            return i, np.nan, np.nan

    def _calculate_batch_size(self, memory_threshold_gb, npx):
        K = self.patch_area_pixels
        T = self.num_frames
        mem_per_pixel_bytes = (K * K + T) * 8
        mem_per_patch = K * T * 8
        threshold_bytes = memory_threshold_gb * 1024**3
        batch_size = int(threshold_bytes / ((mem_per_pixel_bytes + mem_per_patch) * 1.5))
        return max(1, min(batch_size, npx))

    def _batch_generator(self, data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]


class _ForcedBatchMixin:
    """Force a fixed batch size (class-level override; see bench notes on
    why this must not be an instance-attribute monkeypatch)."""
    _forced_batch_size = None

    def _calculate_batch_size(self, *a, **k):
        return self._forced_batch_size


def main():
    parser = make_arg_parser(__doc__)
    parser.add_argument("--force-batch-size", type=int, default=None,
                        help="Force a fixed Stage-1/2 batch size (simulates "
                             "the many-batches regime a large cube would "
                             "naturally trigger).")
    args = parser.parse_args()
    print_system_info()

    cube, pa, psf = load_dataset_from_args(args)
    fwhm = args.fwhm
    T, H, W = cube.shape
    npix = args.npix if args.npix is not None else H * W  # full native grid
    phi0s = make_test_positions(H, W, npix, seed=args.seed, sampling=args.sampling)

    print(f"Dataset: cube={cube.shape}, fwhm={fwhm}, N test positions={len(phi0s)}\n")

    OldCls, NewCls = FastPACOv2_Original, FastPACOv2
    if args.force_batch_size is not None:
        OldCls = type("FastPACOv2_Original_Forced", (_ForcedBatchMixin, FastPACOv2_Original), {})
        NewCls = type("FastPACOv2_Forced", (_ForcedBatchMixin, FastPACOv2), {})
        print(f"Forcing batch_size={args.force_batch_size}\n")

    from _common import parse_nproc_list
    import os
    nprocs = parse_nproc_list(args.nproc, os.cpu_count())
    print(f"nproc sweep: {nprocs}\n")

    table = ResultTable(["nproc", "old_s", "new_s", "speedup", "verdict", "correct"])

    for nproc in nprocs:
        old = OldCls(cube, pa, psf, fwhm=fwhm, verbose=False)
        new = NewCls(cube, pa, psf, fwhm=fwhm, verbose=False)
        if args.force_batch_size is not None:
            old._forced_batch_size = args.force_batch_size
            new._forced_batch_size = args.force_batch_size

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
