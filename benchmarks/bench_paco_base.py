#!/usr/bin/env python3
"""
Benchmark + correctness check: PACO.compute_statistics_parallel original
design (two separate pool_map() Pools: get_patch, then
compute_statistics_at_pixel) vs the current fused single-Pool design
(src.paco_base.PACO).

Usage
-----
    python3 benchmarks/bench_paco_base.py
    python3 benchmarks/bench_paco_base.py --nproc 2,4,8,16,32
    python3 benchmarks/bench_paco_base.py --data synthetic --shape 4800,200,200
"""
import multiprocessing as mp
import time

import numpy as np
from vip_hci.config.utils_conf import iterable, pool_map

from _common import (arrays_equal, make_arg_parser, make_test_positions,
                     load_dataset_from_args, print_system_info, ResultTable)

from src.paco_base import PACO
from src.math_utils import compute_statistics_at_pixel
from src.utils import create_boolean_circular_mask


def old_compute_statistics_parallel(paco_obj, phi0s, cpu):
    """Original: two separate pool_map() calls (get_patch, then stats)."""
    p_data = pool_map(cpu, paco_obj.get_patch, iterable(phi0s))
    raw_patches = list(p_data)
    data = pool_map(cpu, compute_statistics_at_pixel, iterable(raw_patches))
    return data


def old_run(paco_obj, phi0s, cpu):
    if cpu == 1:
        raw = [paco_obj.get_patch(p0) for p0 in phi0s]
        return [compute_statistics_at_pixel(p) for p in raw]
    return old_compute_statistics_parallel(paco_obj, phi0s, cpu)


def new_run(paco_obj, phi0s, cpu):
    if cpu == 1:
        Cinv, m, patches = paco_obj.compute_statistics(phi0s)
    else:
        Cinv, m, patches = paco_obj.compute_statistics_parallel(phi0s, cpu=cpu)
    return Cinv, m


def main():
    parser = make_arg_parser(__doc__)
    args = parser.parse_args()
    print_system_info()

    cube, pa, psf = load_dataset_from_args(args)
    fwhm = args.fwhm
    T, H, W = cube.shape
    npix = args.npix if args.npix is not None else H * W
    phi0s = make_test_positions(H, W, npix, seed=args.seed, sampling=args.sampling).astype(int)

    print(f"Dataset: cube={cube.shape}, fwhm={fwhm}, N test positions={len(phi0s)}\n")

    from _common import parse_nproc_list
    import os
    nprocs = parse_nproc_list(args.nproc, os.cpu_count())
    print(f"nproc sweep: {nprocs}\n")

    table = ResultTable(["nproc", "old_s", "new_s", "speedup", "verdict"])

    for nproc in nprocs:
        paco_old = PACO(cube, pa, psf, fwhm=fwhm, verbose=False)
        paco_new = PACO(cube, pa, psf, fwhm=fwhm, verbose=False)

        t0 = time.perf_counter()
        old_stats = old_run(paco_old, phi0s, nproc)
        t_old = time.perf_counter() - t0

        t0 = time.perf_counter()
        Cinv_new, m_new = new_run(paco_new, phi0s, nproc)
        t_new = time.perf_counter() - t0

        # Compare a sample of results for correctness (m, Cinv per pixel).
        ok = True
        for p0, (m_old, Cinv_old) in list(zip(phi0s, old_stats))[:200]:
            r, c = int(p0[0]), int(p0[1])
            if m_old is None:
                continue
            if not (np.allclose(m_old, m_new[r, c], equal_nan=True) and
                    np.allclose(Cinv_old, Cinv_new[r, c], equal_nan=True)):
                ok = False
                break

        speedup = t_old / t_new if t_new > 0 else float("nan")
        verdict = "GAIN" if speedup > 1.05 else ("TIE" if speedup > 0.95 else "REGRESSION")
        print(f"nproc={nproc}: OLD={t_old:.2f}s NEW={t_new:.2f}s "
              f"speedup={speedup:.2f}x [{verdict}] correct={ok}")

        table.add(nproc=nproc, old_s=f"{t_old:.2f}", new_s=f"{t_new:.2f}",
                  speedup=f"{speedup:.2f}x", verdict=verdict)

    print()
    table.print_table()
    if args.csv:
        table.write_csv(args.csv)


if __name__ == "__main__":
    main()
