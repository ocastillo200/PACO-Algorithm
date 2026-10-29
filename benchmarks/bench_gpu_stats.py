#!/usr/bin/env python3
"""
GPU statistics-batching verification + benchmark. Run this on the server —
it cannot be exercised at all on the dev machine this was written on (no
CuPy/CUDA device there), so this is the first real test of:

  1. math_utils.compute_statistics_batch(..., use_gpu=True) — the same
     batched (m, Cinv) math already validated on CPU (use_gpu=False) against
     the per-pixel reference (compute_statistics_at_pixel), now exercised
     with real CuPy array ops (batched einsum, batched linalg.inv/pinv,
     host<->device transfer).
  2. src.gpu.GPUStatisticsBatcher (VRAM-budgeted chunking on top of #1).
  3. The three call sites wired to use_gpu=True:
       - PACO.PACOCalc / compute_statistics_gpu (base Algorithm 1)
       - FastPACOv2.PACOCalc Stage 1 (compute_stage1_streamed: also folds in
         the PSF contraction to (c, d), see that method's docstring)
       - FullPACOv2.PACOCalc's per-ring GPU cache prefill

For each, this script runs CPU (use_gpu=False) and GPU (use_gpu=True) on the
identical dataset/positions and checks np.allclose on the final (a, b)
output — GPU must reproduce the CPU numbers, not just "run without
crashing". It also reports wall time for both so you can see whether GPU
batching is actually a net win at your cube's (T, K) size once transfer
overhead is included (small K especially may not be — see the printed
per-item byte estimate from GPUStatisticsBatcher's verbose output).

Also exercises the OOM-retry / per-chunk CPU-fallback path added to
src.gpu.GPUBatchDriver: pass a small --gpu-memory-fraction to force
small GPU batches (more chunk boundaries = more chances to see the
retry/fallback logic trigger for real under genuine VRAM pressure,
rather than only against the fake_cupy-based fault injection in
benchmarks/bench_gpu_fake_smoke.py). A UserWarning printed during a run
means a chunk could not be computed on the GPU and silently fell back to
CPU for that chunk only -- the run still completes and `correct` still
gets checked, but if you see these unexpectedly (not because you asked
for a tiny --gpu-memory-fraction), investigate before trusting the speed
numbers from that nproc/stage.

Usage
-----
    python3 benchmarks/bench_gpu_stats.py
    python3 benchmarks/bench_gpu_stats.py --npix 25600      # full native grid
    python3 benchmarks/bench_gpu_stats.py --data synthetic --shape 4800,200,200
    python3 benchmarks/bench_gpu_stats.py --nproc 1,4,8
    python3 benchmarks/bench_gpu_stats.py --gpu-memory-fraction 0.02  # force small batches
    python3 benchmarks/bench_gpu_stats.py --memory-backend pss --csv gpu_stats.csv
"""
import time

import numpy as np

from _common import (arrays_equal, make_arg_parser, make_test_positions,
                     load_dataset_from_args, print_system_info, ResultTable,
                     parse_nproc_list, make_memory_monitor, mem_columns, mem_str)

from src.gpu import GPU_AVAILABLE
from src.paco_base import PACO
from src.fast_paco import FastPACOv2
from src.full_paco import FullPACOv2


def main():
    parser = make_arg_parser(__doc__)
    parser.add_argument("--gpu-memory-fraction", type=float, default=None,
                        help="Fraction of currently-free VRAM "
                             "GPUStatisticsBatcher may use per batch "
                             "(see src.gpu.GPUBatchDriver). Default: "
                             "each class's own default (0.8). Pass a "
                             "small value (e.g. 0.02) to deliberately "
                             "force many small batches and exercise the "
                             "OOM-retry/CPU-fallback path under real VRAM "
                             "pressure instead of just the fake-GPU fault "
                             "injection in bench_gpu_fake_smoke.py.")
    args = parser.parse_args()
    print_system_info()

    if not GPU_AVAILABLE:
        print("\nGPU_AVAILABLE is False (no CuPy import / no CUDA device on "
              "this machine) — nothing to benchmark here. This script is "
              "meant to be run on the server.")
        return

    cube, pa, psf = load_dataset_from_args(args)
    fwhm = args.fwhm
    T, H, W = cube.shape

    npix = args.npix if args.npix is not None else H * W
    phi0s = make_test_positions(H, W, npix, seed=args.seed, sampling=args.sampling)
    print(f"Dataset: cube={cube.shape}, fwhm={fwhm}, N positions={len(phi0s)}\n")

    import os
    nprocs = parse_nproc_list(args.nproc, os.cpu_count())
    print(f"nproc sweep: {nprocs}\n")

    table = ResultTable(["stage", "nproc", "cpu_s", "gpu_s", "speedup",
                        "mem_parent_mb", "mem_maxworker_mb", "mem_combined_mb",
                        "mem_real_mb", "correct"])

    for nproc in nprocs:
        print(f"--- nproc={nproc} ---")

        # 1. Base PACO (Algorithm 1) statistics stage
        p_cpu = PACO(cube, pa, psf, fwhm=fwhm, verbose=False)
        t0 = time.perf_counter()
        a_cpu, b_cpu = p_cpu.PACOCalc(phi0s, cpu=nproc, use_gpu=False)
        t_cpu = time.perf_counter() - t0

        p_gpu = PACO(cube, pa, psf, fwhm=fwhm, verbose=False)
        with make_memory_monitor(args) as mon1:
            t0 = time.perf_counter()
            a_gpu, b_gpu = p_gpu.PACOCalc(phi0s, cpu=nproc, use_gpu=True,
                                          gpu_memory_fraction=args.gpu_memory_fraction)
            t_gpu = time.perf_counter() - t0

        ok, detail = arrays_equal(a_cpu, b_cpu, a_gpu, b_gpu)
        speedup = t_cpu / t_gpu if t_gpu > 0 else float("nan")
        print(f"  [PACO/Algorithm1] cpu={t_cpu:.2f}s gpu={t_gpu:.2f}s "
              f"speedup={speedup:.2f}x {mem_str(mon1)} correct={ok}{detail}")
        table.add(stage="PACO_base", nproc=nproc, cpu_s=f"{t_cpu:.2f}",
                  gpu_s=f"{t_gpu:.2f}", speedup=f"{speedup:.2f}x",
                  **mem_columns(mon1),
                  correct=("OK" if ok else "MISMATCH"))

        # 2. FastPACOv2 Stage 1
        f_cpu = FastPACOv2(cube, pa, psf, fwhm=fwhm, verbose=False)
        t0 = time.perf_counter()
        a_cpu2, b_cpu2 = f_cpu.PACOCalc(phi0s, cpu=nproc, use_gpu=False)
        t_cpu2 = time.perf_counter() - t0

        f_gpu = FastPACOv2(cube, pa, psf, fwhm=fwhm, verbose=False)
        with make_memory_monitor(args) as mon2:
            t0 = time.perf_counter()
            a_gpu2, b_gpu2 = f_gpu.PACOCalc(phi0s, cpu=nproc, use_gpu=True,
                                            gpu_memory_fraction=args.gpu_memory_fraction)
            t_gpu2 = time.perf_counter() - t0

        ok2, detail2 = arrays_equal(a_cpu2, b_cpu2, a_gpu2, b_gpu2)
        speedup2 = t_cpu2 / t_gpu2 if t_gpu2 > 0 else float("nan")
        print(f"  [FastPACOv2]      cpu={t_cpu2:.2f}s gpu={t_gpu2:.2f}s "
              f"speedup={speedup2:.2f}x {mem_str(mon2)} correct={ok2}{detail2}")
        table.add(stage="FastPACOv2", nproc=nproc, cpu_s=f"{t_cpu2:.2f}",
                  gpu_s=f"{t_gpu2:.2f}", speedup=f"{speedup2:.2f}x",
                  **mem_columns(mon2),
                  correct=("OK" if ok2 else "MISMATCH"))

        # 3. FullPACOv2 (ring traversal + GPU prefill)
        fp_cpu = FullPACOv2(cube, pa, psf, fwhm=fwhm, verbose=False)
        t0 = time.perf_counter()
        a_cpu3, b_cpu3 = fp_cpu.PACOCalc(phi0s, cpu=nproc, use_gpu=False)
        t_cpu3 = time.perf_counter() - t0

        fp_gpu = FullPACOv2(cube, pa, psf, fwhm=fwhm, verbose=False)
        with make_memory_monitor(args) as mon3:
            t0 = time.perf_counter()
            a_gpu3, b_gpu3 = fp_gpu.PACOCalc(phi0s, cpu=nproc, use_gpu=True,
                                             gpu_memory_fraction=args.gpu_memory_fraction)
            t_gpu3 = time.perf_counter() - t0

        ok3, detail3 = arrays_equal(a_cpu3, b_cpu3, a_gpu3, b_gpu3)
        speedup3 = t_cpu3 / t_gpu3 if t_gpu3 > 0 else float("nan")
        print(f"  [FullPACOv2]      cpu={t_cpu3:.2f}s gpu={t_gpu3:.2f}s "
              f"speedup={speedup3:.2f}x {mem_str(mon3)} correct={ok3}{detail3}")
        table.add(stage="FullPACOv2", nproc=nproc, cpu_s=f"{t_cpu3:.2f}",
                  gpu_s=f"{t_gpu3:.2f}", speedup=f"{speedup3:.2f}x",
                  **mem_columns(mon3),
                  correct=("OK" if ok3 else "MISMATCH"))
        print()

    table.print_table()
    if args.csv:
        table.write_csv(args.csv)


if __name__ == "__main__":
    main()
