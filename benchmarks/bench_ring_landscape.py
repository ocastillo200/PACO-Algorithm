#!/usr/bin/env python3
"""
Scaling landscape for FullPACOv2 (radius-ring traversal, the production
architecture -- see Sect. "Parallel architecture" of the accompanying
paper) ONLY: no ORIGINAL/Manager-dict or SHM_NORING comparison here (that
story is already told by benchmarks/bench_full_cube_comparison.py and
Sect. "Parallel performance" of the paper). This script answers a
different question: for the architecture we actually ship, how do wall
time, peak memory, and parallel speedup/efficiency behave as a function
of (a) field size (candidate-position / pixel count), (b) cube depth
(number of ADI frames, T), and (c) worker-process count -- independently
of each other, so each axis's effect is visible on its own rather than
folded into a single combined sweep.

Two independent sweep axes, both around a shared baseline shape (default
T=200, H=W=120, matching the nproc=1..63 sweep already reported in the
paper's Sect. "Scaling with the number of worker processes"):
  - --spatial-sizes: H=W values tested at the baseline's T (pixel-count
    axis -- this is also the axis the ring-vs-noring memory-bound story
    in Sect. "Traversal order" depends on: a larger field means more
    candidates per ring and a larger background-pixel working set).
  - --frame-counts: T values tested at the baseline's H, W (depth axis --
    this is the axis that drives per-pixel statistics cost: patch
    covariance/shrinkage/inversion is O(T*K^2 + K^3) per pixel,
    independent of field size).
Every shape in both axes is swept across every value of --nproc, so the
output directly shows how the speedup/efficiency curve (already reported
for one shape in the paper) shifts as problem size changes along each
axis -- e.g. whether a larger field needs more cores to saturate, or
whether per-pixel cost growth (T) changes the serial-fraction/Amdahl
picture.

A lightweight correctness check (np.allclose against the first nproc
tested for each shape) is still run, since reusing the same shape across
many nproc values is a cheap, free opportunity to catch a regression --
not because this script is comparing architectures.

Usage
-----
    # quick sanity sweep (small shapes/nproc), run this FIRST
    python3 benchmarks/bench_ring_landscape.py --nproc 1,4 --spatial-sizes 60,120 --frame-counts 100,200

    # the real server sweep (this can take a long time -- see the module
    # docstring's runtime note below before launching the full thing)
    python3 benchmarks/bench_ring_landscape.py \\
        --nproc 1,2,4,8,16,32,63 \\
        --spatial-sizes 60,120,200,400 \\
        --frame-counts 50,100,200,400,800 \\
        --memory-backend pss \\
        --csv ring_landscape.csv

Runtime note: this is len(spatial-sizes)+len(frame-counts)-1 shapes (the
baseline is shared) times len(nproc) FullPACOv2 runs. At default settings
that is 8 shapes x 7 nproc values = 56 runs; a single run on a large
shape at nproc=1 can itself take many minutes (see the paper's own
nproc=1 timings). Start with a reduced --nproc/--spatial-sizes/
--frame-counts list to estimate your machine's per-run cost before
committing to the full grid; --dry-run prints the planned grid and exits
without running anything.
"""
import argparse
import csv
import os
import time

import numpy as np

from _common import (arrays_equal, make_synthetic_dataset, make_test_positions,
                     parse_nproc_list, print_system_info, ResultTable,
                     make_memory_monitor, mem_columns, mem_str)

from src.full_paco import FullPACOv2


def parse_shape_list(arg: str, dim_index: int, base_shape):
    """
    Parse a comma-separated list of single-axis values (e.g. "60,120,200")
    into full (T, H, W) shapes, substituting into `base_shape` at
    `dim_index` (0=T, 1=H=W spatial). Spatial values set BOTH H and W
    (square field).
    """
    shapes = []
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        val = int(tok)
        shape = list(base_shape)
        if dim_index == 0:
            shape[0] = val
        else:
            shape[1] = val
            shape[2] = val
        shapes.append(tuple(shape))
    return shapes


def build_shape_grid(base_shape, spatial_sizes_arg, frame_counts_arg):
    """
    Baseline shape, then spatial-axis shapes (T fixed at baseline),
    then frame-axis shapes (H,W fixed at baseline) -- de-duplicated,
    order-preserving, so the baseline is never run twice just because it
    also appears in one of the axis lists.
    """
    seen = set()
    grid = []

    def add(shape, axis):
        if shape not in seen:
            seen.add(shape)
            grid.append((shape, axis))

    add(tuple(base_shape), "baseline")
    for shape in parse_shape_list(spatial_sizes_arg, 1, base_shape):
        add(shape, "spatial")
    for shape in parse_shape_list(frame_counts_arg, 0, base_shape):
        add(shape, "frames")
    return grid


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-shape", default="200,120,120",
                        help="T,H,W baseline shape both sweep axes vary "
                             "around (default: 200,120,120, matching the "
                             "nproc=1..63 sweep already in the paper).")
    parser.add_argument("--spatial-sizes", default="60,120,200,400",
                        help="Comma list of H=W values (pixel-count axis), "
                             "tested at the baseline's T.")
    parser.add_argument("--frame-counts", default="50,100,200,400,800",
                        help="Comma list of T values (depth axis), tested "
                             "at the baseline's H,W.")
    parser.add_argument("--nproc", default="auto",
                        help="Comma list (e.g. '2,4,8') or 'auto' to sweep "
                             "powers of 2 up to os.cpu_count() (default: "
                             "auto).")
    parser.add_argument("--fwhm", type=float, default=3.72,
                        help="FWHM in pixels (default: 3.72, sph3's "
                             "value) -- fixed across all shapes; only T, "
                             "H, W vary.")
    parser.add_argument("--sampling", choices=["full", "dense", "random"],
                        default="full",
                        help="Candidate-position sampling (default: "
                             "'full', every native-grid pixel -- matches "
                             "what a real detection run actually computes; "
                             "see benchmarks/_common.py:make_test_positions "
                             "for the others, useful only to shrink "
                             "runtime during a quick sanity sweep).")
    parser.add_argument("--npix", type=int, default=None,
                        help="Candidate-position count when --sampling is "
                             "not 'full'. Ignored for 'full'.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-gpu", dest="use_gpu", action="store_true", default=None,
                        help="Force GPU statistics prefill on (see "
                             "FullPACOv2's own default: on if a CUDA "
                             "device is available).")
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false",
                        help="Force GPU statistics prefill off.")
    parser.add_argument("--memory-backend", choices=["psutil", "pss", "cgroup"],
                        default="psutil",
                        help="See benchmarks/_common.py's "
                             "PssMemoryMonitor docstring. 'pss' "
                             "(RECOMMENDED on a Linux server) sums "
                             "de-duplicated memory across the process "
                             "tree with no special launch command "
                             "needed; 'cgroup' is the more fragile, "
                             "exact-in-principle alternative. Both fall "
                             "back to psutil with a warning if "
                             "unavailable.")
    parser.add_argument("--csv", default=None,
                        help="Optional path to write the full results "
                             "table as CSV.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned (shape, nproc) grid and "
                             "estimated run count, then exit without "
                             "running anything.")
    args = parser.parse_args()

    base_shape = tuple(int(x) for x in args.base_shape.split(","))
    grid = build_shape_grid(base_shape, args.spatial_sizes, args.frame_counts)
    nprocs = parse_nproc_list(args.nproc, os.cpu_count())

    print_system_info()
    print(f"\nBaseline shape (T,H,W): {base_shape}")
    print(f"Shape grid ({len(grid)} shapes):")
    for shape, axis in grid:
        T, H, W = shape
        npos = H * W if args.sampling == "full" else (args.npix or "?")
        print(f"  {axis:9s} T={T:5d} H={H:5d} W={W:5d}  candidates~{npos}")
    print(f"nproc sweep ({len(nprocs)} values): {nprocs}")
    print(f"Total runs: {len(grid) * len(nprocs)}\n")

    if args.dry_run:
        print("--dry-run: exiting without running anything.")
        return

    table = ResultTable(["axis", "T", "H", "W", "n_candidates", "nproc",
                        "time_s", "speedup_vs_nproc1", "efficiency",
                        "mem_parent_mb", "mem_maxworker_mb", "mem_combined_mb",
                        "mem_real_mb", "hit_rate", "correct"])

    script_t0 = time.perf_counter()
    run_counter = 0
    total_runs = len(grid) * len(nprocs)

    for shape, axis in grid:
        T, H, W = shape
        cube, pa, psf = make_synthetic_dataset(T, H, W, fwhm=args.fwhm, seed=args.seed)
        npix = H * W if args.sampling == "full" else (args.npix or H * W)
        phi0s = make_test_positions(H, W, npix, seed=args.seed, sampling=args.sampling)
        n_candidates = len(phi0s)

        print(f"=== shape ({axis}): T={T} H={H} W={W}, "
              f"{n_candidates} candidates ===", flush=True)

        reference_ab = None
        t_nproc1 = None  # wall time at the first (smallest) nproc tested, for speedup

        for nproc in nprocs:
            run_counter += 1
            elapsed_script = time.perf_counter() - script_t0
            print(f">>> [{run_counter}/{total_runs}] nproc={nproc} "
                  f"(script running {elapsed_script/60:.1f} min so far)",
                  flush=True)

            calc_kwargs = {}
            if args.use_gpu is not None:
                calc_kwargs["use_gpu"] = args.use_gpu

            obj = FullPACOv2(cube, pa, psf, fwhm=args.fwhm, verbose=False)
            with make_memory_monitor(args) as mon:
                t0 = time.perf_counter()
                a, b = obj.PACOCalc(phi0s, cpu=nproc, **calc_kwargs)
                elapsed = time.perf_counter() - t0

            if reference_ab is None:
                reference_ab = (a, b)
                ok = True
            else:
                ok, _ = arrays_equal(reference_ab[0], reference_ab[1], a, b)

            if t_nproc1 is None:
                t_nproc1 = elapsed
                speedup = 1.0
            else:
                speedup = t_nproc1 / elapsed if elapsed > 0 else float("nan")
            efficiency = speedup / nproc if nproc > 0 else float("nan")

            hit_rate = getattr(obj, "last_cache_stats", None)
            hit_rate_str = f"{hit_rate['hit_rate']*100:.1f}%" if hit_rate else "n/a"

            print(f"  time={elapsed:8.2f}s  speedup={speedup:6.2f}x  "
                  f"efficiency={efficiency:5.2f}  {mem_str(mon)}  "
                  f"hit_rate={hit_rate_str}  correct={ok}", flush=True)

            table.add(axis=axis, T=T, H=H, W=W, n_candidates=n_candidates,
                      nproc=nproc, time_s=f"{elapsed:.2f}",
                      speedup_vs_nproc1=f"{speedup:.2f}",
                      efficiency=f"{efficiency:.3f}",
                      **mem_columns(mon),
                      hit_rate=hit_rate_str, correct=("OK" if ok else "MISMATCH"))
        print()

    table.print_table()
    if args.csv:
        table.write_csv(args.csv)

    total_elapsed = time.perf_counter() - script_t0
    print(f"\nTotal wall time: {total_elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
