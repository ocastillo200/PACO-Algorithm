#!/usr/bin/env python3
"""
Wall time and peak memory vs. dataset size for PACO (radius-ring
traversal, the production architecture), at a FIXED worker-process count.
This is the "pure dataset-size" experiment: nproc is held constant so the
only thing varying across rows is problem size, which keeps the time/
memory-vs-size curves uncontaminated by any core-count effect (see
benchmarks/bench_ring_speedup.py for that other axis, run separately on
purpose -- see the design discussion in the paper for why these two axes
are reported as separate experiments rather than a single combined grid:
clearer to read and far cheaper to run, at the cost of not measuring the
nproc x size interaction term).

--nproc has no default on purpose: run bench_ring_speedup.py FIRST and
pass whichever nproc showed the best parallel EFFICIENCY there (not
necessarily the highest core count available -- oversubscription can
make efficiency fall again at the top of the range), so this experiment
reports size scaling at a genuinely good operating point rather than an
arbitrary one.

Two independent sweep axes, both around a shared baseline shape (default
T=200, H=W=120, matching the nproc=1..63 sweep already reported in the
paper):
  - --spatial-sizes: H=W values tested at the baseline's T (pixel-count
    axis -- also the axis the ring-vs-noring memory-bound story in the
    paper's "Traversal order" section depends on).
  - --frame-counts: T values tested at the baseline's H, W (depth axis --
    drives per-pixel statistics cost, O(T*K^2 + K^3) per pixel,
    independent of field size).
Each axis is tested independently against the baseline, not as a full
cross-product.

If you later want to check whether the nproc x size interaction term
actually matters (e.g. whether a much larger field needs more cores to
saturate than the baseline does), pass a short comma list to --nproc
instead of a single value -- every shape is then run at every nproc in
that list, and a correctness cross-check between them is reported for
free. The default workflow is a single value, though.

Usage
-----
    # quick sanity check
    python3 benchmarks/bench_ring_size_scaling.py --nproc 4 \\
        --spatial-sizes 60,120 --frame-counts 100,200

    # the real server sweep, nproc=32 assumed as the efficiency-best
    # point found by bench_ring_speedup.py -- substitute your own
    python3 benchmarks/bench_ring_size_scaling.py --nproc 32 \\
        --spatial-sizes 60,120,200,400 \\
        --frame-counts 50,100,200,400,800 \\
        --memory-backend pss \\
        --csv ring_size_scaling.csv

Runtime note: this is len(spatial-sizes)+len(frame-counts)-1 shapes (the
baseline is shared, de-duplicated) times len(nproc) PACO runs. At
default settings with a single --nproc value that is 8 runs; a single
run on a large shape can itself take many minutes. --dry-run prints the
planned grid and exits without running anything.
"""
import argparse
import os
import time

from _common import (arrays_equal, make_synthetic_dataset, make_test_positions,
                     parse_nproc_list, print_system_info, ResultTable,
                     make_memory_monitor, mem_columns, mem_str)

from src.full_paco import PACO


def parse_shape_list(arg: str, dim_index: int, base_shape):
    """
    Parse a comma-separated list of single-axis values into full (T, H, W)
    shapes, substituting into `base_shape` at `dim_index` (0=T, 1=H=W
    spatial). Spatial values set BOTH H and W (square field).
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
    Baseline shape, then spatial-axis shapes (T fixed at baseline), then
    frame-axis shapes (H,W fixed at baseline) -- de-duplicated,
    order-preserving.
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
    parser.add_argument("--nproc", required=True,
                        help="Comma list (usually a single value) -- the "
                             "worker-process count held fixed across all "
                             "shapes. No default: pick the best-efficiency "
                             "value from bench_ring_speedup.py's output. "
                             "Pass a short list only if you also want the "
                             "nproc x size interaction cross-check (see "
                             "module docstring).")
    parser.add_argument("--base-shape", default="200,120,120",
                        help="T,H,W baseline shape both sweep axes vary "
                             "around (default: 200,120,120, matching the "
                             "nproc sweep already in the paper).")
    parser.add_argument("--spatial-sizes", default="60,120,200,400",
                        help="Comma list of H=W values (pixel-count "
                             "axis), tested at the baseline's T.")
    parser.add_argument("--frame-counts", default="50,100,200,400,800",
                        help="Comma list of T values (depth axis), "
                             "tested at the baseline's H,W.")
    parser.add_argument("--fwhm", type=float, default=3.72,
                        help="FWHM in pixels (default: 3.72, sph3's "
                             "value) -- fixed across all shapes; only T, "
                             "H, W vary.")
    parser.add_argument("--sampling", choices=["full", "dense", "random"],
                        default="full",
                        help="Candidate-position sampling (default: "
                             "'full', every native-grid pixel -- matches "
                             "a real detection run).")
    parser.add_argument("--npix", type=int, default=None,
                        help="Candidate-position count when --sampling is "
                             "not 'full'. Ignored for 'full'.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-gpu", dest="use_gpu", action="store_true", default=None,
                        help="Force GPU statistics prefill on.")
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
                             "exact-in-principle alternative.")
    parser.add_argument("--csv", default=None,
                        help="Optional path to write the full results "
                             "table as CSV.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned (shape, nproc) grid and "
                             "exit without running anything.")
    args = parser.parse_args()

    base_shape = tuple(int(x) for x in args.base_shape.split(","))
    grid = build_shape_grid(base_shape, args.spatial_sizes, args.frame_counts)
    nprocs = parse_nproc_list(args.nproc, os.cpu_count())

    print_system_info()
    print(f"\nFixed nproc: {nprocs}")
    print(f"Baseline shape (T,H,W): {base_shape}")
    print(f"Shape grid ({len(grid)} shapes):")
    for shape, axis in grid:
        T, H, W = shape
        npos = H * W if args.sampling == "full" else (args.npix or "?")
        print(f"  {axis:9s} T={T:5d} H={H:5d} W={W:5d}  candidates~{npos}")
    print(f"Total runs: {len(grid) * len(nprocs)}\n")

    if args.dry_run:
        print("--dry-run: exiting without running anything.")
        return

    table = ResultTable(["axis", "T", "H", "W", "n_candidates", "nproc",
                        "time_s", "mem_parent_mb", "mem_maxworker_mb",
                        "mem_combined_mb", "mem_real_mb", "hit_rate",
                        "correct"])

    calc_kwargs = {}
    if args.use_gpu is not None:
        calc_kwargs["use_gpu"] = args.use_gpu

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

        reference_ab = None  # only meaningful when len(nprocs) > 1

        for nproc in nprocs:
            run_counter += 1
            elapsed_script = time.perf_counter() - script_t0
            print(f">>> [{run_counter}/{total_runs}] nproc={nproc} "
                  f"(script running {elapsed_script/60:.1f} min so far)",
                  flush=True)

            obj = PACO(cube, pa, psf, fwhm=args.fwhm, verbose=False)
            with make_memory_monitor(args) as mon:
                t0 = time.perf_counter()
                a, b = obj.PACOCalc(phi0s, cpu=nproc, **calc_kwargs)
                elapsed = time.perf_counter() - t0

            if len(nprocs) == 1:
                ok_str = "n/a"
            elif reference_ab is None:
                reference_ab = (a, b)
                ok_str = "OK"
            else:
                ok, _ = arrays_equal(reference_ab[0], reference_ab[1], a, b)
                ok_str = "OK" if ok else "MISMATCH"

            hit_rate = getattr(obj, "last_cache_stats", None)
            hit_rate_str = f"{hit_rate['hit_rate']*100:.1f}%" if hit_rate else "n/a"

            print(f"  time={elapsed:8.2f}s  {mem_str(mon)}  "
                  f"hit_rate={hit_rate_str}  correct={ok_str}", flush=True)

            table.add(axis=axis, T=T, H=H, W=W, n_candidates=n_candidates,
                      nproc=nproc, time_s=f"{elapsed:.2f}",
                      **mem_columns(mon),
                      hit_rate=hit_rate_str, correct=ok_str)
        print()

    table.print_table()
    if args.csv:
        table.write_csv(args.csv)

    total_elapsed = time.perf_counter() - script_t0
    print(f"\nTotal wall time: {total_elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
