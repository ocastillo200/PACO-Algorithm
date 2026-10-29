#!/usr/bin/env python3
"""
Speedup/efficiency vs. worker-process count for FullPACOv2 (radius-ring
traversal, the production architecture), at a SINGLE fixed, representative
dataset shape. This is the "pure nproc" experiment: dataset size is held
constant so the only thing varying across rows is core count, which keeps
the speedup/efficiency curve uncontaminated by any dataset-size effect
(see benchmarks/bench_ring_size_scaling.py for that other axis, run
separately on purpose -- see the design discussion in the paper's
"Scaling with the number of worker processes" section for why these two
axes are reported as separate experiments rather than a single combined
grid: it is both clearer to read and far cheaper to run, at the cost of
not measuring the nproc x size interaction term, which is covered
qualitatively elsewhere in the paper).

Default shape (T=200, H=W=120) matches the nproc=1..63 sweep already
reported in the paper.

Usage
-----
    # quick sanity check
    python3 benchmarks/bench_ring_speedup.py --nproc 1,4 --shape 60,40,40

    # the real server sweep
    python3 benchmarks/bench_ring_speedup.py \\
        --nproc 1,2,4,8,16,32,63 \\
        --shape 200,120,120 \\
        --memory-backend pss \\
        --csv ring_speedup.csv

After this, use whichever nproc shows the best parallel efficiency (not
necessarily the highest core count -- oversubscription can make
efficiency fall again at the top of the range) as the fixed --nproc for
benchmarks/bench_ring_size_scaling.py.
"""
import argparse
import os
import time

from _common import (arrays_equal, make_synthetic_dataset, make_test_positions,
                     parse_nproc_list, print_system_info, ResultTable,
                     make_memory_monitor, mem_columns, mem_str)

from src.full_paco import FullPACOv2


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", default="200,120,120",
                        help="T,H,W -- the single fixed shape this "
                             "experiment holds constant while nproc "
                             "varies (default: 200,120,120, matching "
                             "the sweep already in the paper).")
    parser.add_argument("--nproc", default="auto",
                        help="Comma list (e.g. '2,4,8') or 'auto' to "
                             "sweep powers of 2 up to os.cpu_count() "
                             "(default: auto).")
    parser.add_argument("--fwhm", type=float, default=3.72,
                        help="FWHM in pixels (default: 3.72, sph3's "
                             "value).")
    parser.add_argument("--sampling", choices=["full", "dense", "random"],
                        default="full",
                        help="Candidate-position sampling (default: "
                             "'full', every native-grid pixel -- "
                             "matches a real detection run).")
    parser.add_argument("--npix", type=int, default=None,
                        help="Candidate-position count when --sampling "
                             "is not 'full'. Ignored for 'full'.")
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
                        help="Optional path to write the results table "
                             "as CSV.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned nproc sweep and exit "
                             "without running anything.")
    args = parser.parse_args()

    T, H, W = (int(x) for x in args.shape.split(","))
    nprocs = parse_nproc_list(args.nproc, os.cpu_count())

    print_system_info()
    print(f"\nFixed shape (T,H,W): ({T}, {H}, {W})")
    print(f"nproc sweep ({len(nprocs)} values): {nprocs}")
    print(f"Total runs: {len(nprocs)}\n")

    if args.dry_run:
        print("--dry-run: exiting without running anything.")
        return

    cube, pa, psf = make_synthetic_dataset(T, H, W, fwhm=args.fwhm, seed=args.seed)
    npix = H * W if args.sampling == "full" else (args.npix or H * W)
    phi0s = make_test_positions(H, W, npix, seed=args.seed, sampling=args.sampling)
    n_candidates = len(phi0s)
    print(f"{n_candidates} candidates\n")

    table = ResultTable(["nproc", "time_s", "speedup", "efficiency",
                        "mem_parent_mb", "mem_maxworker_mb", "mem_combined_mb",
                        "mem_real_mb", "hit_rate", "correct"])

    calc_kwargs = {}
    if args.use_gpu is not None:
        calc_kwargs["use_gpu"] = args.use_gpu

    script_t0 = time.perf_counter()
    reference_ab = None
    t_nproc1 = None

    for i, nproc in enumerate(nprocs):
        elapsed_script = time.perf_counter() - script_t0
        print(f">>> [{i+1}/{len(nprocs)}] nproc={nproc} "
              f"(script running {elapsed_script/60:.1f} min so far)", flush=True)

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

        table.add(nproc=nproc, time_s=f"{elapsed:.2f}",
                  speedup=f"{speedup:.2f}", efficiency=f"{efficiency:.3f}",
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
