#!/usr/bin/env python3
"""
E7 -- our PACO (shared-memory cache, radius-ring traversal) and
FastPACO (Algorithm 2 with interpolation) vs VIP's FastPACO
(vip_hci.invprob.paco), on real sph3 data, across an nproc sweep, with
repeated runs: wall-clock time (median, min, max) and peak memory.

Why VIP's FastPACO is the right comparison target (not VIP's FullPACO):
VIP's FullPACO.PACOCalc explicitly does not implement multiprocessing at
all -- passing cpu > 1 just prints "Multiprocessing for full PACO is not
yet implemented!" and runs serially regardless (see
vip_hci/invprob/paco.py, class FullPACO). VIP's FastPACO is the only one of
their two implementations with real (if naively designed) multiprocessing:
its Stage 1 (compute_statistics_parallel) dispatches a BOUND method
(self.get_patch) to a multiprocessing.Pool via pool_map, so the object
holding the cube is serialized with each chunk of tasks sent to the
workers. VIP's FastPACO Stage 2 (the per-pixel PSF-shift-and-assemble
sweep, using truncated-pixel Cinv/m lookups -- NOT the spatial c/d-map
interpolation of Algorithm 2 that our FastPACO does) runs serially
whatever cpu is, which by Amdahl's law limits its speed-up; this script
measures by how much.
In practice VIP's FastPACO and FullPACO compute almost the same thing
(both do a nearest-pixel lookup sweep); FastPACO just also offers a
(partially) parallel statistics precompute, making it VIP's best/only
real parallel-execution baseline to compare against.

IMPORTANT -- this does NOT check numerical agreement between VIP's output
and ours. VIP's compute_statistics_at_pixel uses a materially different
sample-covariance/shrinkage formula that earlier review in this project
found to be buggy relative to the paper (see src/math_utils.py's
docstrings). This script measures wall-clock time and peak memory only, as
a function of nproc, on the same real dataset and (as close as achievable)
the same patch size K.

K parity: VIP derives its circular patch-mask radius as
int(fwhm/pixscale) -- different from our int(2*fwhm)//2. --vip-pixscale
is auto-computed by default so VIP's radius exactly matches ours (modulo
any difference in the two codebases' create_boolean_circular_mask
implementations) -- the printed patch_area for both confirms whether K
actually came out equal.

Position count is ALWAYS the full native grid, regardless of --npix/
--sampling: VIP's compute_statistics_parallel/compute_statistics
hard-reshape their flat results into (width, height, ...) assuming phi0s
covers the *entire* image (see vip_hci/invprob/paco.py:1059) -- passing it
a partial subset raises ValueError (confirmed empirically). Our own
PACO has no such restriction and would happily take a subset, but
there is no fair "partial run" mode to compare against here: this is an
architectural fact about VIP's implementation, not a benchmark choice.
Expect VIP to be slow: one serial run took ~575 s and an 8-worker run
~695 s on this dataset (E9), so a full sweep with repeats takes hours.

Usage
-----
    python3 benchmarks/bench_vs_vippaco.py --nproc 1,2,4,8,16,32 --repeats 3 \
        --memory-backend pss --csv e7_vs_vip.csv
    python3 benchmarks/bench_vs_vippaco.py --nproc 8 --repeats 1   # quick check
"""
import os
import time

from _common import (make_arg_parser, make_test_positions, load_dataset_from_args,
                     print_system_info, ResultTable, parse_nproc_list,
                     make_memory_monitor, mem_columns, mem_str)

from src.fast_paco import FastPACO
from src.full_paco import PACO

try:
    from vip_hci.invprob.paco import FastPACO as VIP_FastPACO
except ImportError as exc:
    raise SystemExit(
        "Could not import vip_hci.invprob.paco.FastPACO -- is vip_hci "
        f"installed in this environment? ({exc})")


def main():
    parser = make_arg_parser(__doc__)
    parser.add_argument("--vip-pixscale", type=float, default=None,
                        help="Override VIP's pixscale (controls its patch "
                             "mask radius = int(fwhm/pixscale)). Default: "
                             "auto-computed so VIP's radius matches ours "
                             "(int(2*fwhm)//2) exactly.")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Runs per (implementation, nproc); the summary "
                             "reports the median, min and max time.")
    parser.add_argument("--skip-vip", action="store_true",
                        help="Time only our two implementations.")
    parser.add_argument("--impls", default="full,fast,vip",
                        help="Comma-separated subset of full,fast,vip. Run one "
                             "implementation per invocation to measure its "
                             "memory in a fresh process, unaffected by the "
                             "heap the others leave in the parent.")
    args = parser.parse_args()
    print_system_info()

    cube, pa, psf = load_dataset_from_args(args)
    fwhm = args.fwhm
    T, H, W = cube.shape

    # VIP's implementation requires the full native grid (see module
    # docstring) -- --npix/--sampling are intentionally not honored here.
    phi0s = make_test_positions(H, W, H * W, sampling="full")
    # VIP's own compute_statistics/PACOCalc index with raw p0 values
    # (e.g. `m[p0[1]][p0[0]]`, vip_hci/invprob/paco.py:866) without casting
    # to int -- it silently assumes integer-dtype phi0s (its own run()
    # builds phi0s from np.meshgrid(np.arange(...)), which is int64 by
    # default). Our make_test_positions returns float64 (needed elsewhere
    # for our own oversampling support), so confirmed empirically: passing
    # that straight to VIP raises IndexError. Lossless here since every
    # value is an exact integer -- give VIP its own int-cast copy, keep
    # using the float version for our own call.
    phi0s_vip = phi0s.astype(int)

    our_patch_width = int(2 * fwhm)
    our_mask_radius = our_patch_width // 2
    vip_pixscale = (args.vip_pixscale if args.vip_pixscale is not None
                    else fwhm / our_mask_radius)
    vip_mask_radius = int(fwhm / vip_pixscale)

    print(f"Dataset: cube={cube.shape}, fwhm={fwhm}, "
          f"N candidate positions={len(phi0s)} (FULL native grid — required by VIP)")
    print(f"Our mask radius: {our_mask_radius} (patch_width={our_patch_width})")
    print(f"VIP pixscale: {vip_pixscale:.4f} -> VIP mask radius: {vip_mask_radius}\n")

    nprocs = parse_nproc_list(args.nproc, os.cpu_count())
    print(f"nproc sweep: {nprocs}\n")

    table = ResultTable(["impl", "nproc", "repeat", "time_s", "mem_parent_mb",
                        "mem_maxworker_mb", "mem_combined_mb",
                        "mem_real_mb", "K"])

    def run_ours_full(nproc):
        # PACOCalc (not run()) so we time the exact candidate set VIP gets.
        obj = PACO(cube, pa, psf, fwhm=fwhm, verbose=False)
        obj.PACOCalc(phi0s, cpu=nproc, use_gpu=False)
        return obj.patch_area_pixels

    def run_ours_fast(nproc):
        # run() evaluates the full native grid, the same candidate set.
        obj = FastPACO(cube, pa, psf, fwhm=fwhm, verbose=False)
        obj.run(cpu=nproc)
        return obj.patch_area_pixels

    def run_vip(nproc):
        obj = VIP_FastPACO(cube, pa, psf, fwhm=fwhm, pixscale=vip_pixscale,
                           verbose=False)
        obj.PACOCalc(phi0s_vip, cpu=nproc)
        return obj.patch_area_pixels

    chosen = set(args.impls.split(","))
    if args.skip_vip:
        chosen.discard("vip")
    impls = [(n, f) for k, n, f in (("full", "OURS_FullPACOv2", run_ours_full),
                                    ("fast", "OURS_FastPACOv2", run_ours_fast),
                                    ("vip", "VIP_FastPACO", run_vip)) if k in chosen]

    times = {}
    for nproc in nprocs:
        print(f"--- nproc={nproc} ---", flush=True)
        for name, fn in impls:
            for rep in range(args.repeats):
                with make_memory_monitor(args) as mon:
                    t0 = time.perf_counter()
                    K = fn(nproc)
                    dt = time.perf_counter() - t0
                times.setdefault((name, nproc), []).append(dt)
                print(f"  {name:16s} repeat {rep + 1}/{args.repeats} "
                      f"time={dt:9.2f}s  {mem_str(mon)}  K={K}", flush=True)
                table.add(impl=name, nproc=nproc, repeat=rep, time_s=f"{dt:.2f}",
                          **mem_columns(mon), K=K)

    table.print_table()
    if args.csv:
        table.write_csv(args.csv)

    import statistics
    print("\n=== Summary: median [min, max] wall time (s) ===")
    names = [n for n, _ in impls]
    print(f"{'nproc':>6}  " + "".join(f"{n:>30}" for n in names)
          + ("   VIP/ours-Full  VIP/ours-Fast" if len(names) == 3 else ""))
    for nproc in nprocs:
        cells, med = [], {}
        for n in names:
            ts = times[(n, nproc)]
            med[n] = statistics.median(ts)
            cells.append(f"{med[n]:10.2f} [{min(ts):.2f}, {max(ts):.2f}]".rjust(30))
        line = f"{nproc:>6}  " + "".join(cells)
        if {"VIP_FastPACO", "OURS_FullPACOv2", "OURS_FastPACOv2"} <= set(names):
            line += (f"   {med['VIP_FastPACO'] / med['OURS_FullPACOv2']:13.1f}"
                     f"  {med['VIP_FastPACO'] / med['OURS_FastPACOv2']:13.1f}")
        print(line)
    for n in names:
        base = statistics.median(times[(n, nprocs[0])])
        print(f"speed-up of {n} relative to nproc={nprocs[0]}: "
              + ", ".join(f"{p}: {base / statistics.median(times[(n, p)]):.1f}"
                          for p in nprocs))


if __name__ == "__main__":
    main()
