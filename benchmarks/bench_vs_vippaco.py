#!/usr/bin/env python3
"""
Comparison: our production shared-memory FullPACOv2 (radius-ring traversal,
src.full_paco) vs VIP's own PACO implementation (vip_hci.invprob.paco), on
real sph3 data, across an nproc sweep, measuring wall-clock time and peak
memory.

Why VIP's FastPACO is the right comparison target (not VIP's FullPACO):
VIP's FullPACO.PACOCalc explicitly does not implement multiprocessing at
all -- passing cpu > 1 just prints "Multiprocessing for full PACO is not
yet implemented!" and runs serially regardless (see
vip_hci/invprob/paco.py, class FullPACO). VIP's FastPACO is the only one of
their two implementations with real (if naively designed) multiprocessing:
its Stage 1 (compute_statistics_parallel) dispatches BOUND methods
(self.get_patch) to a fresh multiprocessing.Pool per call via pool_map --
the exact "pickles the whole `self`/cube per task" anti-pattern this
codebase's own FastPACOv2/FullPACOv2 started from and moved away from (see
those modules' docstrings). VIP's FastPACO Stage 2 (the per-pixel
PSF-shift-and-assemble sweep, using nearest-pixel Cinv/m lookups -- NOT the
spatial c/d-map interpolation our own FastPACOv2 does) is not parallelized
at all, serial regardless of cpu -- so VIP's achievable speedup has a hard
ceiling set by that always-serial stage, however many cores you give it.
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
FullPACOv2 has no such restriction and would happily take a subset, but
there is no fair "partial run" mode to compare against here: this is an
architectural fact about VIP's implementation, not a benchmark choice.
Expect this to be slow for VIP at nproc>1 -- its Stage 1
(compute_statistics_parallel) dispatches a bound method (self.get_patch)
per task, re-pickling the whole `self` (including the cube) on every one
of the 25600 sph3 pixels; the ORIGINAL variant in
bench_full_cube_comparison.py (a related but less naive anti-pattern --
Manager-dict, not per-task self-pickling) already took 642-824s on this
same dataset, so treat that as a lower bound on how slow VIP's Stage 1
alone could be.

Usage
-----
    python3 benchmarks/bench_vs_vippaco.py
    python3 benchmarks/bench_vs_vippaco.py --nproc 2,4,8,16
"""
import os
import time

from _common import (make_arg_parser, make_test_positions, load_dataset_from_args,
                     print_system_info, ResultTable, parse_nproc_list,
                     make_memory_monitor, mem_columns, mem_str)

from src.full_paco import FullPACOv2

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

    table = ResultTable(["impl", "nproc", "time_s", "mem_parent_mb",
                        "mem_maxworker_mb", "mem_combined_mb",
                        "mem_real_mb", "K"])

    for nproc in nprocs:
        print(f"--- nproc={nproc} ---")

        # Ours: current production shared-memory, ring-traversal FullPACOv2.
        # PACOCalc (not run()) so we time the exact same candidate set VIP
        # gets, bypassing our run()'s own full-grid meshgrid construction.
        ours = FullPACOv2(cube, pa, psf, fwhm=fwhm, verbose=False)
        with make_memory_monitor(args) as mon:
            t0 = time.perf_counter()
            ours.PACOCalc(phi0s, cpu=nproc, use_gpu=False)
            t_ours = time.perf_counter() - t0
        print(f"  {'OURS_FullPACOv2':20s} time={t_ours:9.2f}s  {mem_str(mon)}  "
              f"K={ours.patch_area_pixels}")
        table.add(impl="OURS_FullPACOv2", nproc=nproc, time_s=f"{t_ours:.2f}",
                  **mem_columns(mon), K=ours.patch_area_pixels)

        # VIP's FastPACO -- their only implementation with real multiprocessing.
        vip_obj = VIP_FastPACO(cube, pa, psf, fwhm=fwhm, pixscale=vip_pixscale,
                               verbose=False)
        with make_memory_monitor(args) as mon:
            t0 = time.perf_counter()
            vip_obj.PACOCalc(phi0s_vip, cpu=nproc)
            t_vip = time.perf_counter() - t0
        print(f"  {'VIP_FastPACO':20s} time={t_vip:9.2f}s  {mem_str(mon)}  "
              f"K={vip_obj.patch_area_pixels}")
        table.add(impl="VIP_FastPACO", nproc=nproc, time_s=f"{t_vip:.2f}",
                  **mem_columns(mon), K=vip_obj.patch_area_pixels)

        speedup = t_vip / t_ours if t_ours > 0 else float("nan")
        print(f"  -> speedup (VIP / ours): {speedup:.2f}x\n")

    table.print_table()
    if args.csv:
        table.write_csv(args.csv)


if __name__ == "__main__":
    main()
