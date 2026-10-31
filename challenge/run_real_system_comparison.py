#!/usr/bin/env python3
"""
Real-instrument comparison: our production ring-traversal PACO
(src.full_paco, radius-ring shared-memory cache) vs VIP's own FastPACO
(vip_hci.invprob.paco -- the "pyPACO" of the EIDC challenge), on a real,
non-EIDC ADI dataset with no formal ground truth, measuring:
  - detection capability (top-N S/N peaks per algorithm, and -- if you
    know one -- the S/N each algorithm assigns to a specific companion
    position; there is no EIDC-style injected-and-known ground truth here,
    so this is necessarily qualitative/comparative, not a scored metric)
  - wall-clock time
  - peak memory during execution, via benchmarks/_common.py's
    psutil/cgroup backends (see --memory-backend)

Only FastPACO is used on the VIP side (not VIP's plain PACO class), and
only the ring-traversal PACO on ours (not the NORING/ORIGINAL
variants benchmarks/bench_full_cube_comparison.py also exercises).
Reasoning (matches benchmarks/bench_vs_vippaco.py's own, more detailed
comment): VIP's plain PACO class does not implement multiprocessing at
all (it warns and runs serially regardless of the requested cpu count),
so FastPACO is VIP's only real parallel code path and the only fair
comparison target once nproc > 1 matters, which it does for a real,
full-size instrument cube. The ring variant is this project's own
production default (see the "Radius-ring traversal" section of the
accompanying paper) -- the NORING/ORIGINAL variants exist only to
demonstrate the parallel-architecture evolution on the EIDC benchmark,
not as something you'd actually deploy.

Expected input directory layout (pass via the required `data_dir`
argument) -- this dataset's own file-naming convention, distinct from
both the EIDC challenge's sphere_irdis_{cube,pa,psf}_3.fits names and the
firstepoch/secondepoch center_im-2.fits/finalPSF.fits convention already
used by benchmarks/_common.py:load_epoch_dataset. This is exactly the
layout and loading/preprocessing cycle used by the project's own
production single-system driver, example.py (repo root) -- this script
mirrors that cycle as closely as possible rather than the EIDC-benchmark
convention used elsewhere in challenge/, since the two are genuinely
different pipelines for different purposes (see the PSF-normalization
note below):
    <data_dir>/center_im.fits      -- (T, H, W) cube ready for derotation
                                      (or (2, T, H, W) for an unsplit
                                      dual-band IRDIS cube -- see below)
    <data_dir>/median_unsat.fits   -- the unsaturated PSF, already
                                      median-combined upstream (2D; matched
                                      to the same dual-band case as cube)
    <data_dir>/rotnth.fits         -- (T,) parallactic angles, in degrees

Loaded with vip_hci.fits.open_fits (matching example.py exactly, INCLUDING
its float32 default precision -- not upcast to float64 here, since this
mirrors an established, working pipeline rather than an independent
numerical-fidelity design choice). If the loaded cube has more than 3
dimensions (an unsplit dual-band IRDIS acquisition), both cube and psf are
collapsed by taking index 0 along the leading axis, exactly as example.py
does -- NOT a median, since that extra axis indexes a discrete channel,
not repeated noisy measurements of the same quantity.

PSF normalization -- a deliberate, important difference from the
EIDC/sph3 scripts elsewhere in challenge/ (e.g. make_contrast_curve_sph3.py,
run_vippaco_sph3.py): those pass PACO the RAW, un-normalized PSF by the
EIDC challenge's own convention (documented at length in that script's
own module docstring). example.py's real-system cycle instead runs the
PSF through vip_hci.fm.normalize_psf first (flux-normalized to unit
1-FWHM aperture flux, and refit to a 2D Gaussian model for its FWHM
in pixels) and passes THAT normalized PSF to PACO -- so this script does
the same, and does NOT reuse this project's own 1D radial-profile FWHM
estimator used in the EIDC scripts. --fwhm overrides only the resulting
FWHM value passed to both algorithms; normalize_psf still runs regardless
(its normalized PSF array is still what both algorithms receive).

pixscale -- also mirrors example.py's own active choice, not the
"standard" VLT_SPHERE_IRDIS['plsc'] (0.01225 arcsec/px): example.py
computes that standard constant but its active (uncommented) PACO
call passes a different, hardcoded pixscale=0.0145 instead, together with
fwhm expressed directly in pixels (no arcsec conversion). Default here is
that same 0.0145 for OUR OWN PACO call, for exact fidelity to the
working reference; override with --pixscale if you have a reason to use a
different value for a given dataset.

--pixscale is deliberately NOT reused as-is for VIP's FastPACO call. Our
own PACOBase.__init__ (src/paco_base.py) treats fwhm (pixels) and pixscale as
independent -- patch_width is derived from fwhm alone, pixscale is stored
only for other, unrelated uses. VIP's own PACO.__init__ instead computes
self.fwhm = int(fwhm/pixscale) and uses THAT single coupled value as its
patch-mask RADIUS -- so reusing example.py's (fwhm-in-pixels,
pixscale=0.0145) pair verbatim for VIP, as an earlier version of this
script did, gives VIP a patch radius of int(3.0/0.0145) = 206px against
our own ~4px for the same fwhm=3.0 -- catastrophically mismatched K
between the two implementations, and, on a real cube, either a crash, a
degenerate result, or an extremely long/hung run trying to extract
impossibly large patches (confirmed empirically on a small synthetic
test: VIP's FastPACO never returned). Instead, this script derives a
separate vip_pixscale so VIP's own coupled formula lands on the SAME
patch radius our own PACO actually uses (matching the exact
approach already established and verified in benchmarks/bench_vs_vippaco.py
and challenge/run_vippaco_sph3.py) -- pass --vip-pixscale to override
this derivation directly if you have a specific reason to.

Angle sign convention -- IMPORTANT, and specific to this dataset: by
convention, rotnth.fits's angles must be NEGATED before use for correct
detection with either algorithm (example.py itself always passes
angle_list=-derotation_angs, uncommented, in every one of its PACO calls).
Passing them as stored silently produces a detection map where real
companion signal is smeared/partially cancelled across frames instead of
reinforced -- there is no crash to flag this, so it is easy to miss. This
is applied by default (multiply by -1); pass --no-negate-angles to
disable if you are pointing this script at a different dataset where the
convention does not apply, and verify by checking whether negating raises
the peak S/N near a known point source rather than assuming it transfers.

Usage
-----
    python3 challenge/run_real_system_comparison.py /path/to/data_dir
    python3 challenge/run_real_system_comparison.py /path/to/data_dir --cpu 32
    python3 challenge/run_real_system_comparison.py /path/to/data_dir \\
        --cpu 32 --memory-backend cgroup --companion-xy 480,75
"""
import argparse
import csv
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import numpy as np
from scipy.ndimage import maximum_filter, label, find_objects
from vip_hci.fits import open_fits
from vip_hci.fm import normalize_psf

from src.full_paco import PACO
from _common import make_memory_monitor, mem_columns, mem_str

try:
    from vip_hci.invprob.paco import FastPACO as VIP_FastPACO
except ImportError as exc:
    raise SystemExit(
        "Could not import vip_hci.invprob.paco.FastPACO -- is vip_hci "
        f"installed in this environment? ({exc})")


def load_real_system(data_dir):
    """
    Load + preprocess exactly as example.py's process_system() does: load
    via vip_hci.fits.open_fits (float32 default precision, unlike this
    project's other, float64 EIDC-oriented loaders), and collapse a
    dual-band (ndim > 3) cube/psf pair by taking index 0 along the leading
    axis (not a median -- see the module docstring).
    """
    def fp(name):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected {path} -- see this script's module docstring "
                "for the required directory layout (center_im.fits, "
                "median_unsat.fits, rotnth.fits).")
        return path

    cube = open_fits(fp("center_im.fits"), verbose=False)
    angles_raw = np.asarray(open_fits(fp("rotnth.fits"), verbose=False)).ravel()
    psf = open_fits(fp("median_unsat.fits"), verbose=False)

    if cube.ndim > 3:
        cube = cube[0]
        psf = psf[0]

    return cube, angles_raw, psf


def find_peaks(snr_map, fwhm, threshold, top_n=10):
    """
    Local maxima above `threshold`, separated by ~fwhm, sorted by
    descending S/N -- the same local-maxima restriction used in
    challenge/plot_challenge_results.py:build_peak_map (needed so nearby
    correlated-noise pixels don't merge into one giant blob; see that
    function's docstring for why). Returns up to `top_n` (x, y, snr)
    tuples, matching this project's (x, y) = (column, row) convention.
    """
    data_max = maximum_filter(snr_map, size=max(1, int(round(fwhm))))
    maxima = (snr_map == data_max) & (snr_map > threshold) & np.isfinite(snr_map)
    labeled, _ = label(maxima)
    slices = find_objects(labeled)
    peaks = []
    for dy, dx in slices:
        x = (dx.start + dx.stop - 1) / 2
        y = (dy.start + dy.stop - 1) / 2
        peaks.append((x, y, float(snr_map[int(round(y)), int(round(x))])))
    peaks.sort(key=lambda p: -p[2])
    return peaks[:top_n]


def snr_at(snr_map, xy):
    x, y = xy
    iy, ix = int(round(y)), int(round(x))
    if 0 <= iy < snr_map.shape[0] and 0 <= ix < snr_map.shape[1]:
        return float(snr_map[iy, ix])
    return float("nan")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir",
                        help="Directory containing center_im.fits, "
                             "median_unsat.fits, rotnth.fits.")
    parser.add_argument("--cpu", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                        help="Worker processes for both algorithms "
                             "(default: cpu_count-1 on this machine).")
    parser.add_argument("--fwhm", type=float, default=None,
                        help="Override the FWHM (pixels) passed to both "
                             "algorithms. Default: the 2D-Gaussian-fit "
                             "value returned by vip_hci.fm.normalize_psf, "
                             "matching example.py's own convention (see "
                             "module docstring) -- normalize_psf still "
                             "runs regardless, since its normalized PSF "
                             "array is what both algorithms are given.")
    parser.add_argument("--psf-fit-size", type=int, default=19,
                        help="Cropped size passed to normalize_psf's own "
                             "'size' argument (default: 19, matching "
                             "example.py exactly).")
    parser.add_argument("--patch-radius-dec", type=int, default=-2,
                        help="Passed to our own PACO only (VIP's "
                             "FastPACO has no equivalent parameter -- see "
                             "src/paco_base.py's patch_radius_dec). "
                             "Default -2, matching example.py's active "
                             "PACO call.")
    parser.add_argument("--pixscale", type=float, default=0.0145,
                        help="Arcsec/pixel, passed to OUR OWN PACO's "
                             "constructor (and used for figure/CSV "
                             "labeling). Default 0.0145, matching "
                             "example.py's own active (uncommented) "
                             "PACO call -- NOT the 'standard' "
                             "VLT_SPHERE_IRDIS['plsc'] (0.01225), which "
                             "example.py computes but does not actually "
                             "use in that call. NOT reused directly for "
                             "VIP's FastPACO -- see --vip-pixscale and "
                             "the module docstring's K-matching note.")
    parser.add_argument("--vip-pixscale", type=float, default=None,
                        help="Pixscale passed to VIP's FastPACO "
                             "constructor specifically. Default: derived "
                             "so VIP's own coupled "
                             "self.fwhm=int(fwhm/pixscale) formula lands "
                             "on the SAME patch radius our own PACO "
                             "computes from --fwhm/--patch-radius-dec "
                             "(see the module docstring for why reusing "
                             "--pixscale directly for VIP is wrong and, "
                             "empirically, can hang). Override only if "
                             "you have a specific reason to give VIP a "
                             "different (possibly mismatched-K) patch "
                             "radius on purpose.")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="S/N threshold for peak-finding (default: "
                             "5.0, matching the EIDC challenge's own "
                             "convention used elsewhere in this project).")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top S/N peaks to report and plot "
                             "per algorithm.")
    parser.add_argument("--companion-xy", default=None,
                        help="Optional known companion position 'x,y' "
                             "(native pixel grid, this project's (col,row) "
                             "convention) to report each algorithm's S/N "
                             "at directly and mark on the detection maps. "
                             "Not required -- omit it to fall back to a "
                             "purely comparative top-N-peaks report, since "
                             "a real system has no EIDC-style ground truth.")
    parser.add_argument("--no-negate-angles", action="store_true",
                        help="Do NOT negate rotnth.fits's angles before "
                             "use (default: negate -- see this script's "
                             "module docstring for why this dataset needs "
                             "it, and verify before assuming it applies to "
                             "a different dataset).")
    parser.add_argument("--memory-backend", choices=["psutil", "cgroup"],
                        default="psutil",
                        help="How to measure peak memory -- see "
                             "benchmarks/_common.py's CgroupMemoryMonitor "
                             "docstring. 'cgroup' (Linux only) gives a "
                             "single real number from the kernel's own "
                             "accounting; falls back to 'psutil' (a "
                             "portable RSS-based estimate) with a printed "
                             "warning if this host doesn't support it.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress each algorithm's own progress "
                             "printing (default: verbose, matching "
                             "example.py's own convention -- useful for a "
                             "real, potentially long-running full-size "
                             "cube).")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write the comparison figure/CSV/"
                             "S/N maps (default: <data_dir>/paco_vs_pypaco/).")
    args = parser.parse_args()

    cube, angles_raw, psf_raw = load_real_system(args.data_dir)
    negate = not args.no_negate_angles
    angles = -angles_raw if negate else angles_raw

    # Matches example.py exactly: normalize_psf both normalizes the PSF's
    # flux (to unit aperture flux in 1 FWHM -- see the module docstring's
    # note on why this differs from the EIDC scripts' raw-PSF convention)
    # AND fits a 2D Gaussian model for its FWHM in pixels. `psf` (the
    # normalized array), not `psf_raw`, is what gets passed to both
    # algorithms below.
    psf, psf_flux, fwhm_fit = normalize_psf(
        psf_raw, debug=False, full_output=True, size=args.psf_fit_size,
        model="gauss", verbose=False)
    fwhm = args.fwhm if args.fwhm is not None else float(fwhm_fit)

    # Our own patch radius (see src/paco_base.py: patch_width =
    # int(2*fwhm) - patch_radius_dec, mask radius = patch_width // 2),
    # and the VIP-side pixscale that makes VIP's own coupled
    # self.fwhm=int(fwhm/pixscale) formula land on that SAME radius --
    # see the module docstring's K-matching note for why --pixscale
    # cannot simply be reused for VIP directly.
    our_patch_width = int(2 * fwhm) - args.patch_radius_dec
    our_mask_radius = max(1, our_patch_width // 2)
    vip_pixscale = (args.vip_pixscale if args.vip_pixscale is not None
                    else fwhm / our_mask_radius)

    out_dir = args.out_dir or os.path.join(args.data_dir, "paco_vs_pypaco")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Real system: {args.data_dir}")
    print(f"  cube={cube.shape}, psf_raw={psf_raw.shape} -> normalized "
          f"psf={psf.shape}, fwhm={fwhm:.3f}px (normalize_psf fit: "
          f"{float(fwhm_fit):.3f}px), cpu={args.cpu}")
    print(f"  ours: pixscale={args.pixscale}\"/px, patch_radius_dec="
          f"{args.patch_radius_dec} -> mask radius={our_mask_radius}px")
    print(f"  vip:  pixscale={vip_pixscale:.4f}\"/px (derived to match "
          f"our mask radius -- effective self.fwhm should print as "
          f"{our_mask_radius}px below)")
    print(f"  angles: {'NEGATED (default convention for this dataset)' if negate else 'as-is (--no-negate-angles)'}")

    companion_xy = None
    if args.companion_xy:
        cx_, cy_ = (float(v) for v in args.companion_xy.split(","))
        companion_xy = (cx_, cy_)

    results = {}

    # dit_psf/dit_science/nd_transmission/rescaling_factor=1.0 passed
    # explicitly to both constructors below, matching example.py's own
    # calls verbatim (functionally identical to each class's own defaults,
    # but kept explicit for fidelity to the reference and as the natural
    # place to change them if a given dataset's reduction needs it).

    # --- Ours: production ring-traversal PACO ---
    print("\n=== Ours: PACO (ring traversal) ===", flush=True)
    obj = PACO(cube, angles, psf, fwhm=fwhm, pixscale=args.pixscale,
                     dit_psf=1.0, dit_science=1.0, nd_transmission=1.0,
                     rescaling_factor=1.0,
                     patch_radius_dec=args.patch_radius_dec,
                     verbose=not args.quiet)
    with make_memory_monitor(args) as mon:
        t0 = time.perf_counter()
        snr_ours, flux_ours = obj.run(cpu=args.cpu, use_gpu=False, full_output=True)
        t_ours = time.perf_counter() - t0
    print(f"  time={t_ours:.1f}s  {mem_str(mon)}", flush=True)
    results["ours"] = {"label": "Ours: PACO (ring)", "snr": snr_ours,
                       "time_s": t_ours, "mon": mon}

    # --- VIP FastPACO ("pyPACO") ---
    # No patch_radius_dec equivalent exists on VIP's side (see
    # --patch-radius-dec's help) -- an unavoidable, documented asymmetry
    # between the two implementations, not an oversight.
    print("\n=== VIP FastPACO (\"pyPACO\") ===", flush=True)
    vip_obj = VIP_FastPACO(cube, angles, psf, fwhm=fwhm, pixscale=vip_pixscale,
                           dit_psf=1.0, dit_science=1.0, nd_transmission=1.0,
                           rescaling_factor=1.0, verbose=not args.quiet)
    with make_memory_monitor(args) as mon:
        t0 = time.perf_counter()
        snr_vip, flux_vip = vip_obj.run(cpu=args.cpu)
        t_vip = time.perf_counter() - t0
    print(f"  time={t_vip:.1f}s  {mem_str(mon)}  "
          f"(effective fwhm={vip_obj.fwhm}px, K={vip_obj.patch_area_pixels})",
          flush=True)
    results["vip"] = {"label": 'VIP FastPACO ("pyPACO")', "snr": snr_vip,
                      "time_s": t_vip, "mon": mon}

    # --- Detection capability ---
    print(f"\n=== Detection: top-{args.top_n} S/N peaks above "
          f"threshold={args.threshold} ===")
    for key, r in results.items():
        peaks = find_peaks(r["snr"], fwhm, args.threshold, top_n=args.top_n)
        r["peaks"] = peaks
        print(f"\n{r['label']}:")
        if not peaks:
            print("  (no peaks above threshold)")
        for (px, py, val) in peaks:
            print(f"  (x={px:7.2f}, y={py:7.2f})  S/N={val:6.2f}")
        if companion_xy is not None:
            val = snr_at(r["snr"], companion_xy)
            print(f"  S/N at supplied companion position "
                 f"({companion_xy[0]:.2f},{companion_xy[1]:.2f}) = {val:.3f}")

    # --- CSV summary ---
    csv_path = os.path.join(out_dir, "real_system_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["algorithm", "time_s", "mem_parent_mb", "mem_maxworker_mb",
                  "mem_combined_mb", "mem_real_cgroup_mb",
                  "top1_snr", "top1_x", "top1_y"]
        if companion_xy is not None:
            header.append("companion_xy_snr")
        w.writerow(header)
        for key, r in results.items():
            cols = mem_columns(r["mon"])
            top1 = r["peaks"][0] if r["peaks"] else None
            row = [r["label"], f"{r['time_s']:.2f}",
                  cols["mem_parent_mb"], cols["mem_maxworker_mb"],
                  cols["mem_combined_mb"], cols["mem_real_cgroup_mb"],
                  f"{top1[2]:.3f}" if top1 else "",
                  f"{top1[0]:.2f}" if top1 else "",
                  f"{top1[1]:.2f}" if top1 else ""]
            if companion_xy is not None:
                row.append(f"{snr_at(r['snr'], companion_xy):.3f}")
            w.writerow(row)
    print(f"\nWrote {csv_path}")

    # --- Figure: side-by-side S/N maps ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6))
    for ax, key in zip(axes, ["ours", "vip"]):
        r = results[key]
        ax.set_facecolor("black")
        im = ax.imshow(np.nan_to_num(r["snr"], nan=0.0), origin="lower",
                       cmap="inferno", vmin=0, vmax=args.threshold,
                       interpolation="nearest")
        for (px, py, val) in r["peaks"]:
            ax.add_patch(Circle((px, py), fwhm, edgecolor="gold",
                                facecolor="none", lw=1.5))
        if companion_xy is not None:
            ax.add_patch(Circle(companion_xy, fwhm * 1.5, edgecolor="cyan",
                                facecolor="none", lw=1.8, linestyle="--"))
        ax.set_title(f"{r['label']}\ntime={r['time_s']:.1f}s  {mem_str(r['mon'])}",
                    fontsize=8, color="black")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    fig.suptitle(f"Real-system comparison: {dataset_name}", fontsize=11)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "real_system_comparison.pdf")
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Wrote {fig_path}")

    np.save(os.path.join(out_dir, "snr_ours.npy"), snr_ours)
    np.save(os.path.join(out_dir, "snr_vip.npy"), snr_vip)
    print(f"Wrote {os.path.join(out_dir, 'snr_ours.npy')}")
    print(f"Wrote {os.path.join(out_dir, 'snr_vip.npy')}")


if __name__ == "__main__":
    main()
