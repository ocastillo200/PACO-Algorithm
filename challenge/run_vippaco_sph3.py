#!/usr/bin/env python3
"""
Exploratory sanity check (Appendix, paco_paper.tex): try to reproduce the
EIDC challenge's own published "pyPACO" row (Table 4 of Cantalloube et al.
2021: sph3 F1=0.46, AUC_TPR=0.97, AUC_FDR=0.60) by running VIP's own
FastPACO implementation (vip_hci.invprob.paco.FastPACO -- the "pyPACO"
the challenge refers to) on the same sph3 dataset, then scoring its output
SNR map with the exact same evaluation code (compute_binary_map, ground
truth positions, threshold) used everywhere else in this project.

Why this matters: our own Full/Fast PACO get F1=0.889 on this dataset
(Sect. 5.1), clearly better than the EIDC-published pyPACO row (F1=0.46)
but short of the EIDC-published PACO reference row (F1=1.00). If running
VIP's own code ourselves reproduces something close to 0.46, that is
evidence our evaluation methodology (ground truth, matching tolerance,
threshold) is consistent with the challenge's own scoring, and the
remaining gap to the reference (1.00) is a property of the algorithms
being compared, not of how we score them. If it does NOT reproduce 0.46,
that is itself informative: it means the "pyPACO" number in Table 4 is
not straightforwardly reproducible from VIP's current public API with
default/naive settings, and the residual difference could come from
several unverifiable sources: the exact VIP version/commit used for the
original submission (VIP has changed over time, see Sect. "Numerical
fidelity"'s finding of a shrinkage-weight bug in the CURRENT public
version, which may or may not have been present in whatever version EIDC
used), undocumented parameter choices (fwhm/pixscale unit convention --
see below -- oversampling, patch radius), or whether the EIDC evaluation
even used the public vip_hci package at all rather than a private/modified
copy. This script does not resolve that ambiguity; it only checks whether
the most natural, out-of-the-box use of the CURRENT public VIP code gets
close.

Unit-convention ambiguity in VIP's own API (a concrete source of the
uncertainty above): PACO.__init__ computes
    self.fwhm = int(fwhm_param / pixscale_param)
and then uses self.fwhm as BOTH the effective FWHM in pixels AND the
patch-mask radius everywhere downstream (vip_hci/invprob/paco.py). This
is only self-consistent if fwhm_param and pixscale_param are in the same
angular unit. VIP's own broader convention (e.g. fm.normalize_psf, whose
`fwhm='fit'` option fits a 2D model to the PSF and returns the FWHM in
PIXELS) is "fwhm in pixels" everywhere else in the library, while
vip_hci.config.VLT_SPHERE_IRDIS['plsc'] = 0.01225 provides IRDIS's real
plate scale in arcsec/px as a standard, citable instrument constant (not
something a user would leave at PACO's own pixscale=1.0 default if they
know to look for it). Two readings of "use VIP as intended" therefore
both seem defensible:
  (A) fwhm = the PSF's fitted FWHM in pixels (via normalize_psf),
      pixscale left at the constructor's own default of 1.0 -- i.e.
      nothing about this dataset's real plate scale is used;
  (B) fwhm = that same fitted pixel FWHM converted to arcsec using
      IRDIS's real plate scale, pixscale = that same real plate scale --
      i.e. genuinely physical units throughout, consistent with a user
      who deliberately looked up VLT_SPHERE_IRDIS['plsc'] rather than
      leaving pixscale at its meaningless default.
We do not know which convention (if either) the original EIDC pyPACO
submission used, so both are run here and reported separately rather
than silently picking one.

Usage
-----
    python3 challenge/run_vippaco_sph3.py --cpu 4
"""
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from scipy.ndimage import maximum_filter, label, find_objects
from vip_hci.metrics import compute_binary_map

from challenge.make_contrast_curve_sph3 import load_sph3, compute_fwhm

try:
    from vip_hci.invprob.paco import FastPACO as VIP_FastPACO
    from vip_hci.fm import normalize_psf as vip_normalize_psf
    from vip_hci.config import VLT_SPHERE_IRDIS
except ImportError as exc:
    raise SystemExit(
        "Could not import vip_hci.invprob.paco.FastPACO -- is vip_hci "
        f"installed in this environment? ({exc})")

FIG_DIR = os.path.join(ROOT, "challenge", "figures")
PAPER_FIG_DIR = os.path.join(ROOT, "Astronomy_and_Astrophysics_Template", "figures")
SNR_DIR = os.path.join(ROOT, "challenge", "outputs", "snr_maps")

THRESHOLD = 5.0

# Same 5 positions used everywhere else in this project (see
# challenge/plot_challenge_results.py:SPH3_GROUND_TRUTH_XY for full
# provenance/derivation notes) -- kept in sync manually since importing
# plot_challenge_results.py directly would re-run its entire figure
# pipeline (no __main__ guard).
SPH3_GROUND_TRUTH_XY = [
    (112.67, 90.22),
    (82.70, 58.70),
    (41.05, 86.18),
    (55.40, 95.61),
    (129.24, 82.90),
]

# Published EIDC reference rows -- Table 4 of Cantalloube et al. 2021,
# sph3-ONLY column (labeled "(3)" in that table), NOT the sph1+sph3 mean
# column ("Mean") also present there. Read directly off the table image
# (challenge/2101.05080v1-2.pdf, page 13): row "pyPACO", VLT/SPHERE-IRDIS
# block, column (3) = 0.83 (F1, black) / 0.94 (AUC_TPR, green) / 0.50
# (AUC_FDR, red) -- the mean column (0.46/0.97/0.60) was used here in an
# earlier version of this script by mistake (confirmed against the same
# table image that the PACO row's values below already used column (3)
# correctly, matching plot_challenge_results.py's own
# PAPER_PACO_SPH3 comment).
PAPER_PACO_SPH3 = {"f1": 1.00, "auc_tpr": 0.93, "auc_fdr": 0.32}
PAPER_PYPACO_SPH3 = {"f1": 0.83, "auc_tpr": 0.94, "auc_fdr": 0.50}


def build_peak_map(snr, fwhm, floor=-100.0):
    # Kept in sync with plot_challenge_results.py's build_peak_map -- see
    # that function's docstring for why max(1, round(fwhm)) (not int(fwhm))
    # and local-maxima restriction are needed to match the EIDC's own
    # low-threshold scoring behavior.
    data_max = maximum_filter(snr, size=max(1, int(round(fwhm))))
    return np.where(snr == data_max, snr, floor)


def circular_mask(ny, nx, r_inner, r_outer):
    yy, xx = np.ogrid[:ny, :nx]
    r = np.sqrt((yy - ny // 2) ** 2 + (xx - nx // 2) ** 2)
    return (r >= r_inner) & (r <= r_outer)


def auc(y, x):
    if len(x) < 2:
        return 0.0
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def find_peaks(snr_map, fwhm, threshold):
    data_max = maximum_filter(snr_map, size=int(fwhm))
    maxima = (snr_map == data_max) & (snr_map > threshold)
    labeled, _ = label(maxima)
    slices = find_objects(labeled)
    peaks = []
    for dy, dx in slices:
        x = (dx.start + dx.stop - 1) / 2
        y = (dy.start + dy.stop - 1) / 2
        peaks.append((x, y))
    return peaks


def match_peaks_to_injections(peaks, injections_xy, fwhm):
    injections_left = list(injections_xy)
    tp, fp = [], []
    candidates = []
    for (px, py) in peaks:
        if injections_left:
            dists = [np.hypot(px - ix, py - iy) for (ix, iy) in injections_left]
            j = int(np.argmin(dists))
            candidates.append((dists[j], (px, py), j))
        else:
            candidates.append((np.inf, (px, py), None))
    candidates.sort(key=lambda c: c[0])
    claimed = set()
    for dist, (px, py), j in candidates:
        if j is not None and j not in claimed and dist <= fwhm:
            tp.append((px, py))
            claimed.add(j)
        else:
            fp.append((px, py))
    missed = [inj for j, inj in enumerate(injections_left) if j not in claimed]
    return tp, fp, missed


def compute_metrics_for_positions(snr, positions_xy, fwhm, threshold=THRESHOLD):
    ny, nx = snr.shape
    peak_map = build_peak_map(snr, fwhm)
    thr_grid = np.linspace(0, 2 * threshold, 80)
    n_inj = len(positions_xy)
    r_inner = 2 * fwhm
    r_outer = min(ny, nx) / 2 - 1
    mask = circular_mask(ny, nx, r_inner, r_outer)
    n_res = max(1, int(mask.sum() / (np.pi * (fwhm / 2) ** 2)))

    tpr_arr, fdr_arr, fpr_arr = [], [], []
    for thr in thr_grid:
        try:
            res = compute_binary_map(
                peak_map, thresholds=[thr], injections=tuple(positions_xy),
                fwhm=fwhm, npix=1, overlap_threshold=0.7,
                max_blob_fact=2, debug=False, plot=False)
            n_det, n_fp, _ = res
            n_det, n_fp = n_det[0], n_fp[0]
        except Exception:
            n_det, n_fp = 0, 0
        tpr_arr.append(n_det / n_inj if n_inj > 0 else 0.0)
        fdr_arr.append(n_fp / (n_fp + n_det) if (n_fp + n_det) > 0 else 0.0)
        fpr_arr.append(n_fp / (n_fp + n_res) if (n_fp + n_res) > 0 else 0.0)
    tpr_arr = np.array(tpr_arr)
    fdr_arr = np.array(fdr_arr)
    fpr_arr = np.array(fpr_arr)

    try:
        rt = compute_binary_map(
            peak_map, thresholds=[threshold], injections=tuple(positions_xy),
            fwhm=fwhm, npix=1, overlap_threshold=0.7,
            max_blob_fact=2, debug=False, plot=False)
        nd, nfp, _ = rt
        nd, nfp = nd[0], nfp[0]
    except Exception:
        nd, nfp = 0, 0
    prec = nd / (nd + nfp) if (nd + nfp) > 0 else 0.0
    rec = nd / n_inj if n_inj > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    auc_tpr = auc(tpr_arr, thr_grid)
    auc_fdr = auc(fdr_arr, thr_grid)
    auc_fpr = auc(fpr_arr, thr_grid)

    return {"thr": thr_grid, "tpr": tpr_arr, "fdr": fdr_arr, "fpr": fpr_arr,
            "f1": f1, "auc_tpr": auc_tpr, "auc_fdr": auc_fdr, "auc_fpr": auc_fpr,
            "n_det": nd, "n_fp": nfp}


def run_vip_fastpaco(cube, pa, psf, fwhm_param, pixscale_param, cpu, label_):
    print(f"\n=== VIP FastPACO run: {label_} "
          f"(fwhm_param={fwhm_param:.5f}, pixscale_param={pixscale_param:.5f}) ===")
    vip_obj = VIP_FastPACO(cube, pa, psf, fwhm=fwhm_param,
                           pixscale=pixscale_param, verbose=False)
    print(f"  -> effective self.fwhm (== patch mask radius, px) = {vip_obj.fwhm}, "
          f"patch_area_pixels={vip_obj.patch_area_pixels}")
    t0 = time.perf_counter()
    snr, flux = vip_obj.run(cpu=cpu)
    dt = time.perf_counter() - t0
    print(f"  wall time: {dt:.1f}s (cpu={cpu})")
    nan_frac = float(np.mean(np.isnan(snr)))
    print(f"  snr map: nan_frac={nan_frac:.4f}, "
          f"finite range=[{np.nanmin(snr):.3g}, {np.nanmax(snr):.3g}]")
    return snr, vip_obj.fwhm


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--cached", action="store_true",
                        help="Reuse the S/N maps already saved under "
                             "challenge/outputs/snr_maps/ from a previous "
                             "run instead of re-running VIP FastPACO "
                             "(which takes ~7 min per convention).")
    args = parser.parse_args()

    cube, pa, psf, pxscale_fits = load_sph3()
    our_fwhm_px = compute_fwhm(psf)

    # VIP's own FWHM estimator (2D Gaussian fit via fm.normalize_psf,
    # fwhm='fit'), rather than our own 1D radial-profile fit, so this
    # experiment uses VIP's own idiom end to end, not a mix of ours and
    # theirs. full_output=True additionally returns the fitted FWHM (mean
    # of the X/Y Gaussian fit for model='gauss').
    _, _, vip_fwhm_fit = vip_normalize_psf(psf, fwhm='fit', full_output=True,
                                           verbose=False)
    vip_fwhm_px = float(np.mean(vip_fwhm_fit))

    # IRDIS's real plate scale, from VIP's own standard instrument-constant
    # dict (vip_hci.config.VLT_SPHERE_IRDIS) rather than the FITS header's
    # own pxscale extension, so the "real pixscale" used below is traceable
    # to something VIP itself publishes as canonical for this instrument.
    pxscale_std = VLT_SPHERE_IRDIS["plsc"]

    print(f"sph3: cube={cube.shape}")
    print(f"  our compute_fwhm(psf)              = {our_fwhm_px:.3f} px")
    print(f"  VIP normalize_psf(fwhm='fit')       = {vip_fwhm_px:.3f} px")
    print(f"  pxscale (FITS header, this dataset) = {pxscale_fits:.5f} \"/px")
    print(f"  pxscale (VLT_SPHERE_IRDIS['plsc'])  = {pxscale_std:.5f} \"/px")

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(PAPER_FIG_DIR, exist_ok=True)
    os.makedirs(SNR_DIR, exist_ok=True)

    results = {}
    path_a = os.path.join(SNR_DIR, "sphere_irdis_3_vippaco_convA_snr.npy")
    path_b = os.path.join(SNR_DIR, "sphere_irdis_3_vippaco_convB_snr.npy")

    if args.cached and os.path.exists(path_a) and os.path.exists(path_b):
        print(f"\n--cached: loading saved S/N maps from {SNR_DIR} "
              "instead of re-running VIP FastPACO")
        results["A"] = (np.load(path_a), 3)
        results["B"] = (np.load(path_b), 3)
    else:
        # Convention A: fwhm = VIP's own fitted FWHM in pixels, pixscale
        # left at the constructor's own default of 1.0 -- consistent with
        # "fwhm is always in pixels" as used throughout the rest of
        # vip_hci, but ignores this dataset's real plate scale entirely.
        snr_a, eff_fwhm_a = run_vip_fastpaco(
            cube, pa, psf, fwhm_param=vip_fwhm_px, pixscale_param=1.0,
            cpu=args.cpu, label_="Convention A (fwhm in px, pixscale=1.0)")
        np.save(path_a, snr_a)
        results["A"] = (snr_a, eff_fwhm_a)

        # Convention B: that same VIP-fitted pixel FWHM converted to arcsec
        # using IRDIS's real, standard plate scale, together with that
        # same real plate scale -- genuine physical units throughout,
        # consistent with a user who deliberately looked up
        # VLT_SPHERE_IRDIS['plsc'] rather than leaving pixscale at its
        # meaningless default.
        fwhm_arcsec = vip_fwhm_px * pxscale_std
        snr_b, eff_fwhm_b = run_vip_fastpaco(
            cube, pa, psf, fwhm_param=fwhm_arcsec, pixscale_param=pxscale_std,
            cpu=args.cpu, label_="Convention B (fwhm in arcsec, real pixscale)")
        np.save(path_b, snr_b)
        results["B"] = (snr_b, eff_fwhm_b)

    print("\n" + "=" * 78)
    print("Comparison against EIDC-published rows and our own implementation")
    print("=" * 78)
    rows = []
    rows.append(("PACO (EIDC ref., published)", PAPER_PACO_SPH3["f1"],
                PAPER_PACO_SPH3["auc_tpr"], PAPER_PACO_SPH3["auc_fdr"], "5/5 (reported)"))
    rows.append(("pyPACO (EIDC, published)", PAPER_PYPACO_SPH3["f1"],
                PAPER_PYPACO_SPH3["auc_tpr"], PAPER_PYPACO_SPH3["auc_fdr"], "(reported)"))
    rows.append(("Full PACO (this work)", 0.8889, 0.8380, 0.3392, "4/5"))
    rows.append(("Fast PACO (this work)", 0.8889, 0.8304, 0.3250, "4/5"))

    for label_, (snr, eff_fwhm) in results.items():
        # Score with OUR fwhm (our_fwhm_px) for the matching/threshold
        # tolerance, matching how every other row in this project's
        # tables is scored -- not VIP's own (possibly different,
        # integer-truncated) effective fwhm, so the comparison isolates
        # the algorithm/map, not an extra scoring-tolerance difference.
        m = compute_metrics_for_positions(snr, SPH3_GROUND_TRUTH_XY, our_fwhm_px)
        rows.append((f"VIP FastPACO, convention {label_} (this run)",
                    m["f1"], m["auc_tpr"], m["auc_fdr"],
                    f"{m['n_det']}/5, {m['n_fp']} FP"))
        print(f"Convention {label_}: F1={m['f1']:.4f} AUC_TPR={m['auc_tpr']:.4f} "
              f"AUC_FDR={m['auc_fdr']:.4f} n_det={m['n_det']} n_fp={m['n_fp']}")

    print(f"\n{'Row':<38}{'F1':>7}{'AUC_TPR':>9}{'AUC_FDR':>9}  Detections")
    for name, f1, at, af, det in rows:
        print(f"{name:<38}{f1:7.3f}{at:9.3f}{af:9.3f}  {det}")

    csv_path = os.path.join(FIG_DIR, "vippaco_sph3_comparison.csv")
    import csv as csv_mod
    with open(csv_path, "w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow(["source", "f1", "auc_tpr", "auc_fdr", "detections"])
        for name, f1, at, af, det in rows:
            w.writerow([name, f"{f1:.4f}", f"{at:.4f}", f"{af:.4f}", det])
    print(f"\nWrote {csv_path}")

    # ---- Figure: detection maps (top) + TPR/FDR/FPR curves (bottom) for
    # both conventions, styled like Fig. 6 -- the curve row is what lets a
    # reader see how differently VIP FastPACO's threshold behavior looks
    # from the challenge's own published curve shape (high TPR bought with
    # a much larger FDR at every threshold), not just the single-threshold
    # F1 snapshot in Table~\ref{tab:vippaco-repro}.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.0))
    for col, label_ in enumerate(["A", "B"]):
        snr, eff_fwhm = results[label_]
        ax = axes[0, col]
        ax.set_facecolor("black")
        im = ax.imshow(np.nan_to_num(snr, nan=0.0), origin="lower",
                       cmap="inferno", vmin=0, vmax=THRESHOLD,
                       interpolation="nearest")
        peaks = find_peaks(snr, our_fwhm_px, THRESHOLD)
        tp, fp, missed = match_peaks_to_injections(peaks, SPH3_GROUND_TRUTH_XY, our_fwhm_px)
        marker_r = (our_fwhm_px / 2) + 4
        for (px, py) in tp:
            ax.add_patch(Circle((px, py), marker_r, edgecolor="gold",
                                facecolor="none", lw=1.8))
        for (px, py) in fp:
            ax.add_patch(plt.Rectangle((px - marker_r, py - marker_r),
                                       2 * marker_r, 2 * marker_r,
                                       edgecolor="red", facecolor="none", lw=1.8))
        for (px, py) in missed:
            ax.add_patch(Circle((px, py), marker_r, edgecolor="cyan",
                                facecolor="none", lw=1.8, linestyle="--"))
        m = compute_metrics_for_positions(snr, SPH3_GROUND_TRUTH_XY, our_fwhm_px)
        ax.set_title(f"VIP FastPACO, convention {label_}\n"
                     f"(effective fwhm={eff_fwhm}px) -- F1={m['f1']:.2f}",
                     fontsize=10, color="black", pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # TPR/FDR/FPR vs threshold, matching Fig. 6's bottom-panel style.
        axc = axes[1, col]
        axc.fill_between(m["thr"], m["tpr"], 1.0, color="green", alpha=0.25, zorder=3)
        axc.fill_between(m["thr"], 0.0, m["fdr"], color="red", alpha=0.20, zorder=2)
        axc.plot(m["thr"], m["tpr"], color="green", lw=1.8, zorder=4)
        axc.plot(m["thr"], m["fdr"], color="red", lw=1.6, zorder=3)
        axc.plot(m["thr"], m["fpr"], color="steelblue", lw=1.2, linestyle="-.", zorder=3)
        axc.axvline(THRESHOLD, color="black", linestyle=":", lw=1.2)
        axc.set_xlim(0, 2 * THRESHOLD)
        axc.set_ylim(0, 1.05)
        axc.set_xlabel("Threshold", fontsize=9)
        axc.set_ylabel("TPR / FDR / FPR", fontsize=9)
        axc.set_title(f"AUC$_{{TPR}}$={m['auc_tpr']:.3f}  "
                     f"AUC$_{{FDR}}$={m['auc_fdr']:.3f}", fontsize=9)
        axc.tick_params(labelsize=8)

    tp_proxy = Line2D([0], [0], marker='o', color='gold', linestyle='None',
                      markerfacecolor='none', markersize=9, markeredgewidth=1.8,
                      label='True positive')
    fp_proxy = Line2D([0], [0], marker='s', color='red', linestyle='None',
                      markerfacecolor='none', markersize=9, markeredgewidth=1.8,
                      label='False positive')
    missed_proxy = Line2D([0], [0], marker='o', color='cyan', linestyle='--',
                          markerfacecolor='none', markersize=9, markeredgewidth=1.8,
                          label='Missed detection')
    tpr_proxy = Line2D([0], [0], color='green', lw=1.8, label='TPR')
    fdr_proxy = Line2D([0], [0], color='red', lw=1.6, label='FDR')
    fpr_proxy = Line2D([0], [0], color='steelblue', lw=1.2, linestyle='-.', label='FPR')
    fig.legend(handles=[tp_proxy, fp_proxy, missed_proxy, tpr_proxy, fdr_proxy, fpr_proxy],
              loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=6, fontsize=8,
              frameon=False)
    fig.suptitle("VIP FastPACO (\"pyPACO\") on sph3 -- exploratory reproduction "
                "attempt of the EIDC-published baseline", fontsize=11, y=1.0)
    fig.tight_layout()
    for d in (FIG_DIR, PAPER_FIG_DIR):
        out_pdf = os.path.join(d, "fig_vippaco_sph3.pdf")
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
