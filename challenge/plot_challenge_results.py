#!/usr/bin/env python3
"""
Detection map and TPR/FDR/FPR curves of the EIDC sph3 dataset (Fig. 2 and
Table 6 of the paper), from the S/N maps of rerun_sph3_challenge.py.

Outputs (challenge/figures/):
  figure6_sph3_full.pdf, figure6_sph3_fast.pdf  map + curves per method
  figure6_sph3_metrics.csv                      F1 and AUCs
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.ndimage import maximum_filter

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(ROOT, "challenge", "subchallenge_1")
OUT_DIR  = os.path.join(ROOT, "challenge", "outputs")
SNR_DIR  = os.path.join(OUT_DIR, "snr_maps")
FIG_DIR  = os.path.join(ROOT, "challenge", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Datasets with known 0 injections (paper Table 3)

THRESHOLD = 5.0

SPH3_GROUND_TRUTH_XY = [
    (112.67, 90.22),
    (82.70, 58.70),
    (41.05, 86.18),
    (55.40, 95.61),
    (129.24, 82.90),
]

# Paper Table 4, sph3-only column (not the sph1+sph3 mean) for PACO.
PAPER_PACO_SPH3 = {"f1": 1.00, "auc_tpr": 0.93, "auc_fdr": 0.32}


def build_peak_map(snr, fwhm, floor=-100.0):
    """
    Reduce a raw (nonstationary, spatially correlated) S/N map to its local
    maxima only, everything else set to `floor`.

    `compute_binary_map` (photutils `detect_sources` under the hood) forms
    *connected components* of pixels above a threshold. On a raw, correlated
    S/N map, that fails badly close to threshold=0: nearby positive-noise
    pixels merge into one enormous blob (confirmed empirically: >1000 px for
    some sph3 companions, out of 25600 px total), which `max_blob_fact`
    then reclassifies entirely as false-positive area — swallowing whatever
    true companions happen to sit inside it and making TPR far below 1 at
    low thresholds. This does NOT match the official EIDC results: rerunning
    the *unmodified* `vip_hci.invprob.paco` module through the exact same
    `compute_binary_map` on this same sph3 dataset reproduces the same
    collapse (TPR=0.40 at threshold 0, vs ~1.0 in the published Fig. 3 for
    PACO/pyPACO) — so the discrepancy is in the scoring methodology, not in
    either PACO implementation. Restricting the map to local maxima first
    (so each source is an isolated ~1px blob regardless of the surrounding
    correlation) reproduces the published low-threshold behaviour almost
    exactly (AUC_TPR=0.922 vs the paper's 0.935 for unmodified VIP PACO).
    """
    data_max = maximum_filter(snr, size=max(1, int(round(fwhm))))
    return np.where(snr == data_max, snr, floor)


def compute_metrics_for_positions(snr, positions_xy, fwhm, threshold=THRESHOLD):
    """Compute F1/AUC_TPR/AUC_FDR/TPR/FDR/FPR curves for known (x,y) positions."""
    from vip_hci.metrics import compute_binary_map

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

# ──────────────────────────────────────────────────────────────────────────────
# Published paper results (Tables 3 & 4 of Cantalloube et al. 2021)
# Each entry: [F1_sph_mean, AUC_TPR_sph_mean, AUC_FDR_sph_mean,
#              F1_nrc_mean,  AUC_TPR_nrc_mean,  AUC_FDR_nrc_mean,
#              F1_lmr_mean,  AUC_TPR_lmr_mean,  AUC_FDR_lmr_mean]
# means over only the datasets with injections (u entries excluded).
# From Table 3 (speckle subtraction) and Table 4 (inverse problem).
# Columns = (sph_mean, nrc_mean, lmr_mean) for each metric (F1, AUC_TPR, AUC_FDR)
# ──────────────────────────────────────────────────────────────────────────────

# Paper Table 3 – speckle subtraction algorithms
# Format: {name: {"sph": [f1, auc_tpr, auc_fdr], "nrc": [...], "lmr": [...]}}

# Paper Table 4 – inverse problem algorithms

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_snr(inst, idx, variant):
    path = os.path.join(SNR_DIR, f"{inst}_{idx}_{variant}_snr.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def compute_fwhm_from_psf(inst, idx):
    psf_path = os.path.join(DATA_DIR, f"{inst}_psf_{idx}.fits")
    psf = fits.getdata(psf_path).astype(float)
    if psf.ndim == 3:
        psf = np.nanmedian(psf, axis=0)

    def _gauss1d(r, sigma):
        return np.exp(-r**2 / (2.0 * sigma**2))

    try:
        cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
        peak = psf[cy, cx]
        if peak <= 0:
            peak = psf.max()
        ny, nx = psf.shape
        yy, xx = np.ogrid[:ny, :nx]
        r_arr = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        r_max = int(min(cy, cx, ny - cy, nx - cx))
        r_vals = np.arange(r_max, dtype=float)
        profile = np.array([np.mean(psf[(r_arr >= rv - 0.5) & (r_arr < rv + 0.5)])
                            for rv in r_vals])
        pn = profile / max(float(profile[0]), 1e-10)
        n_fit = max(5, r_max // 2)
        popt, _ = curve_fit(_gauss1d, r_vals[:n_fit], pn[:n_fit], p0=[2.0],
                             maxfev=2000, bounds=([0.5], [r_max]))
        return max(2.355 * abs(float(popt[0])), 2.0)
    except Exception:
        return 5.0


def find_peaks(snr_map, fwhm, threshold):
    """Return (x, y) list of local maxima above threshold, separated by fwhm."""
    from scipy.ndimage import maximum_filter, label, find_objects
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


def circular_mask(ny, nx, r_inner, r_outer):
    yy, xx = np.ogrid[:ny, :nx]
    r = np.sqrt((yy - ny//2)**2 + (xx - nx//2)**2)
    return (r >= r_inner) & (r <= r_outer)


def match_peaks_to_injections(peaks, injections_xy, fwhm):
    """
    Classify each detected peak as TP (matched to an injection within one
    resolution element) or FP, using greedy nearest-neighbour matching so
    each injection is claimed by at most one peak (mirrors the TP/FP
    definition in Cantalloube et al. 2021, Sec. 3.1).

    Returns (tp_list, fp_list, missed_list) of (x, y) tuples, where
    missed_list holds the injected positions left unclaimed by any peak
    (i.e. companions the algorithm failed to detect above `threshold`).
    """
    injections_left = list(injections_xy)
    tp, fp = [], []
    # process peaks by descending distance-to-nearest-injection last, i.e.
    # give priority (claim first) to the closest matches overall.
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


def plot_detection_map_paper_style(ax, snr, fwhm, threshold, injections_xy,
                                    title, cmap="inferno", fig=None,
                                    marker_radius_extra=2.0):
    """
    Detection-map panel styled like Fig. 1 of Cantalloube et al. 2021:
    black background, color-bar clipped to [0, threshold], true positives
    as gold circles, false positives as red squares.

    `marker_radius_factor` scales the marker size relative to `fwhm/2` so
    the circle/square outline sits clear of the bright detection core
    instead of overlapping it (a factor of 1.0 draws the marker edge right
    through the source, as in the original, tighter version of this plot).
    """
    ny, nx = snr.shape
    ax.set_facecolor("black")
    im = ax.imshow(np.nan_to_num(snr, nan=0.0), origin="lower",
                   cmap=cmap, vmin=0, vmax=threshold,
                   interpolation="nearest")

    peaks = find_peaks(snr, fwhm, threshold)
    tp, fp, missed = match_peaks_to_injections(peaks, injections_xy, fwhm)
    marker_r = (fwhm / 2) + marker_radius_extra

    for (px, py) in tp:
        ax.add_patch(Circle((px, py), marker_r, edgecolor="gold",
                            facecolor="none", lw=1.8))
    for (px, py) in fp:
        ax.add_patch(plt.Rectangle((px - marker_r, py - marker_r),
                                   2 * marker_r, 2 * marker_r,
                                   edgecolor="red", facecolor="none", lw=1.8))
    # Missed detections: an injected companion with no peak claimed above
    # `threshold` within one resolution element. Drawn at the *injected*
    # position (there is no detected peak to center on) using a dashed
    # cyan circle so it reads as "present but not flagged" rather than a
    # false positive or a clean detection -- the point of this marker is
    # to make the near-miss visible in the image, not to imply the pixel
    # itself has any signature to look at.
    for (px, py) in missed:
        ax.add_patch(Circle((px, py), marker_r, edgecolor="cyan",
                            facecolor="none", lw=1.8, linestyle="--"))

    # NOTE: set_title() places text above the Axes bounding box, i.e. outside
    # the black imshow area, on the *figure's* background — which is white
    # for this layout. White title text there is invisible; use black.
    ax.set_title(title, fontsize=11, color="black", pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7, colors="black")
    return im, tp, fp, missed


def auc(y, x):
    """Normalised area under curve."""
    if len(x) < 2:
        return 0.0
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


# ──────────────────────────────────────────────────────────────────────────────
# Collect injection-test metrics saved by run script
# ──────────────────────────────────────────────────────────────────────────────


# Re-load full metric arrays from the injection SNR maps


# Known injection positions from run script (sep_arcsec, pa_deg, contrast)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 – Detailed sph3 comparison (5 injections in challenge data)
# One separate figure per algorithm (map on top, TPR/FDR/FPR curve below);
# numeric results are written to a companion CSV rather than onto the PDF.
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 6: sph3 detailed comparison …")

import csv

inst, idx = "sphere_irdis", 3
fwhm      = compute_fwhm_from_psf(inst, idx)
snr_full  = load_snr(inst, idx, "full")
snr_fast  = load_snr(inst, idx, "fast")

sph3_metrics = {}
for variant, snr in [("full", snr_full), ("fast", snr_fast)]:
    if snr is not None:
        sph3_metrics[variant] = compute_metrics_for_positions(
            snr, SPH3_GROUND_TRUTH_XY, fwhm)

ref_data = [
    ("Full PACO (this work)", "figure6_sph3_full.pdf", snr_full, "full"),
    ("Fast PACO (this work)", "figure6_sph3_fast.pdf", snr_fast, "fast"),
]

for lbl, fname, snr, variant in ref_data:
    fig7 = plt.figure(figsize=(6.5, 11))
    fig7.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.28, figure=fig7)
    ax_map = fig7.add_subplot(gs[0])
    ax_c   = fig7.add_subplot(gs[1])
    fig7.suptitle(f"VLT/SPHERE-IRDIS sph3 — {lbl}", fontsize=12)

    if snr is None:
        ax_map.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax_map.transAxes)
        fig7.savefig(os.path.join(FIG_DIR, fname), bbox_inches="tight", dpi=150)
        plt.close(fig7)
        print(f"  ✓ {fname} (N/A)")
        continue

    m = sph3_metrics[variant]

    # --- top: detection map, title reduced to just the F1 score ---
    plot_detection_map_paper_style(
        ax_map, snr, fwhm, THRESHOLD, SPH3_GROUND_TRUTH_XY,
        title=f"F1 score: {m['f1']:.2f}", marker_radius_extra=4)
    tp_proxy = Line2D([0], [0], marker='o', color='gold', linestyle='None',
                      markerfacecolor='none', markersize=9, markeredgewidth=1.8,
                      label='True positive')
    fp_proxy = Line2D([0], [0], marker='s', color='red', linestyle='None',
                      markerfacecolor='none', markersize=9, markeredgewidth=1.8,
                      label='False positive')
    missed_proxy = Line2D([0], [0], marker='o', color='cyan', linestyle='--',
                      markerfacecolor='none', markersize=9, markeredgewidth=1.8,
                      label='Missed detection')
    ax_map.legend(handles=[tp_proxy, fp_proxy, missed_proxy], loc="upper center",
                 bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8, frameon=False)

    # --- bottom: TPR/FDR/FPR vs threshold, matching Fig. 1's fill style:
    # green shades the region ABOVE the TPR curve (the "TPR gap" from 1.0),
    # red/salmon shades the region BELOW the FDR curve (from 0 up to FDR).
    ax_c.fill_between(m["thr"], m["tpr"], 1.0, color="green", alpha=0.25,
                      zorder=3)
    ax_c.fill_between(m["thr"], 0.0, m["fdr"], color="red", alpha=0.20,
                      zorder=2)
    line_tpr, = ax_c.plot(m["thr"], m["tpr"], color="green", lw=1.8, zorder=4)
    line_fdr, = ax_c.plot(m["thr"], m["fdr"], color="red", lw=1.6, zorder=3)
    line_fpr, = ax_c.plot(m["thr"], m["fpr"], color="steelblue", lw=1.2,
                          linestyle="-.", zorder=3)
    ax_c.axvline(THRESHOLD, color="black", linestyle=":", lw=1.2)
    ax_c.set_xlim(0, 2 * THRESHOLD)
    ax_c.set_ylim(0, 1.05)
    ax_c.set_xlabel("Threshold", fontsize=9)
    ax_c.set_ylabel("TPR / FDR / FPR", fontsize=9)
    ax_c.legend(
        [line_tpr, line_fdr, line_fpr],
        [f"AUC_TPR = {m['auc_tpr']:.3f}", f"AUC_FDR = {m['auc_fdr']:.3f}",
         f"AUC_FPR = {m['auc_fpr']:.3f}"],
        fontsize=8, loc="center right")
    ax_c.tick_params(labelsize=8)

    fig7.savefig(os.path.join(FIG_DIR, fname), bbox_inches="tight", dpi=150)
    plt.close(fig7)
    print(f"  ✓ {fname}")

# --- numeric results -> CSV (not on the PDF) ---
csv_path = os.path.join(FIG_DIR, "figure6_sph3_metrics.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "algorithm", "f1", "auc_tpr", "auc_fdr", "auc_fpr",
               "n_det", "n_fp", "n_injected"])
    w.writerow(["paper_table4", "PACO", PAPER_PACO_SPH3["f1"],
               PAPER_PACO_SPH3["auc_tpr"], PAPER_PACO_SPH3["auc_fdr"],
               "", "", "", 5])
    for variant, lbl2 in [("full", "Full PACO (this work)"), ("fast", "Fast PACO (this work)")]:
        if variant not in sph3_metrics:
            w.writerow(["ours", lbl2, "", "", "", "", "", "", 5])
            continue
        m = sph3_metrics[variant]
        w.writerow(["ours", lbl2, f"{m['f1']:.4f}", f"{m['auc_tpr']:.4f}",
                   f"{m['auc_fdr']:.4f}", f"{m['auc_fpr']:.4f}",
                   m["n_det"], m["n_fp"], 5])
print(f"  ✓ {os.path.basename(csv_path)}")

print(f"\nAll figures saved to {FIG_DIR}/")
