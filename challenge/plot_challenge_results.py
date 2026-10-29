#!/usr/bin/env python3
"""
Generate all comparison PDF figures for the PACO challenge results.

Figures produced:
  figure1_detection_maps_all.pdf   – 9 datasets × 2 algorithms gallery
  figure2_sphere_irdis_detail.pdf  – per-dataset detection + curve (SPHERE)
  figure3_nirc2_detail.pdf         – per-dataset detection + curve (NIRC2)
  figure4_lmircam_detail.pdf       – per-dataset detection maps (LMIRCam)
  figure5_metric_curves.pdf        – TPR/FDR/FPR curves from injection test
  figure6_ranking_comparison.pdf   – ranking bar chart vs paper algorithms
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
from matplotlib.colors import Normalize
from matplotlib import cm
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
ZERO_INJECTION = {"sphere_irdis_2", "nirc2_2"}

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
SPECKLE_RESULTS = {
    "Baseline":      {"sph": [0.88, 0.53, 0.32], "nrc": [0.00, 0.29, 0.41], "lmr": [0.39, 0.38, 0.37]},
    "LOCI":          {"sph": [0.83, 0.83, 0.39], "nrc": [0.68, 0.39, 0.40], "lmr": [0.72, 0.60, 0.39]},
    r"TLOCI$_{SpeCal}$": {"sph": [0.38, 0.30, 0.23], "nrc": [0.15, 0.08, 0.38], "lmr": [0.08, 0.22, 0.49]},
    r"PCA$_{MPIA}$":  {"sph": [0.75, 0.67, 0.30], "nrc": [0.33, 0.31, 0.34], "lmr": [0.66, 0.55, 0.36]},
    r"PCA$_{Padova}$":{"sph": [0.67, 0.75, 0.46], "nrc": [0.00, 0.17, 0.25], "lmr": [0.37, 0.58, 0.57]},
    r"STIM$_{full}$": {"sph": [1.00, 0.88, 0.38], "nrc": [0.25, 0.19, 0.39], "lmr": [0.39, 0.44, 0.42]},
    r"STIM$_{ann}$":  {"sph": [0.83, 0.81, 0.43], "nrc": [0.00, 0.21, 0.45], "lmr": [0.54, 0.68, 0.40]},
    "RSM":           {"sph": [1.00, 0.88, 0.06], "nrc": [0.68, 0.38, 0.04], "lmr": [0.89, 0.60, 0.17]},
    "SLIMask":       {"sph": [1.00, 0.89, 0.07], "nrc": [0.58, 0.33, 0.40], "lmr": [0.82, 0.67, 0.31]},
}

# Paper Table 4 – inverse problem algorithms
INVERSE_RESULTS = {
    "ANDROMEDA":     {"sph": [1.00, 0.68, 0.36], "nrc": [0.25, 0.35, 0.38], "lmr": [0.33, 0.34, 0.38]},
    "FMMF":          {"sph": [0.83, 0.93, 0.38], "nrc": [0.58, 0.39, 0.39], "lmr": [0.75, 0.77, 0.40]},
    "PACO":          {"sph": [1.00, 0.97, 0.36], "nrc": [0.50, 0.39, 0.45], "lmr": [0.43, 0.46, 0.57]},
    "pyPACO":        {"sph": [0.46, 0.97, 0.60], "nrc": [0.00, 0.16, 0.55], "lmr": [0.09, 0.93, 0.78]},
    "TRAP":          {"sph": [1.00, 0.80, 0.36], "nrc": [0.25, 0.33, 0.33], "lmr": [0.76, 0.60, 0.37]},
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_snr(inst, idx, variant):
    path = os.path.join(SNR_DIR, f"{inst}_{idx}_{variant}_snr.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def load_inject_snr(inst, idx, variant):
    path = os.path.join(SNR_DIR, f"{inst}_{idx}_{variant}_inject_snr.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def load_inject_metrics(inst, idx, variant):
    """Re-compute metrics from saved snr map + known injection positions."""
    path = os.path.join(SNR_DIR, f"inject_metrics_{inst}_{idx}_{variant}.npy")
    if os.path.exists(path):
        return np.load(path, allow_pickle=True).item()
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


def plot_snr_map(ax, snr, fwhm, threshold, title, cmap="magma"):
    """Display SNR map and mark peaks above threshold."""
    ny, nx = snr.shape
    vmin, vmax = -3, min(float(np.nanpercentile(snr, 99.5)), 15)
    im = ax.imshow(np.nan_to_num(snr, nan=0.0), origin="lower",
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    # circular field mask
    mask = circular_mask(ny, nx, 2 * fwhm, min(ny, nx) / 2 - 1)
    overlay = np.zeros((ny, nx, 4))
    overlay[~mask, 3] = 0.45
    ax.imshow(overlay, origin="lower")
    # detected peaks
    peaks = find_peaks(snr, fwhm, threshold)
    for (px, py) in peaks:
        c = Circle((px, py), fwhm / 2, color="gold",
                   fill=False, lw=1.5, linestyle="-")
        ax.add_patch(c)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    return im, peaks


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


def plot_metric_curves(ax, thr, tpr, fdr, fpr, threshold,
                        auc_tpr, auc_fdr, f1, title=""):
    ax.fill_between(thr, tpr, alpha=0.3, color="green")
    ax.fill_between(thr, fdr, alpha=0.3, color="red")
    ax.plot(thr, tpr, color="green",  lw=1.5,
            label=f"AUC$_{{TPR}}$={auc_tpr:.3f}")
    ax.plot(thr, fdr, color="red",    lw=1.5,
            label=f"AUC$_{{FDR}}$={auc_fdr:.3f}")
    ax.plot(thr, fpr, color="steelblue", lw=1.2, linestyle="--",
            label=f"FPR")
    ax.axvline(threshold, color="gray", linestyle=":", lw=1)
    ax.set_xlim(0, thr[-1])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Threshold", fontsize=8)
    ax.set_ylabel("TPR / FDR / FPR", fontsize=8)
    ax.set_title(f"F1={f1:.3f}", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=7)


def auc(y, x):
    """Normalised area under curve."""
    if len(x) < 2:
        return 0.0
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


# ──────────────────────────────────────────────────────────────────────────────
# Collect injection-test metrics saved by run script
# ──────────────────────────────────────────────────────────────────────────────

def load_injection_metrics():
    """Load pre-computed injection metrics from disk."""
    meta_path = os.path.join(OUT_DIR, "results_meta.npy")
    if not os.path.exists(meta_path):
        return {}
    data = np.load(meta_path, allow_pickle=True).item()
    return data.get("inject_metrics", {})


inj_meta = load_injection_metrics()

# Re-load full metric arrays from the injection SNR maps
def rebuild_inject_metrics(inst, idx, variant, positions_xy, fwhm):
    """Recompute metric sweep from saved injection SNR map."""
    from vip_hci.metrics import compute_binary_map

    snr = load_inject_snr(inst, idx, variant)
    if snr is None:
        return None
    ny, nx = snr.shape
    peak_map = build_peak_map(snr, fwhm)
    thr_grid = np.linspace(0, 2 * THRESHOLD, 80)
    n_inj = len(positions_xy)
    r_inner = 2 * fwhm
    r_outer = min(ny, nx) / 2 - 1
    mask = circular_mask(ny, nx, r_inner, r_outer)
    n_res = max(1, int(mask.sum() / (np.pi * (fwhm / 2)**2)))

    tpr_arr, fdr_arr, fpr_arr = [], [], []
    for thr in thr_grid:
        try:
            res = compute_binary_map(
                peak_map, thresholds=[thr],
                injections=tuple((p[0], p[1]) for p in positions_xy),
                fwhm=fwhm, npix=1, overlap_threshold=0.7,
                max_blob_fact=2, debug=False, plot=False)
            n_det, n_fp, _ = res
            n_det, n_fp = n_det[0], n_fp[0]
        except Exception:
            n_det, n_fp = 0, 0
        tpr = n_det / n_inj if n_inj > 0 else 0.0
        fdr = n_fp / (n_fp + n_det) if (n_fp + n_det) > 0 else 0.0
        fpr = n_fp / (n_fp + n_res) if (n_fp + n_res) > 0 else 0.0
        tpr_arr.append(tpr); fdr_arr.append(fdr); fpr_arr.append(fpr)

    tpr_arr = np.array(tpr_arr)
    fdr_arr = np.array(fdr_arr)
    fpr_arr = np.array(fpr_arr)

    # at threshold
    try:
        rt = compute_binary_map(
            peak_map, thresholds=[THRESHOLD],
            injections=tuple((p[0], p[1]) for p in positions_xy),
            fwhm=fwhm, npix=1, overlap_threshold=0.7,
            max_blob_fact=2, debug=False, plot=False)
        nd, nfp, _ = rt; nd, nfp = nd[0], nfp[0]
    except Exception:
        nd, nfp = 0, 0

    prec = nd / (nd + nfp) if (nd + nfp) > 0 else 0.0
    rec  = nd / n_inj if n_inj > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    auc_tpr = auc(tpr_arr, thr_grid)
    auc_fdr = auc(fdr_arr, thr_grid)

    return {"thr": thr_grid, "tpr": tpr_arr, "fdr": fdr_arr, "fpr": fpr_arr,
            "f1": f1, "auc_tpr": auc_tpr, "auc_fdr": auc_fdr, "n_det": nd, "n_fp": nfp}


# Known injection positions from run script (sep_arcsec, pa_deg, contrast)
INJECTIONS_CONFIG = [(0.35, 45, 2e-5), (0.55, 155, 2e-5), (0.75, 270, 2e-5)]

def get_injection_positions(inst, idx):
    """Compute injection x,y positions for a given dataset."""
    pxs_file = os.path.join(DATA_DIR, f"{inst}_pxscale_{idx}.fits")
    pa_file  = os.path.join(DATA_DIR, f"{inst}_pa_{idx}.fits")
    pxs = float(fits.getdata(pxs_file).flat[0])
    if np.isnan(pxs) or pxs <= 0:
        pxs = {"nirc2": 0.020}.get(inst, 0.020)
    pa  = fits.getdata(pa_file).ravel()

    cube = fits.getdata(os.path.join(DATA_DIR, f"{inst}_cube_{idx}.fits"))
    ny, nx = cube.shape[1], cube.shape[2]
    cy, cx = ny // 2, nx // 2

    positions = []
    for sep_as, pa_deg, _ in INJECTIONS_CONFIG:
        sep_px = sep_as / pxs
        # Derotated (sky/output-frame) position: pa_deg alone, no pa[0] term.
        # Must match run_paco_challenge.py's inject_companion_into_cube.
        theta  = np.deg2rad(pa_deg)
        x = cx + sep_px * np.sin(theta)
        y = cy - sep_px * np.cos(theta)
        positions.append((x, y))
    return positions


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Detection Maps Gallery (all 9 datasets × 2 algorithms)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 1: detection maps gallery …")

INST_LABELS = {
    "sphere_irdis": "VLT/SPHERE-IRDIS",
    "nirc2":        "Keck/NIRC2",
    "lmircam":      "LBT/LMIRCam",
}

fig, axes = plt.subplots(
    9, 3,
    figsize=(11, 30),
    gridspec_kw={"wspace": 0.05, "hspace": 0.35},
)
fig.patch.set_facecolor("#111111")

col_labels = ["FullPACO", "FastPACO", "Difference\n(Fast − Full)"]
for col, lbl in enumerate(col_labels):
    axes[0, col].set_title(lbl, fontsize=11, color="white", pad=8)

row = 0
for inst in ["sphere_irdis", "nirc2", "lmircam"]:
    for idx in [1, 2, 3]:
        label = f"{inst}_{idx}"
        fwhm  = compute_fwhm_from_psf(inst, idx)
        snr_fast = load_snr(inst, idx, "fast")
        snr_full = load_snr(inst, idx, "full")

        for col, (variant, snr) in enumerate([("full", snr_full), ("fast", snr_fast)]):
            ax = axes[row, col]
            ax.set_facecolor("#111111")
            if snr is None:
                ax.text(0.5, 0.5, "Not computed\n(LMIRCam)", color="white",
                        ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                row_label = f"{INST_LABELS[inst]}\n#{idx}"
                if col == 0:
                    ax.set_ylabel(row_label, color="white", fontsize=8)
                continue
            im, peaks = plot_snr_map(ax, snr, fwhm, THRESHOLD,
                                     title=f"{INST_LABELS[inst]} #{idx} – "
                                           f"{'Full' if variant=='full' else 'Fast'}PACO",
                                     cmap="magma")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
                labelsize=6, colors="white")

        # Difference panel
        ax = axes[row, 2]
        ax.set_facecolor("#111111")
        if snr_fast is not None and snr_full is not None:
            diff = snr_fast - snr_full
            vd   = np.nanpercentile(np.abs(diff), 99)
            im2  = ax.imshow(np.nan_to_num(diff, nan=0.0), origin="lower",
                             cmap="RdBu_r", vmin=-vd, vmax=vd,
                             interpolation="nearest")
            ax.set_title(f"Δ SNR  std={np.nanstd(diff):.2f}", fontsize=8,
                         color="white", pad=3)
            plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
                labelsize=6, colors="white")
        else:
            ax.text(0.5, 0.5, "—", color="white", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])
        row += 1

for ax in axes.ravel():
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

plt.suptitle("FullPACO vs FastPACO – ADI Subchallenge Detection Maps",
             color="white", fontsize=14, y=1.002)

fig.savefig(os.path.join(FIG_DIR, "figure1_detection_maps_all.pdf"),
            bbox_inches="tight", facecolor="#111111", dpi=150)
plt.close(fig)
print("  ✓ figure1_detection_maps_all.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Per-dataset detail: SPHERE-IRDIS
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 2: SPHERE-IRDIS detail …")

inst = "sphere_irdis"
fig2, axes2 = plt.subplots(3, 4, figsize=(16, 12),
                            gridspec_kw={"wspace": 0.15, "hspace": 0.35})
fig2.suptitle("SPHERE-IRDIS Datasets – FullPACO vs FastPACO", fontsize=13)

for row, idx in enumerate([1, 2, 3]):
    fwhm = compute_fwhm_from_psf(inst, idx)
    snr_full = load_snr(inst, idx, "full")
    snr_fast = load_snr(inst, idx, "fast")
    label = f"{inst}_{idx}"
    zero_inj = label in ZERO_INJECTION

    for col, (variant, snr) in enumerate([("full", snr_full), ("fast", snr_fast)]):
        ax = axes2[row, col]
        if snr is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        peaks = find_peaks(snr, fwhm, THRESHOLD)
        im, _ = plot_snr_map(ax, snr, fwhm, THRESHOLD,
                              title=f"sph{idx} – {'Full' if variant=='full' else 'Fast'}PACO "
                                    f"(det={len(peaks)})",
                              cmap="magma")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
        if zero_inj:
            ax.text(0.02, 0.98, "0 injections", color="cyan", fontsize=7,
                    transform=ax.transAxes, va="top")

    # Peak SNR histogram (col 2)
    ax3 = axes2[row, 2]
    for variant, snr, color, lbl in [
            ("full", snr_full, "steelblue", "FullPACO"),
            ("fast", snr_fast, "darkorange", "FastPACO")]:
        if snr is None:
            continue
        ny, nx = snr.shape
        mask = circular_mask(ny, nx, 2 * fwhm, min(ny, nx) / 2 - 1)
        vals = snr[mask]
        vals = vals[~np.isnan(vals)]
        ax3.hist(vals, bins=50, alpha=0.5, color=color, label=lbl, density=True)
    ax3.axvline(THRESHOLD, color="red", linestyle="--", lw=1, label=f"{THRESHOLD}σ")
    ax3.set_xlabel("SNR", fontsize=8)
    ax3.set_ylabel("Density", fontsize=8)
    ax3.legend(fontsize=7)
    ax3.set_title(f"sph{idx} SNR distribution", fontsize=9)
    ax3.tick_params(labelsize=7)

    # Stats (col 3)
    ax4 = axes2[row, 3]
    ax4.axis("off")
    lines = [f"Dataset: sph{idx}",
             f"Zero inj: {zero_inj}",
             f"FWHM: {fwhm:.2f} px", ""]
    for variant, snr, lbl in [("full", snr_full, "FullPACO"),
                                ("fast", snr_fast, "FastPACO")]:
        if snr is None:
            lines += [f"{lbl}: N/A", ""]
            continue
        ny, nx = snr.shape
        mask = circular_mask(ny, nx, 2 * fwhm, min(ny, nx) / 2 - 1)
        vals = snr[mask]
        vals = vals[~np.isnan(vals)]
        peaks = find_peaks(snr, fwhm, THRESHOLD)
        lines += [f"{lbl}:",
                  f"  max SNR: {np.nanmax(snr):.2f}",
                  f"  std: {np.nanstd(vals):.2f}",
                  f"  peaks>{THRESHOLD}σ: {len(peaks)}", ""]
    ax4.text(0.05, 0.95, "\n".join(lines), transform=ax4.transAxes,
             va="top", ha="left", fontsize=8, family="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))

fig2.savefig(os.path.join(FIG_DIR, "figure2_sphere_irdis_detail.pdf"),
             bbox_inches="tight", dpi=150)
plt.close(fig2)
print("  ✓ figure2_sphere_irdis_detail.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Per-dataset detail: NIRC2
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 3: NIRC2 detail …")

inst = "nirc2"
fig3, axes3 = plt.subplots(3, 4, figsize=(16, 12),
                            gridspec_kw={"wspace": 0.15, "hspace": 0.35})
fig3.suptitle("Keck/NIRC2 Datasets – FullPACO vs FastPACO", fontsize=13)

for row, idx in enumerate([1, 2, 3]):
    fwhm = compute_fwhm_from_psf(inst, idx)
    snr_full = load_snr(inst, idx, "full")
    snr_fast = load_snr(inst, idx, "fast")
    label = f"{inst}_{idx}"
    zero_inj = label in ZERO_INJECTION

    for col, (variant, snr) in enumerate([("full", snr_full), ("fast", snr_fast)]):
        ax = axes3[row, col]
        if snr is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        peaks = find_peaks(snr, fwhm, THRESHOLD)
        im, _ = plot_snr_map(ax, snr, fwhm, THRESHOLD,
                              title=f"nrc{idx} – {'Full' if variant=='full' else 'Fast'}PACO "
                                    f"(det={len(peaks)})",
                              cmap="magma")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
        if zero_inj:
            ax.text(0.02, 0.98, "0 injections", color="cyan", fontsize=7,
                    transform=ax.transAxes, va="top")

    ax3 = axes3[row, 2]
    for variant, snr, color, lbl in [
            ("full", snr_full, "steelblue", "FullPACO"),
            ("fast", snr_fast, "darkorange", "FastPACO")]:
        if snr is None:
            continue
        ny, nx = snr.shape
        mask = circular_mask(ny, nx, 2 * fwhm, min(ny, nx) / 2 - 1)
        vals = snr[mask]; vals = vals[~np.isnan(vals)]
        ax3.hist(vals, bins=50, alpha=0.5, color=color, label=lbl, density=True)
    ax3.axvline(THRESHOLD, color="red", linestyle="--", lw=1)
    ax3.set_xlabel("SNR", fontsize=8); ax3.legend(fontsize=7)
    ax3.set_title(f"nrc{idx} SNR distribution", fontsize=9)
    ax3.tick_params(labelsize=7)

    ax4 = axes3[row, 3]; ax4.axis("off")
    lines = [f"Dataset: nrc{idx}", f"Zero inj: {zero_inj}",
             f"FWHM: {fwhm:.2f} px", ""]
    for variant, snr, lbl in [("full", snr_full, "FullPACO"),
                                ("fast", snr_fast, "FastPACO")]:
        if snr is None:
            lines += [f"{lbl}: N/A", ""]
            continue
        ny, nx = snr.shape
        mask = circular_mask(ny, nx, 2 * fwhm, min(ny, nx) / 2 - 1)
        vals = snr[mask]; vals = vals[~np.isnan(vals)]
        peaks = find_peaks(snr, fwhm, THRESHOLD)
        lines += [f"{lbl}:", f"  max SNR: {np.nanmax(snr):.2f}",
                  f"  std: {np.nanstd(vals):.2f}",
                  f"  peaks>{THRESHOLD}σ: {len(peaks)}", ""]
    ax4.text(0.05, 0.95, "\n".join(lines), transform=ax4.transAxes,
             va="top", ha="left", fontsize=8, family="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))

fig3.savefig(os.path.join(FIG_DIR, "figure3_nirc2_detail.pdf"),
             bbox_inches="tight", dpi=150)
plt.close(fig3)
print("  ✓ figure3_nirc2_detail.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – LMIRCam detection maps (FastPACO only)
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 4: LMIRCam detection maps …")

inst = "lmircam"
fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5),
                            gridspec_kw={"wspace": 0.12})
fig4.suptitle("LBT/LMIRCam Datasets – FastPACO Detection Maps", fontsize=13)

for col, idx in enumerate([1, 2, 3]):
    fwhm = compute_fwhm_from_psf(inst, idx)
    snr  = load_snr(inst, idx, "fast")
    ax   = axes4[col]
    if snr is None:
        ax.text(0.5, 0.5, "Running…", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_title(f"lmr{idx} – FastPACO (pending)", fontsize=10)
        continue
    peaks = find_peaks(snr, fwhm, THRESHOLD)
    im, _ = plot_snr_map(ax, snr, fwhm, THRESHOLD,
                         title=f"lmr{idx} – FastPACO (det={len(peaks)})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

fig4.savefig(os.path.join(FIG_DIR, "figure4_lmircam_detection.pdf"),
             bbox_inches="tight", dpi=150)
plt.close(fig4)
print("  ✓ figure4_lmircam_detection.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 – Metric curves from self-injection test
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 5: metric curves from injection test …")

INJECT_DSETS = [("sphere_irdis", 2), ("nirc2", 2)]

fig5, axes5 = plt.subplots(
    len(INJECT_DSETS), 4,
    figsize=(18, 5 * len(INJECT_DSETS)),
    gridspec_kw={"wspace": 0.25, "hspace": 0.4})

for row, (inst, idx) in enumerate(INJECT_DSETS):
    fwhm = compute_fwhm_from_psf(inst, idx)
    positions = get_injection_positions(inst, idx)
    inst_label = {"sphere_irdis": "SPHERE-IRDIS", "nirc2": "NIRC2"}[inst]

    for col, variant in enumerate(["full", "fast"]):
        m = rebuild_inject_metrics(inst, idx, variant, positions, fwhm)
        snr_inj = load_inject_snr(inst, idx, variant)
        vname = "FullPACO" if variant == "full" else "FastPACO"

        # col 0/1: SNR map with injections
        ax_map = axes5[row, col]
        if snr_inj is None:
            ax_map.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax_map.transAxes)
        else:
            im, peaks = plot_snr_map(ax_map, snr_inj, fwhm, THRESHOLD,
                                     title=f"{inst_label} #{idx}\n{vname} + injections")
            plt.colorbar(im, ax=ax_map, fraction=0.046).ax.tick_params(labelsize=7)
            for px, py in positions:
                c = Circle((px, py), fwhm / 2 * + 4,
                            color="deepskyblue", fill=False, lw=1.5, linestyle="--")
                ax_map.add_patch(c)

        # col 2/3: metric curves
        ax_c = axes5[row, 2 + col]
        if m is not None:
            plot_metric_curves(ax_c, m["thr"], m["tpr"], m["fdr"], m["fpr"],
                               THRESHOLD, m["auc_tpr"], m["auc_fdr"], m["f1"],
                               title=f"{vname} – Self-injection")
        else:
            ax_c.text(0.5, 0.5, "No data", ha="center", va="center",
                      transform=ax_c.transAxes)

fig5.suptitle("Self-Injection Metric Test\n"
              "(Blue dashed circles = injected companion positions; "
              "gold = detected above 5σ)", fontsize=12)
fig5.savefig(os.path.join(FIG_DIR, "figure5_metric_curves.pdf"),
             bbox_inches="tight", dpi=150)
plt.close(fig5)
print("  ✓ figure5_metric_curves.pdf")

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
    ("FullPACO (ours)", "figure6_sph3_full.pdf", snr_full, "full"),
    ("FastPACO (ours)", "figure6_sph3_fast.pdf", snr_fast, "fast"),
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
    for variant, lbl2 in [("full", "FullPACO (ours)"), ("fast", "FastPACO (ours)")]:
        if variant not in sph3_metrics:
            w.writerow(["ours", lbl2, "", "", "", "", "", "", 5])
            continue
        m = sph3_metrics[variant]
        w.writerow(["ours", lbl2, f"{m['f1']:.4f}", f"{m['auc_tpr']:.4f}",
                   f"{m['auc_fdr']:.4f}", f"{m['auc_fpr']:.4f}",
                   m["n_det"], m["n_fp"], 5])
print(f"  ✓ {os.path.basename(csv_path)}")

print(f"\nAll figures saved to {FIG_DIR}/")
