#!/usr/bin/env python3
"""
E1 -- null-distribution test: is the S/N map N(0,1) where there is no
companion, for this implementation and for VIP?

PACO's central guarantee (Flasseur et al. 2018, Eq. 24) is that the S/N
map b/sqrt(a) is N(0,1) under the no-companion hypothesis. We cannot
compare against the authors' (private) MATLAB code, so we test the
implementation against that prediction instead.

A companion-free ("null") reduction is built by deliberately mis-rotating
the frames, so that no real or injected source follows a PACO trajectory
(same idea as the source-free reduction of Wahhaj et al. 2013, ApJ 779, 80,
Sect. 4.1, who multiply the PAs by 30). The default here is a *sign
reversal* of the parallactic angles: it keeps the rotation amplitude, so
trajectory lengths -- and the cache geometry -- are the same as in the real
run. --null scale30 reproduces the Wahhaj et al. variant.

Two parts:

1. --check-covariance (seconds): computes (m, C^-1) for real sph3 patch
   columns with this project's math_utils and with VIP's own functions,
   and reports the C^-1 ratio and the two shrinkage factors. Reading VIP's
   code predicts S_VIP = S_paper / 2 (its sample_covariance uses np.cov on
   the 2-row stack (p, m)), hence C^-1 x2 and S/N inflated by sqrt(2).
   This checks that prediction directly, independent of the full runs.

2. Null runs (minutes per map): FullPACOv2 (production, ring traversal),
   FastPACOv2, and VIP FastPACO (convention A of the appendix: fwhm in px,
   pixscale=1.0 -- same K=29 patch as ours), all on the sign-reversed sph3
   cube. Reports mean, std, robust std (1.4826*MAD), skewness, excess
   kurtosis, tail fractions vs the Gaussian expectation, and the KS
   distance to N(0,1), inside an annulus that excludes the coronagraphic
   core and the frame edge. Pixels of an S/N map are spatially correlated
   (~one FWHM), so the KS p-value is NOT meaningful; the KS distance and
   the moments are what to compare.

Optional --signal: compares the S/N at the 5 injected companions in the
*real* (non-null) maps already saved under challenge/outputs/snr_maps/,
ours vs VIP. If the covariance factor is the whole story, VIP/ours should
be ~sqrt(2) there too.

Outputs go to challenge/outputs/e1_null/: the null S/N maps (.npy), a
statistics CSV, and a histogram + map figure.

Usage
-----
    python3 benchmarks/e1_null_test.py --check-covariance
    python3 benchmarks/e1_null_test.py --cpu 16
    python3 benchmarks/e1_null_test.py --cpu 16 --cached --signal
    python3 benchmarks/e1_null_test.py --cpu 16 --null scale30 --skip-vip
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

import numpy as np
from scipy import stats

from challenge.make_contrast_curve_sph3 import load_sph3, compute_fwhm
from src.fast_paco import FastPACOv2
from src.full_paco import FullPACOv2
from src.math_utils import compute_statistics_at_pixel
from src.utils import create_boolean_circular_mask

OUT_DIR = os.path.join(ROOT, "challenge", "outputs", "e1_null")
SNR_DIR = os.path.join(ROOT, "challenge", "outputs", "snr_maps")

# Same 5 positions used everywhere else in this project (see
# challenge/run_vippaco_sph3.py for provenance).
SPH3_GROUND_TRUTH_XY = [
    (112.67, 90.22),
    (82.70, 58.70),
    (41.05, 86.18),
    (55.40, 95.61),
    (129.24, 82.90),
]

# Two-sided Gaussian tail probabilities, for comparison with the measured
# fractions.
GAUSS_TAIL = {3: 2 * stats.norm.sf(3), 5: 2 * stats.norm.sf(5)}


# ---------------------------------------------------------------------------
# Part 1: covariance check
# ---------------------------------------------------------------------------

def patch_column(cube, y, x, radius):
    """(T, K) stack of the circular patch centred on (y, x), every frame."""
    T, H, W = cube.shape
    mask = create_boolean_circular_mask((H, W), radius=radius, center=(y, x))
    return cube[:, mask]


def check_covariance(cube, fwhm, n_pixels=200, seed=0):
    """Compare C^-1 and rho between this project and VIP on real patches."""
    from vip_hci.invprob.paco import (compute_statistics_at_pixel as vip_stats,
                                      sample_covariance as vip_sample_cov,
                                      shrinkage_factor as vip_rho)
    from src.math_utils import shrinkage_factor as our_rho

    T, H, W = cube.shape
    radius = int(2 * fwhm) // 2  # same patch_width//2 convention as the runners
    rng = np.random.default_rng(seed)
    margin = radius + 1
    ys = rng.integers(margin, H - margin, n_pixels)
    xs = rng.integers(margin, W - margin, n_pixels)

    # Matched-filter template over the patch: a Gaussian of the PSF's FWHM
    # centred on the patch centre (what a and b are built from).
    pm = create_boolean_circular_mask((2 * radius + 1,) * 2, radius=radius,
                                      center=(radius, radius))
    gy, gx = np.nonzero(pm)
    sigma = fwhm / 2.355
    h = np.exp(-((gy - radius) ** 2 + (gx - radius) ** 2) / (2 * sigma ** 2))

    def null_std(cinv, s_ref):
        # Per-frame null std of b/sqrt(a) when the true covariance is s_ref:
        # sqrt(h' Ci S Ci h) / sqrt(h' Ci h). Equals 1 when Ci = S^-1.
        w = cinv @ h
        return np.sqrt(w @ s_ref @ w) / np.sqrt(h @ w)

    a_ratio, std_ours, std_vip, rho_ours, rho_vip, s_ratio = [], [], [], [], [], []
    for y, x in zip(ys, xs):
        p = patch_column(cube, y, x, radius)
        _, cinv_o = compute_statistics_at_pixel(p)
        _, cinv_v = vip_stats(p)
        if cinv_o is None or cinv_v is None or p.shape[1] != h.size:
            continue
        m = p.mean(axis=0)
        s_paper = (p - m).T @ (p - m) / T          # Flasseur et al. Eq. 7
        s_vip = vip_sample_cov(p, m, T)
        s_ratio.append(np.median(s_vip / s_paper))
        rho_ours.append(our_rho(s_paper, T))
        rho_vip.append(vip_rho(s_vip, T))
        a_ratio.append((h @ cinv_v @ h) / (h @ cinv_o @ h))
        # s_paper is an in-sample proxy for the true covariance, so these
        # absolute values are optimistic; their ratio is what matters.
        std_ours.append(null_std(cinv_o, s_paper))
        std_vip.append(null_std(cinv_v, s_paper))

    s_ratio, a_ratio = np.array(s_ratio), np.array(a_ratio)
    std_ours, std_vip = np.array(std_ours), np.array(std_vip)
    rho_ours, rho_vip = np.array(rho_ours), np.array(rho_vip)
    print(f"\n=== Covariance check: {len(s_ratio)} real sph3 patch columns "
          f"(T={T}, K={h.size}) ===")
    print(f"  median S_VIP / S_paper (elementwise) : {np.median(s_ratio):.4f}"
          f"   [prediction: 0.5]")
    print(f"  rho, this work : median {np.median(rho_ours):.4f} "
          f"[{rho_ours.min():.4f}, {rho_ours.max():.4f}]")
    print(f"  rho, VIP       : median {np.median(rho_vip):.4f} "
          f"[{rho_vip.min():.4f}, {rho_vip.max():.4f}]")
    print(f"  median a_VIP / a_ours (h' Ci h)       : {np.median(a_ratio):.4f}"
          f"   [2.0 if only the S factor differed]")
    print(f"  predicted per-frame null std, ours    : {np.median(std_ours):.4f}")
    print(f"  predicted per-frame null std, VIP     : {np.median(std_vip):.4f}")
    print(f"  predicted inflation VIP / ours        : "
          f"{np.median(std_vip / std_ours):.4f}   [sqrt(2) = {np.sqrt(2):.4f}]")
    return {"s_ratio": float(np.median(s_ratio)),
            "a_ratio": float(np.median(a_ratio)),
            "rho_ours": float(np.median(rho_ours)),
            "rho_vip": float(np.median(rho_vip)),
            "pred_inflation": float(np.median(std_vip / std_ours))}


# ---------------------------------------------------------------------------
# Part 2: null runs
# ---------------------------------------------------------------------------

def null_angles(pa, mode):
    if mode == "reverse":
        return -pa
    if mode == "scale30":
        return 30.0 * pa
    raise ValueError(mode)


def run_ours_full(cube, pa, psf, fwhm, pxs, cpu):
    obj = FullPACOv2(cube, pa, psf, fwhm=fwhm, pixscale=pxs, verbose=False)
    return obj.run(cpu=cpu)


def run_ours_fast(cube, pa, psf, fwhm, pxs, cpu):
    obj = FastPACOv2(cube, pa, psf, fwhm=fwhm, pixscale=pxs, verbose=False)
    return obj.run(cpu=cpu)


def run_vip_fast(cube, pa, psf, fwhm, pxs, cpu):
    from vip_hci.invprob.paco import FastPACO as VIP_FastPACO
    # Convention A of Appendix B: fwhm in px, pixscale left at 1.0, so
    # VIP's self.fwhm = int(fwhm) is used as the patch radius (K=29 here,
    # matching ours).
    obj = VIP_FastPACO(cube, pa, psf, fwhm=fwhm, pixscale=1.0, verbose=False)
    snr, _flux = obj.run(cpu=cpu)
    return snr


RUNNERS = {
    "ours_full": ("FullPACOv2 (this work)", run_ours_full),
    "ours_fast": ("FastPACOv2 (this work)", run_ours_fast),
    "vip_fast": ("VIP FastPACO", run_vip_fast),
}


def analysis_mask(shape, fwhm):
    """Annulus excluding the coronagraphic core (< 2 FWHM) and the edge."""
    ny, nx = shape
    yy, xx = np.ogrid[:ny, :nx]
    r = np.hypot(yy - ny // 2, xx - nx // 2)
    return (r >= 2 * fwhm) & (r <= min(ny, nx) / 2 - 1)


def null_stats(snr, mask, fwhm):
    v = snr[mask & np.isfinite(snr)]
    n = v.size
    mad_std = 1.4826 * np.median(np.abs(v - np.median(v)))
    n_indep = n / (np.pi * (fwhm / 2) ** 2)  # ~ resolution elements
    row = {
        "n_pix": n,
        "n_resel": round(n_indep),
        "mean": v.mean(),
        "std": v.std(ddof=1),
        "robust_std": mad_std,
        "skew": stats.skew(v),
        "excess_kurt": stats.kurtosis(v),
        "ks_distance": stats.kstest(v, "norm").statistic,
    }
    for k in (3, 5):
        row[f"frac_abs_gt_{k}"] = np.mean(np.abs(v) > k)
        row[f"gauss_abs_gt_{k}"] = GAUSS_TAIL[k]
    row["n_gt_5"] = int(np.sum(v > 5))
    return row, v


def signal_comparison(fwhm):
    """S/N at the 5 injected companions in the saved REAL maps."""
    paths = {
        "ours_full": "sphere_irdis_3_full_snr.npy",
        "ours_fast": "sphere_irdis_3_fast_snr.npy",
        "vip_fast": "sphere_irdis_3_vippaco_convA_snr.npy",
    }
    maps = {}
    for k, name in paths.items():
        p = os.path.join(SNR_DIR, name)
        if os.path.exists(p):
            maps[k] = np.load(p)
        else:
            print(f"  [signal] missing {p} -- skipped")
    if "vip_fast" not in maps:
        return
    r = max(1, int(round(fwhm / 2)))

    def peak(snr, x, y):
        xi, yi = int(round(x)), int(round(y))
        win = snr[max(0, yi - r):yi + r + 1, max(0, xi - r):xi + r + 1]
        return np.nanmax(win)

    print("\n=== Signal: local peak S/N at the 5 injected companions (real maps) ===")
    header = f"{'companion (x,y)':<20}" + "".join(f"{k:>12}" for k in maps)
    for k in maps:
        if k != "vip_fast":
            header += f"{'VIP/' + k:>16}"
    print(header)
    for x, y in SPH3_GROUND_TRUTH_XY:
        vals = {k: peak(m, x, y) for k, m in maps.items()}
        line = f"({x:6.1f},{y:6.1f})      " + "".join(f"{vals[k]:12.2f}" for k in maps)
        for k in maps:
            if k != "vip_fast":
                line += f"{vals['vip_fast'] / vals[k]:16.3f}"
        print(line)
    print(f"  [sqrt(2) = {np.sqrt(2):.3f}]")


def make_figure(results, mask, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(results)
    fig, axes = plt.subplots(2, len(keys), figsize=(4.2 * len(keys), 8.0),
                             squeeze=False)
    grid = np.linspace(-6, 6, 400)
    for col, key in enumerate(keys):
        label, snr, row, v = results[key]
        ax = axes[0, col]
        shown = np.where(mask, snr, np.nan)
        im = ax.imshow(shown, origin="lower", cmap="RdBu_r", vmin=-5, vmax=5,
                       interpolation="nearest")
        ax.set_title(f"{label}\nnull S/N map", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        axh = axes[1, col]
        axh.hist(v, bins=120, range=(-8, 8), density=True, color="0.4",
                 histtype="stepfilled", alpha=0.7, label="null S/N")
        axh.plot(grid, stats.norm.pdf(grid), "k--", lw=1.2, label="N(0,1)")
        axh.plot(grid, stats.norm.pdf(grid, row["mean"], row["std"]), "C3", lw=1.0,
                 label=f"N({row['mean']:.2f}, {row['std']:.2f}$^2$)")
        axh.set_yscale("log")
        axh.set_ylim(1e-6, 1)
        axh.set_xlabel("S/N")
        axh.legend(fontsize=8, loc="upper right")
        axh.set_title(f"std={row['std']:.3f}  robust={row['robust_std']:.3f}  "
                      f"KS D={row['ks_distance']:.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    out_pdf = os.path.join(ROOT, "Astronomy_and_Astrophysics_Template", "figures",
                           os.path.basename(out_png).replace(".png", ".pdf"))
    fig.savefig(out_pdf)
    print(f"Wrote {out_png}\nWrote {out_pdf}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu", type=int, default=4)
    ap.add_argument("--null", choices=["reverse", "scale30"], default="reverse",
                    help="How to build the companion-free cube (default: "
                         "sign-reversed parallactic angles).")
    ap.add_argument("--check-covariance", action="store_true",
                    help="Only run the fast C^-1 / rho comparison and exit.")
    ap.add_argument("--only", nargs="+", choices=list(RUNNERS),
                    default=list(RUNNERS), help="Subset of maps to compute.")
    ap.add_argument("--skip-vip", action="store_true")
    ap.add_argument("--cached", action="store_true",
                    help="Reuse null maps already saved in the output folder.")
    ap.add_argument("--signal", action="store_true",
                    help="Also compare companion S/N in the saved real maps.")
    args = ap.parse_args()

    cube, pa, psf, pxs = load_sph3()
    fwhm = compute_fwhm(psf)
    print(f"sph3: cube={cube.shape}, fwhm={fwhm:.3f} px, pxscale={pxs:.5f}, "
          f"PA range=[{pa.min():.1f}, {pa.max():.1f}] deg "
          f"(total rotation {np.ptp(pa):.1f} deg)")

    cov = check_covariance(cube, fwhm)
    if args.check_covariance:
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    pa_null = null_angles(pa, args.null)
    keys = [k for k in args.only if not (args.skip_vip and k.startswith("vip"))]
    mask = analysis_mask(cube.shape[1:], fwhm)

    results = {}
    for key in keys:
        label, runner = RUNNERS[key]
        path = os.path.join(OUT_DIR, f"sph3_null_{args.null}_{key}.npy")
        if args.cached and os.path.exists(path):
            snr = np.load(path)
            print(f"\n[{label}] loaded {path}")
        else:
            print(f"\n[{label}] running on the {args.null} null cube (cpu={args.cpu}) ...")
            t0 = time.perf_counter()
            snr = np.asarray(runner(cube, pa_null, psf, fwhm, pxs, args.cpu))
            print(f"  done in {time.perf_counter() - t0:.1f} s")
            np.save(path, snr)
        row, v = null_stats(snr, mask, fwhm)
        results[key] = (label, snr, row, v)

    print(f"\n=== Null S/N statistics ({args.null}; annulus 2 FWHM .. edge) ===")
    cols = ["n_pix", "n_resel", "mean", "std", "robust_std", "skew",
            "excess_kurt", "ks_distance", "frac_abs_gt_3", "frac_abs_gt_5", "n_gt_5"]
    print(f"{'map':<26}" + "".join(f"{c:>14}" for c in cols))
    for key, (label, _snr, row, _v) in results.items():
        cells = []
        for c in cols:
            x = row[c]
            cells.append(f"{x:14d}" if isinstance(x, (int, np.integer))
                         else f"{x:14.3g}" if c.startswith("frac") else f"{x:14.4f}")
        print(f"{label:<26}" + "".join(cells))
    print(f"{'N(0,1) expectation':<26}{'':>14}{'':>14}{0:14.4f}{1:14.4f}{1:14.4f}"
          f"{0:14.4f}{0:14.4f}{'':>14}{GAUSS_TAIL[3]:14.3g}{GAUSS_TAIL[5]:14.3g}")

    if "vip_fast" in results:
        for key in ("ours_full", "ours_fast"):
            if key in results:
                a, b = results["vip_fast"][2], results[key][2]
                print(f"  std ratio VIP / {key}: {a['std'] / b['std']:.3f}   "
                      f"robust: {a['robust_std'] / b['robust_std']:.3f}   "
                      f"[sqrt(2) = {np.sqrt(2):.3f}]")

    csv_path = os.path.join(OUT_DIR, f"sph3_null_{args.null}_stats.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        fields = ["map"] + list(next(iter(results.values()))[2].keys())
        w.writerow(fields)
        for key, (label, _s, row, _v) in results.items():
            w.writerow([key] + [row[k] for k in fields[1:]])
        w.writerow([])
        w.writerow(["covariance_check"] + [f"{k}={v:.4f}" for k, v in cov.items()])
    print(f"Wrote {csv_path}")

    make_figure(results, mask, os.path.join(OUT_DIR, f"sph3_null_{args.null}.png"))

    if args.signal:
        signal_comparison(fwhm)


if __name__ == "__main__":
    main()
