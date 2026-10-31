#!/usr/bin/env python3
"""
Experiments requested by the second simulated review. Results go to
challenge/outputs/round2/<dataset>/.

  null      Companion-free realizations of one or more datasets, built by
            cyclically rolling the angle list by m frames, with the offsets
            m drawn WITHOUT replacement from [T/4, 3T/4): every map is
            distinct. For each map: global and per-annulus robust null
            width, tail fractions at 3, 4 and 5 sigma (raw and per-annulus
            normalized) against the Gaussian, groups of >= 2 pixels above 5,
            and the fraction of frames the roll leaves within one FWHM of
            their true position at the inner and outer analysis radius
            (frames along which a real source stays coherent). The summary
            gives T, K and T/K per dataset, so the null width can be read
            against T/K, and the false-alarm rate per resolution element
            with a 95% Poisson interval.

  ablation  The two numerical deviations of VIP's PACO module switched on in
            this code, one at a time and together:
              published  Flasseur et al. (2018): 1/T sample covariance,
                         diagonal sum in the shrinkage numerator
              halfcov    sample covariance halved (VIP)
              fullsum    shrinkage numerator summed over the whole matrix
                         (VIP)
              vip        both
            Optionally VIP's own FastPACO as a fifth variant (--vip-code).
            For each variant: null width from rolled maps (same offsets for
            every variant), then injected companions (sub-pixel positions,
            same cubes for every variant) and the dataset's real companions,
            each calibrated by that variant's own per-annulus null. This is
            the matched-false-alarm comparison: it shows which deviation
            widens the null and whether a recalibrated map loses S/N.

Datasets: sph1, sph2, sph3, nirc2_1, nirc2_2, nirc2_3 (EIDC sub-challenge 1
files in challenge/subchallenge_1/), and hr8799 (--cube/--angles/--psf;
angles negated as for rotnth.fits, centered crop --size).

Usage
-----
    python3 benchmarks/round2_experiments.py null --datasets sph1 sph2 sph3 \\
        nirc2_1 nirc2_2 nirc2_3 --realizations 20 --cpu 32
    python3 benchmarks/round2_experiments.py null --datasets sph3 --methods fast full \\
        --realizations 114 --full 20 --cpu 32
    python3 benchmarks/round2_experiments.py ablation --datasets sph3 --null 20 \\
        --cubes 6 --per-cube 6 --cpu 32
    python3 benchmarks/round2_experiments.py ablation --datasets hr8799 \\
        --cube data/HR8799/center_im.fits --angles data/HR8799/rotnth.fits \\
        --psf data/HR8799/median_unsat.fits --size 400 --null 10 --cubes 4 \\
        --per-cube 8 --cpu 32
"""
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import numpy as np
from astropy.io import fits
from scipy import ndimage, stats

import src.fast_paco as fast_mod
import src.full_paco as full_mod
from challenge.make_contrast_curve_sph3 import compute_fwhm
from src.fast_paco import FastPACO
from src.full_paco import PACO
from src.math_utils import compute_statistics_at_pixel
from review_experiments import inject, peak_near, wilson, write_csv

OUT = os.path.join(ROOT, "challenge", "outputs", "round2")
EIDC = os.path.join(ROOT, "challenge", "subchallenge_1")
EIDC_FILES = {"sph": "sphere_irdis", "nirc2": "nirc2"}
PXSCALE = {"sph": 0.01225, "nirc2": 0.00996}
SPH3_GT = [(112.67, 90.22), (82.70, 58.70), (41.05, 86.18), (55.40, 95.61), (129.24, 82.90)]
HR8799_SEP = {"b": 1.72, "c": 0.95, "d": 0.68, "e": 0.41}  # arcsec
GAUSS_TAIL = {k: float(2 * stats.norm.sf(k)) for k in (3, 4, 5)}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(name, args):
    """cube, angles, psf, fwhm, pixscale, known sources [(x, y)]."""
    if name == "hr8799":
        from e10_full_frame import crop, load as e10_load
        ns = argparse.Namespace(cube=args.cube, angles=args.angles, psf=args.psf,
                                channel=args.channel, psf_size=None, frames=None, fwhm=None)
        cube, pa, psf, fwhm = e10_load(ns)
        cube, pa = crop(cube, args.size), -pa  # rotnth.fits convention
        return cube, pa, psf, fwhm, 0.01225, None
    inst, idx = name.rstrip("0123456789").rstrip("_"), name[-1]
    stem = EIDC_FILES[inst]

    def fp(kind):
        return os.path.join(EIDC, f"{stem}_{kind}_{idx}.fits")

    cube = fits.getdata(fp("cube")).astype(np.float64)
    pa = fits.getdata(fp("pa")).astype(np.float64).ravel()
    psf = fits.getdata(fp("psf")).astype(np.float64)
    if psf.ndim == 3:
        psf = np.nanmedian(psf, axis=0)
    pxs = float(np.asarray(fits.getdata(fp("pxscale"))).flat[0])
    if not np.isfinite(pxs) or pxs <= 0:
        pxs = PXSCALE[inst]
    return np.nan_to_num(cube), pa, psf, compute_fwhm(psf), pxs, \
        (SPH3_GT if name == "sph3" else None)


def hr8799_sources(snr, fwhm, pxs):
    """Planet positions: the S/N peak within 0.08 arcsec of each known
    separation (as in e10_full_frame.py)."""
    H, W = snr.shape
    yy, xx = np.mgrid[:H, :W]
    r = np.hypot(yy - H // 2, xx - W // 2)
    out = []
    for sep in HR8799_SEP.values():
        ring = np.abs(r - sep / pxs) <= 0.08 / pxs
        i = np.nanargmax(np.where(ring, snr, -np.inf))
        out.append((float(xx.flat[i]), float(yy.flat[i])))
    return out


def radius(shape):
    H, W = shape
    yy, xx = np.mgrid[:H, :W]
    return np.hypot(yy - H // 2, xx - W // 2)


def annuli(shape, fwhm):
    width = max(2 * fwhm, 8.0)
    return np.arange(2 * fwhm, min(shape) / 2 - 1, width)


def patch_area(fwhm):
    w = int(2 * fwhm)
    y, x = np.mgrid[:w, :w]
    return int((np.hypot(y - w // 2, x - w // 2) <= w // 2).sum())


def robust_std(v):
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size > 10 else float("nan")


def annulus_widths(snr, edges):
    r = radius(snr.shape)
    return [robust_std(snr[(r >= a) & (r < b)]) for a, b in zip(edges[:-1], edges[1:])]


def normalize(snr, edges, widths):
    r = radius(snr.shape)
    z = np.full(snr.shape, np.nan)
    for (a, b), w in zip(zip(edges[:-1], edges[1:]), widths):
        sel = (r >= a) & (r < b) & np.isfinite(snr)
        if np.isfinite(w) and w > 0:
            z[sel] = snr[sel] / w
    return z


def groups_above(z, thr):
    lab, n = ndimage.label(np.nan_to_num(z, nan=0.0) > thr)
    sizes = np.bincount(lab.ravel())[1:]
    return int((sizes >= 2).sum())


def offsets(T, n, rng):
    pool = np.arange(T // 4, 3 * T // 4)
    return [int(m) for m in rng.choice(pool, size=min(n, pool.size), replace=False)]


def coherent_fraction(pa, m, r_px, fwhm):
    """Fraction of frames a source at r_px moves by less than one FWHM when
    the angle list is rolled by m."""
    d = np.abs((np.roll(pa, m) - pa + 180.0) % 360.0 - 180.0)
    return float(np.mean(r_px * np.deg2rad(d) < fwhm))


# ---------------------------------------------------------------------------
# Running PACO
# ---------------------------------------------------------------------------

def run(method, cube, pa, psf, fwhm, pxs, cpu):
    cls = PACO if method == "full" else FastPACO
    obj = cls(cube, pa, psf, fwhm=fwhm, pixscale=pxs, verbose=False)
    out = obj.run(cpu=cpu)
    return np.asarray(out[0] if isinstance(out, tuple) else out), obj


def run_vip_code(cube, pa, psf, fwhm, cpu):
    from e10_full_frame import run_vip
    return np.asarray(run_vip(cube, pa, psf, fwhm, cpu))


def make_statistics(halfcov, fullsum):
    """Patch statistics with VIP's deviations optionally switched on."""
    def statistics(patch):
        if patch is None or np.all(np.isnan(patch)):
            return None, None
        T, K = patch.shape
        m = patch.mean(axis=0)
        d = patch - m
        S = d.T @ d / T
        if halfcov:
            S = 0.5 * S
        tr2, tr, dsq = np.trace(S @ S), np.trace(S), np.sum(np.diag(S) ** 2)
        num = tr2 + tr ** 2 - 2.0 * (np.sum(S ** 2) if fullsum else dsq)
        den = (T + 1.0) * (tr2 - dsq)
        rho = 1.0 if den == 0 else min(max(num / den, 0.0), 1.0)
        C = (1.0 - rho) * S + rho * np.diag(np.diag(S))
        try:
            return m, np.linalg.inv(C)
        except np.linalg.LinAlgError:
            return m, np.linalg.pinv(C)
    return statistics


VARIANTS = {"published": None,
            "halfcov": make_statistics(True, False),
            "fullsum": make_statistics(False, True),
            "vip": make_statistics(True, True)}


def set_variant(name):
    """Swap the statistics both implementations call (before their worker
    pools fork, so the workers inherit it)."""
    fn = compute_statistics_at_pixel if VARIANTS[name] is None else VARIANTS[name]
    fast_mod.compute_statistics_at_pixel = fn
    full_mod.compute_statistics_at_pixel = fn


# ---------------------------------------------------------------------------
# null
# ---------------------------------------------------------------------------

def null_row(snr, edges, fwhm):
    r = radius(snr.shape)
    mask = (r >= edges[0]) & (r < edges[-1]) & np.isfinite(snr)
    v = snr[mask]
    widths = annulus_widths(snr, edges)
    z = normalize(snr, edges, widths)
    vz = z[np.isfinite(z)]
    row = {"std": float(v.std(ddof=1)), "robust_std": robust_std(v),
           "mean": float(v.mean()), "n_pix": int(v.size),
           "n_res": round(float(v.size / (np.pi * (fwhm / 2) ** 2)), 1),
           "groups_gt5_raw": groups_above(np.where(mask, snr, np.nan), 5),
           "groups_gt5_norm": groups_above(z, 5)}
    for k in (3, 4, 5):
        row[f"frac_gt{k}_raw"] = float(np.mean(np.abs(v) > k))
        row[f"frac_gt{k}_norm"] = float(np.mean(np.abs(vz) > k))
    row["width_inner"] = widths[0]
    row["width_outer"] = widths[-1]
    row["widths"] = " ".join(f"{w:.3f}" for w in widths)
    return row


def poisson_ci(k, conf=0.95):
    a = (1 - conf) / 2
    lo = stats.chi2.ppf(a, 2 * k) / 2 if k > 0 else 0.0
    return lo, stats.chi2.ppf(1 - a, 2 * k + 2) / 2


def cmd_null(args):
    rng = np.random.default_rng(args.seed)
    summary = []
    for name in args.datasets:
        cube, pa, psf, fwhm, pxs, _ = load(name, args)
        T, H, W = cube.shape
        K = patch_area(fwhm)
        edges = annuli((H, W), fwhm)
        out = os.path.join(OUT, name)
        os.makedirs(out, exist_ok=True)
        print(f"\n=== {name}: T={T}, {H}x{W}, FWHM={fwhm:.2f} px, K={K}, T/K={T / K:.2f}, "
              f"rotation {np.ptp(pa):.1f} deg", flush=True)
        ms = offsets(T, max(args.realizations, args.full), rng)
        rows = []
        for method, n in (("fast", args.realizations), ("full", args.full)):
            if method not in args.methods or n == 0:
                continue
            for i, m in enumerate(ms[:n]):
                t0 = time.perf_counter()
                snr, _ = run(method, cube, np.roll(pa, m), psf, fwhm, pxs, args.cpu)
                row = {"dataset": name, "method": method, "m": m, "T": T, "K": K,
                       "T_over_K": round(T / K, 2),
                       "coherent_inner": coherent_fraction(pa, m, edges[0], fwhm),
                       "coherent_outer": coherent_fraction(pa, m, edges[-1], fwhm),
                       "time_s": round(time.perf_counter() - t0, 1)}
                row.update(null_row(snr, edges, fwhm))
                rows.append(row)
                print(f"  {method} [{i + 1}/{n}] m={m:3d} robust={row['robust_std']:.3f} "
                      f"inner={row['width_inner']:.2f} outer={row['width_outer']:.2f} "
                      f">5:{row['groups_gt5_norm']} coherent={row['coherent_inner']:.2f}/"
                      f"{row['coherent_outer']:.2f} ({row['time_s']} s)", flush=True)
                write_csv(os.path.join(out, f"null_maps_{method}.csv"),
                          [x for x in rows if x["method"] == method])
        for method in ("fast", "full"):
            sel = [r for r in rows if r["method"] == method]
            if not sel:
                continue
            k = sum(r["groups_gt5_norm"] for r in sel)
            nres = sum(r["n_res"] for r in sel)
            lo, hi = poisson_ci(k)
            s = {"dataset": name, "method": method, "n_maps": len(sel), "T": T, "K": K,
                 "T_over_K": round(T / K, 2), "rotation_deg": round(float(np.ptp(pa)), 1)}
            for key in ("robust_std", "std", "width_inner", "width_outer",
                        "coherent_inner", "coherent_outer"):
                p16, p50, p84 = np.nanpercentile([r[key] for r in sel], [16, 50, 84])
                s[key] = round(p50, 4)
                s[key + "_p16_p84"] = f"{p16:.3f}-{p84:.3f}"
            for kk in (3, 4, 5):
                s[f"frac_gt{kk}_norm"] = float(np.mean([r[f"frac_gt{kk}_norm"] for r in sel]))
                s[f"gauss_gt{kk}"] = GAUSS_TAIL[kk]
            s.update({"groups_gt5": k, "res_elements": round(nres),
                      "rate_per_res": k / nres, "rate_ci95_lo": lo / nres,
                      "rate_ci95_hi": hi / nres})
            summary.append(s)
            print(f"  {method}: robust width {s['robust_std']:.3f} [{s['robust_std_p16_p84']}], "
                  f"inner {s['width_inner']:.2f}, outer {s['width_outer']:.2f}; "
                  f">5 groups {k} in {nres:.0f} res. el. -> {k / nres:.2e} "
                  f"[{lo / nres:.1e}, {hi / nres:.1e}]", flush=True)
    # Merge with earlier runs: replace only the (dataset, method) rows redone.
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "null_summary.csv")
    done = {(x["dataset"], x["method"]) for x in summary}
    if os.path.exists(path):
        import csv
        with open(path) as f:
            summary = [x for x in csv.DictReader(f)
                       if (x["dataset"], x["method"]) not in done] + summary
    write_csv(path, summary)


# ---------------------------------------------------------------------------
# ablation
# ---------------------------------------------------------------------------

def cmd_ablation(args):
    rng = np.random.default_rng(args.seed)
    for name in args.datasets:
        cube, pa, psf, fwhm, pxs, sources = load(name, args)
        T, H, W = cube.shape
        edges = annuli((H, W), fwhm)
        r = radius((H, W))
        out = os.path.join(OUT, name)
        os.makedirs(out, exist_ok=True)
        print(f"\n=== ablation {name}: T={T}, {H}x{W}, K={patch_area(fwhm)}", flush=True)

        variants = list(args.variants) + (["vipcode"] if args.vip_code else [])
        runs = [(v, m) for v in variants for m in (["fast"] if v == "vipcode" else args.methods)]

        def snr_of(v, m, c, ang):
            if v == "vipcode":
                return run_vip_code(c, ang, psf, fwhm, args.cpu), None
            set_variant(v)
            try:
                return run(m, c, ang, psf, fwhm, pxs, args.cpu)
            finally:
                set_variant("published")

        # Reference run: sigma for the injections, sources for HR 8799.
        base, obj = snr_of("published", "fast", cube, pa)
        sigma, psf_inj = np.asarray(obj.std), obj.normalised_psf
        if sources is None:
            sources = hr8799_sources(base, fwhm, pxs) if name == "hr8799" else []

        # 1. Null width per variant, same roll offsets for all.
        ms = offsets(T, args.null, rng)
        calib, null_rows = {}, []
        for v, m in runs:
            widths = []
            for off in ms:
                snr, _ = snr_of(v, m, cube, np.roll(pa, off))
                w = annulus_widths(snr, edges)
                widths.append(w)
                rr = (r >= edges[0]) & (r < edges[-1])
                null_rows.append({"variant": v, "method": m, "m": off,
                                  "robust_std": robust_std(snr[rr]),
                                  "width_inner": w[0], "width_outer": w[-1]})
            calib[(v, m)] = np.nanmedian(np.array(widths, float), axis=0)
            print(f"  null {v:9s} {m}: width {np.nanmedian([x['robust_std'] for x in null_rows if x['variant'] == v and x['method'] == m]):.3f}"
                  f" inner {calib[(v, m)][0]:.2f} outer {calib[(v, m)][-1]:.2f}", flush=True)
            write_csv(os.path.join(out, "ablation_null.csv"), null_rows)

        def calibrated_peak(snr, v, m, x, y):
            rad = np.hypot(y - H // 2, x - W // 2)
            i = np.searchsorted(edges, rad) - 1
            if not 0 <= i < len(calib[(v, m)]):
                return float("nan"), float("nan")
            p = peak_near(snr, x, y)
            return p, p / calib[(v, m)][i]

        # 2. Real companions, on the original cube.
        src_rows = []
        for v, m in runs:
            snr, _ = snr_of(v, m, cube, pa)
            for j, (x, y) in enumerate(sources):
                raw, cal = calibrated_peak(snr, v, m, x, y)
                src_rows.append({"variant": v, "method": m, "source": j,
                                 "x": x, "y": y, "raw": round(raw, 3), "cal": round(cal, 3)})
        if src_rows:
            write_csv(os.path.join(out, "ablation_sources.csv"), src_rows)

        # 3. Injected companions, sub-pixel positions, same cubes for all.
        ok = np.isfinite(sigma) & (r >= 3 * fwhm) & (r <= edges[-1] - 2 * fwhm)
        yy, xx = np.mgrid[:H, :W]
        for x, y in sources:
            ok &= np.hypot(xx - x, yy - y) > 4 * fwhm
        cand = np.argwhere(ok)
        inj_rows = []
        for k in range(args.cubes):
            rng.shuffle(cand)
            pos = []
            for row, col in cand:
                if all(np.hypot(row - a, col - b) > 4 * fwhm for a, b in pos):
                    pos.append((float(row + rng.uniform(-0.5, 0.5)),
                                float(col + rng.uniform(-0.5, 0.5))))
                if len(pos) == args.per_cube:
                    break
            tgt = [args.targets[(k * args.per_cube + j) % len(args.targets)]
                   for j in range(len(pos))]
            amps = [t * sigma[int(round(a)), int(round(b))] for t, (a, b) in zip(tgt, pos)]
            cube_k = inject(cube, pa, psf_inj, pos, amps)
            maps = {(v, m): snr_of(v, m, cube_k, pa)[0] for v, m in runs}
            for (row, col), t in zip(pos, tgt):
                rec = {"cube": k, "row": round(row, 2), "col": round(col, 2),
                       "radius_px": round(float(np.hypot(row - H // 2, col - W // 2)), 1),
                       "target": t}
                for (v, m), snr in maps.items():
                    raw, cal = calibrated_peak(snr, v, m, col, row)
                    rec[f"{v}_{m}_raw"] = round(raw, 3)
                    rec[f"{v}_{m}_cal"] = round(cal, 3)
                    rec[f"{v}_{m}_det"] = int(cal >= args.threshold)
                inj_rows.append(rec)
            write_csv(os.path.join(out, "ablation_inject.csv"), inj_rows)
            print(f"  injected cube {k + 1}/{args.cubes}", flush=True)

        # Summary, relative to the published estimator.
        print(f"\n=== {name}: matched-calibration comparison (relative to published) ===")
        for v, m in runs:
            ref = ("published", m if v != "vipcode" else "fast")
            w = np.nanmedian([x["robust_std"] for x in null_rows if x["variant"] == v and x["method"] == m])
            w0 = np.nanmedian([x["robust_std"] for x in null_rows
                               if x["variant"] == ref[0] and x["method"] == ref[1]])
            ratio = np.array([x[f"{v}_{m}_cal"] / x[f"{ref[0]}_{ref[1]}_cal"] for x in inj_rows
                              if x[f"{ref[0]}_{ref[1]}_cal"] > 0 and np.isfinite(x[f"{v}_{m}_cal"])])
            boot = [np.mean(rng.choice(ratio, ratio.size)) for _ in range(2000)] if ratio.size else [np.nan]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            ndet = sum(x[f"{v}_{m}_det"] for x in inj_rows)
            wl, wh = wilson(ndet, len(inj_rows))
            srcs = [x["cal"] for x in src_rows if x["variant"] == v and x["method"] == m]
            print(f"  {v:9s} {m}: null width {w:.3f} (x{w / w0:.2f}); injected cal. S/N "
                  f"x{np.mean(ratio):.3f} [{lo:.3f}, {hi:.3f}]; detected {ndet}/{len(inj_rows)} "
                  f"[{wl:.2f}, {wh:.2f}]; real sources cal. S/N "
                  + " ".join(f"{s:.1f}" for s in srcs))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--datasets", nargs="+", required=True,
                       choices=["sph1", "sph2", "sph3", "nirc2_1", "nirc2_2", "nirc2_3", "hr8799"])
        p.add_argument("--cpu", type=int, default=8)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--methods", nargs="+", default=["fast"], choices=["fast", "full"])
        p.add_argument("--cube"); p.add_argument("--angles"); p.add_argument("--psf")
        p.add_argument("--channel", type=int, default=0)
        p.add_argument("--size", type=int, default=400, help="hr8799: centered crop (px)")

    p = sub.add_parser("null"); common(p)
    p.add_argument("--realizations", type=int, default=20, help="Fast PACO maps per dataset")
    p.add_argument("--full", type=int, default=0, help="Full PACO maps per dataset")

    p = sub.add_parser("ablation"); common(p)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--vip-code", action="store_true", help="also VIP's own FastPACO (slow)")
    p.add_argument("--null", type=int, default=10, help="rolled null maps per variant")
    p.add_argument("--cubes", type=int, default=4)
    p.add_argument("--per-cube", type=int, default=6)
    p.add_argument("--targets", type=float, nargs="+", default=[4, 5, 6, 7, 8, 9])
    p.add_argument("--threshold", type=float, default=5.0)

    args = ap.parse_args()
    if args.cmd == "null" and args.full:
        args.methods = sorted(set(args.methods) | {"full"})
    {"null": cmd_null, "ablation": cmd_ablation}[args.cmd](args)


if __name__ == "__main__":
    main()
