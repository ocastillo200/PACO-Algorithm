#!/usr/bin/env python3
"""
Experiments requested by the simulated review of the paper, on the EIDC
SPHERE/IRDIS dataset (sph3). Four subcommands; results go to
challenge/outputs/review_exp/.

  null      Many companion-free realizations, with constructions that
            cannot realign real sources, for Fast PACO (and optionally Full
            PACO and VIP): global and per-annulus null standard deviation,
            tail fractions, >5 sigma groups, the (51,111) feature. Writes
            the per-annulus null std used to calibrate the other tests.
              constructions:
                reverse  angles x -1 (the paper's construction; may realign
                         real sources if the EIDC angles are already
                         reversed)
                roll     the angle list cyclically shifted by m frames,
                         m random in [T/4, 3T/4] (same rotation amplitude,
                         nothing follows a trajectory)
                permute  the angle list randomly permuted
                scale    angles x k, k in {-3, -2, 2, 3}
  inject    Injection-recovery at a matched, calibrated threshold: fake
            companions injected along the exact trajectories PACO tests
            (src.utils.get_rotated_pixel_coords), at target S/N values set
            from Full PACO's own sigma_alpha map, away from the five EIDC
            companions. Full and Fast PACO are run on the same cubes; a
            companion is detected if its peak S/N divided by the method's
            null std in that annulus reaches the threshold (default 5).
            Reports TPR per target S/N with Wilson intervals and the paired
            calibrated S/N ratio Fast/Full with a bootstrap interval.
            The target S/N is nominal (amplitude / sigma_alpha without the
            companion): a companion's own flux enters the background
            statistics PACO learns, so its measured S/N is lower (about
            0.8 of the target on sph3; Algorithm 3 corrects this bias).
            This affects both methods alike.
  sampling  Full PACO (and Fast PACO) on a candidate grid twice as fine as
            the pixel grid (oversampling 2): peak S/N of the fifth
            companion at s = 1 and s = 2.
  timing    Repeated wall-clock timings of Full and Fast PACO (median and
            interquartile range).

Usage
-----
    python3 benchmarks/review_experiments.py null --realizations 50 --cpu 8
    python3 benchmarks/review_experiments.py null --realizations 20 --full 3 --vip 1 --cpu 8
    python3 benchmarks/review_experiments.py inject --cubes 6 --per-cube 6 --cpu 8
    python3 benchmarks/review_experiments.py sampling --cpu 8
    python3 benchmarks/review_experiments.py timing --repeats 5 --cpu 8
    python3 benchmarks/review_experiments.py all --cpu 8

Run `null` before `inject`: the injection test is calibrated with the
per-annulus null std that `null` writes (null_calibration.json).
"""
import argparse
import csv
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from scipy import ndimage, stats

from challenge.make_contrast_curve_sph3 import load_sph3, compute_fwhm
from src.fast_paco import FastPACO
from src.full_paco import PACO
from src.utils import get_rotated_pixel_coords

OUT = os.path.join(ROOT, "challenge", "outputs", "review_exp")
SPH3_GT = [(112.67, 90.22), (82.70, 58.70), (41.05, 86.18), (55.40, 95.61), (129.24, 82.90)]
FIFTH = SPH3_GT[4]
FEATURE_XY = (51, 111)
ANNULI = [7.44, 15, 20, 30, 45, 60, 79]  # px; first edge = 2 FWHM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def data():
    cube, pa, psf, pxs = load_sph3()
    return cube, pa, psf, compute_fwhm(psf), pxs


def radius_map(shape):
    ny, nx = shape
    yy, xx = np.mgrid[:ny, :nx]
    return np.hypot(yy - ny // 2, xx - nx // 2), xx, yy


def analysis_mask(shape, fwhm):
    r, _, _ = radius_map(shape)
    return (r >= 2 * fwhm) & (r <= min(shape) / 2 - 1)


def run_method(method, cube, pa, psf, fwhm, pxs, cpu, oversampling=1, full_output=False):
    cls = PACO if method == "full" else FastPACO
    obj = cls(cube, pa, psf, fwhm=fwhm, pixscale=pxs, full_output=full_output)
    t0 = time.perf_counter()
    out = obj.run(cpu=cpu, oversampling=oversampling)
    return out, time.perf_counter() - t0, obj


def run_vip(cube, pa, psf, fwhm, cpu):
    from vip_hci.invprob.paco import FastPACO as VIP_FastPACO
    snr, _ = VIP_FastPACO(cube, pa, psf, fwhm=fwhm, pixscale=1.0, verbose=False).run(cpu=cpu)
    return np.asarray(snr)


def annulus_std(snr, fwhm):
    r, _, _ = radius_map(snr.shape)
    out = []
    for a, b in zip(ANNULI[:-1], ANNULI[1:]):
        v = snr[(r >= a) & (r < b) & np.isfinite(snr)]
        out.append(float(v.std(ddof=1)) if v.size > 10 else float("nan"))
    return out


def annulus_index(radius):
    for i, (a, b) in enumerate(zip(ANNULI[:-1], ANNULI[1:])):
        if a <= radius < b:
            return i
    return None


def null_stats(snr, fwhm):
    m = analysis_mask(snr.shape, fwhm) & np.isfinite(snr)
    v = snr[m]
    r, _, _ = radius_map(snr.shape)
    stds = annulus_std(snr, fwhm)
    z = np.full(snr.shape, np.nan)
    for (a, b), sd in zip(zip(ANNULI[:-1], ANNULI[1:]), stds):
        sel = (r >= a) & (r < b) & m
        z[sel] = snr[sel] / sd
    vz = z[np.isfinite(z)]
    lab, n = ndimage.label(np.where(m, np.nan_to_num(snr), 0) > 5)
    sizes = np.bincount(lab.ravel())[1:]
    fx, fy = FEATURE_XY
    row = {"std": v.std(ddof=1),
           "robust_std": 1.4826 * np.median(np.abs(v - np.median(v))),
           "mean": v.mean(), "excess_kurt": stats.kurtosis(v),
           "frac_gt3": np.mean(np.abs(v) > 3),
           "frac_gt3_annulus_norm": np.mean(np.abs(vz) > 3),
           "excess_kurt_annulus_norm": stats.kurtosis(vz),
           "groups_gt5": int(n), "groups_gt5_2px": int((sizes >= 2).sum()),
           "max": float(np.nanmax(np.where(m, snr, np.nan))),
           "feature_51_111": float(np.nanmax(snr[fy - 2:fy + 3, fx - 2:fx + 3]))}
    for i, sd in enumerate(stds):
        row[f"std_r{ANNULI[i]:.0f}_{ANNULI[i + 1]:.0f}"] = sd
    return row


def peak_near(snr, x, y, radius=2.0):
    _, xx, yy = radius_map(snr.shape)
    sel = (np.hypot(xx - x, yy - y) <= radius) & np.isfinite(snr)
    return float(np.max(snr[sel])) if sel.any() else float("nan")


def wilson(k, n, z=1.0):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def write_csv(path, rows):
    keys = list(rows[0])
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# null
# ---------------------------------------------------------------------------

def null_angles(pa, construction, rng):
    T = len(pa)
    if construction == "reverse":
        return -pa, "k=-1"
    if construction == "roll":
        m = int(rng.integers(T // 4, 3 * T // 4))
        return np.roll(pa, m), f"m={m}"
    if construction == "permute":
        return rng.permutation(pa), "perm"
    if construction == "scale":
        k = float(rng.choice([-3, -2, 2, 3]))
        return k * pa, f"k={k:g}"
    raise ValueError(construction)


def cmd_null(args):
    cube, pa, psf, fwhm, pxs = data()
    rng = np.random.default_rng(args.seed)
    rows = []
    plan = [("fast", c) for c in args.constructions for _ in range(
        1 if c == "reverse" else args.realizations)]
    plan += [("full", c) for c in args.constructions for _ in range(
        min(args.full, 1) if c == "reverse" else args.full)]
    plan += [("vip", c) for c in args.constructions for _ in range(
        min(args.vip, 1) if c == "reverse" else args.vip)]
    for i, (method, c) in enumerate(plan):
        ang, tag = null_angles(pa, c, rng)
        t0 = time.perf_counter()
        if method == "vip":
            snr = run_vip(cube, ang, psf, fwhm, args.cpu)
        else:
            snr, _, _ = run_method(method, cube, ang, psf, fwhm, pxs, args.cpu)
        row = {"method": method, "construction": c, "param": tag,
               "time_s": round(time.perf_counter() - t0, 1)}
        row.update({k: (round(v, 5) if isinstance(v, float) else v)
                    for k, v in null_stats(np.asarray(snr), fwhm).items()})
        rows.append(row)
        print(f"[{i + 1}/{len(plan)}] {method:4s} {c:8s} {tag:8s} std={row['std']:.3f} "
              f"inner={row['std_r7_15']:.2f} >5:{row['groups_gt5_2px']} "
              f"feature={row['feature_51_111']:.2f} ({row['time_s']} s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    write_csv(os.path.join(OUT, "null_realizations.csv"), rows)

    # Summary per (method, construction) and calibration for `inject`.
    print("\n=== Null summary (median [16th, 84th percentile]) ===")
    keys = ["std", "frac_gt3", "frac_gt3_annulus_norm", "groups_gt5_2px", "max",
            "feature_51_111"] + [k for k in rows[0] if k.startswith("std_r")]
    calib = {}
    for method in ("fast", "full", "vip"):
        for c in args.constructions:
            sel = [r for r in rows if r["method"] == method and r["construction"] == c]
            if not sel:
                continue
            parts = []
            for k in keys:
                v = np.array([r[k] for r in sel], float)
                lo, med, hi = np.nanpercentile(v, [16, 50, 84])
                parts.append(f"{k}={med:.3g} [{lo:.3g},{hi:.3g}]")
            print(f"{method:4s} {c:8s} n={len(sel)}: " + "  ".join(parts))
            # Calibrate on `roll` only: it keeps the rotation amplitude and the
            # temporal order, and cannot realign sources. `permute` and
            # `scale` whiten the speckle noise (narrower null), and `reverse`
            # may realign real sources.
            if c == "roll":
                calib.setdefault(method, []).extend(
                    [[r[k] for k in rows[0] if k.startswith("std_r")] for r in sel])
    calib = {m: {"annuli_px": ANNULI, "std": np.nanmedian(np.array(v, float), axis=0).tolist(),
                 "n_maps": len(v), "construction": "roll"} for m, v in calib.items()}
    with open(os.path.join(OUT, "null_calibration.json"), "w") as f:
        json.dump(calib, f, indent=1)
    print(f"Wrote {os.path.join(OUT, 'null_calibration.json')} (per-annulus null std, "
          f"median over `roll` realizations)")


# ---------------------------------------------------------------------------
# inject
# ---------------------------------------------------------------------------

def inject(cube, pa, psf, positions, amplitudes):
    """Add amplitude x PSF along the trajectory PACO tests for each (row, col)."""
    T, H, W = cube.shape
    gx, gy = np.meshgrid(np.arange(-W / 2, W / 2), np.arange(-H / 2, H / 2))
    out = cube.copy()
    ph, pw = psf.shape
    for (row, col), amp in zip(positions, amplitudes):
        traj = get_rotated_pixel_coords(gx, gy, (row, col), pa)  # [:,0]=col, [:,1]=row
        for l in range(T):
            c, r = traj[l, 0], traj[l, 1]
            ri, ci = int(np.floor(r)), int(np.floor(c))
            stamp = ndimage.shift(psf, (r - ri, c - ci), order=3, mode="constant")
            r0, c0 = ri - ph // 2, ci - pw // 2
            rs, cs = max(r0, 0), max(c0, 0)
            re, ce = min(r0 + ph, H), min(c0 + pw, W)
            if rs < re and cs < ce:
                out[l, rs:re, cs:ce] += amp * stamp[rs - r0:re - r0, cs - c0:ce - c0]
    return out


def pick_positions(sigma_map, fwhm, n, rng, taken):
    H, W = sigma_map.shape
    r, _, _ = radius_map(sigma_map.shape)
    ok = np.isfinite(sigma_map) & (r >= 3 * fwhm) & (r <= ANNULI[-1] - 2)
    for x, y in SPH3_GT:
        yy, xx = np.mgrid[:H, :W]
        ok &= np.hypot(xx - x, yy - y) > 3 * fwhm
    cand = np.argwhere(ok)
    rng.shuffle(cand)
    chosen = []
    for row, col in cand:
        if all(np.hypot(row - a, col - b) > 4 * fwhm for a, b in chosen + taken):
            chosen.append((int(row), int(col)))
        if len(chosen) == n:
            break
    return chosen


def cmd_inject(args):
    cube, pa, psf, fwhm, pxs = data()
    calib_path = os.path.join(OUT, "null_calibration.json")
    if not os.path.exists(calib_path):
        sys.exit("Run the `null` subcommand first (it writes null_calibration.json).")
    calib = json.load(open(calib_path))
    methods = [m for m in args.methods if m in calib]
    if len(methods) < len(args.methods):
        print(f"Note: no null calibration for {set(args.methods) - set(methods)}; skipped. "
              f"Run `null --full N` to calibrate Full PACO.")
    rng = np.random.default_rng(args.seed)

    sig_path = os.path.join(OUT, "sigma_alpha_full.npy")
    if os.path.exists(sig_path):
        sigma = np.load(sig_path)
    else:
        print("Baseline Full PACO run (no injection) for sigma_alpha ...", flush=True)
        _, _, obj = run_method("full", cube, pa, psf, fwhm, pxs, args.cpu)
        sigma = np.asarray(obj.std)
        os.makedirs(OUT, exist_ok=True)
        np.save(sig_path, sigma)

    targets = args.targets
    rows = []
    for k in range(args.cubes):
        pos = pick_positions(sigma, fwhm, args.per_cube, rng, [])
        tgt = [targets[(k * args.per_cube + j) % len(targets)] for j in range(len(pos))]
        amps = [t * sigma[r, c] for t, (r, c) in zip(tgt, pos)]
        cube_k = inject(cube, pa, psf, pos, amps)
        maps = {}
        for m in methods:
            snr, dt, _ = run_method(m, cube_k, pa, psf, fwhm, pxs, args.cpu)
            maps[m] = np.asarray(snr)
            print(f"cube {k + 1}/{args.cubes} {m}: {dt:.1f} s", flush=True)
        for (r, c), t in zip(pos, tgt):
            rad = float(np.hypot(r - cube.shape[1] // 2, c - cube.shape[2] // 2))
            ai = annulus_index(rad)
            row = {"cube": k, "row": r, "col": c, "radius_px": round(rad, 1), "target_snr": t}
            for m in methods:
                peak = peak_near(maps[m], c, r)
                cal = peak / calib[m]["std"][ai]
                row[f"{m}_peak"] = round(peak, 3)
                row[f"{m}_calibrated"] = round(cal, 3)
                row[f"{m}_detected"] = int(cal >= args.threshold)
            rows.append(row)
    os.makedirs(OUT, exist_ok=True)
    write_csv(os.path.join(OUT, "injection_recovery.csv"), rows)

    print(f"\n=== Injection-recovery: TPR at calibrated threshold {args.threshold} "
          f"(1-sigma Wilson interval) ===")
    for t in sorted(set(r["target_snr"] for r in rows)):
        sel = [r for r in rows if r["target_snr"] == t]
        parts = []
        for m in methods:
            kdet = sum(r[f"{m}_detected"] for r in sel)
            lo, hi = wilson(kdet, len(sel))
            parts.append(f"{m}: {kdet}/{len(sel)} [{lo:.2f},{hi:.2f}]")
        print(f"target S/N {t:g}: " + "   ".join(parts))
    if {"full", "fast"} <= set(methods):
        ratio = np.array([r["fast_calibrated"] / r["full_calibrated"] for r in rows
                          if r["full_calibrated"] > 0])
        boot = [np.mean(rng.choice(ratio, ratio.size)) for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        only_full = sum(r["full_detected"] and not r["fast_detected"] for r in rows)
        only_fast = sum(r["fast_detected"] and not r["full_detected"] for r in rows)
        print(f"Calibrated S/N Fast/Full: mean {ratio.mean():.3f} (95% bootstrap "
              f"[{lo:.3f},{hi:.3f}], n={ratio.size}); detected by Full only: {only_full}, "
              f"by Fast only: {only_fast}")


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def cmd_sampling(args):
    cube, pa, psf, fwhm, pxs = data()
    x, y = FIFTH
    rows = []
    for m in args.methods:
        for s in (1, 2):
            snr, dt, _ = run_method(m, cube, pa, psf, fwhm, pxs, args.cpu, oversampling=s)
            snr = np.asarray(snr)
            peak = peak_near(snr, x * s, y * s, radius=2.0 * s)
            rows.append({"method": m, "oversampling": s, "peak_snr_fifth": round(peak, 3),
                         "time_s": round(dt, 1)})
            print(f"{m} s={s}: fifth companion peak S/N {peak:.2f} ({dt:.1f} s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    write_csv(os.path.join(OUT, "sampling_fifth_companion.csv"), rows)


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

def cmd_timing(args):
    cube, pa, psf, fwhm, pxs = data()
    rows = []
    for m in args.methods:
        ts = []
        for i in range(args.repeats):
            _, dt, _ = run_method(m, cube, pa, psf, fwhm, pxs, args.cpu)
            ts.append(dt)
            rows.append({"method": m, "cpu": args.cpu, "repeat": i, "time_s": round(dt, 3)})
        q1, med, q3 = np.percentile(ts, [25, 50, 75])
        print(f"{m}: median {med:.2f} s, IQR [{q1:.2f}, {q3:.2f}] over {len(ts)} runs "
              f"(cpu={args.cpu})", flush=True)
    os.makedirs(OUT, exist_ok=True)
    write_csv(os.path.join(OUT, f"timing_cpu{args.cpu}.csv"), rows)
    meds = {m: np.median([r["time_s"] for r in rows if r["method"] == m]) for m in args.methods}
    if {"full", "fast"} <= set(meds):
        print(f"Full/Fast median ratio: {meds['full'] / meds['fast']:.1f}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--cpu", type=int, default=4)
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("null"); common(p)
    p.add_argument("--realizations", type=int, default=50,
                   help="Fast PACO realizations per non-reversing construction.")
    p.add_argument("--full", type=int, default=0, help="Full PACO realizations per construction.")
    p.add_argument("--vip", type=int, default=0, help="VIP realizations per construction.")
    p.add_argument("--constructions", nargs="+", default=["reverse", "roll", "permute", "scale"],
                   choices=["reverse", "roll", "permute", "scale"])

    p = sub.add_parser("inject"); common(p)
    p.add_argument("--cubes", type=int, default=6)
    p.add_argument("--per-cube", type=int, default=6)
    p.add_argument("--targets", type=float, nargs="+", default=[4, 5, 6, 7, 8, 9],
                   help="Nominal target S/N values (amplitude / Full PACO's sigma_alpha).")
    p.add_argument("--threshold", type=float, default=5.0)
    p.add_argument("--methods", nargs="+", default=["full", "fast"], choices=["full", "fast"])

    p = sub.add_parser("sampling"); common(p)
    p.add_argument("--methods", nargs="+", default=["full", "fast"], choices=["full", "fast"])

    p = sub.add_parser("timing"); common(p)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--methods", nargs="+", default=["full", "fast"], choices=["full", "fast"])

    p = sub.add_parser("all"); common(p)

    args = ap.parse_args()
    if args.cmd == "all":
        defaults = dict(realizations=50, full=5, vip=0,
                        constructions=["reverse", "roll", "permute", "scale"])
        cmd_null(argparse.Namespace(**vars(args), **defaults))
        cmd_inject(argparse.Namespace(**vars(args), cubes=6, per_cube=6, targets=[4, 5, 6, 7, 8, 9],
                                      threshold=5.0, methods=["full", "fast"]))
        cmd_sampling(argparse.Namespace(**vars(args), methods=["full", "fast"]))
        cmd_timing(argparse.Namespace(**vars(args), repeats=5, methods=["full", "fast"]))
        return
    {"null": cmd_null, "inject": cmd_inject, "sampling": cmd_sampling,
     "timing": cmd_timing}[args.cmd](args)


if __name__ == "__main__":
    main()
