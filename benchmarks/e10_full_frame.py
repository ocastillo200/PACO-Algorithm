#!/usr/bin/env python3
"""
E10: full-frame test on a real ADI sequence of typical SPHERE/IRDIS size
(1024 x 1024 x T), far larger than the 160 x 160 EIDC field used elsewhere.
Results go to challenge/outputs/e10_full_frame/.

Two subcommands:

  scale   Wall time and aggregate memory of Full and Fast PACO (and, opt-in,
          VIP's FastPACO) on centered crops of increasing size (default
          256, 512, 1024 px), full native candidate grid. Tests that time
          grows with the field while working memory stays within the budget.
          VIP is skipped for a crop when its estimated peak memory (twice
          its whole-field statistics, as measured in E9) exceeds the
          available memory, unless --force-vip.
  inject  Detection on the full frame. Fast PACO's null is calibrated per
          annulus from --null companion-free realizations (angles rolled
          by m in [T/4, 3T/4] frames). Fake companions are injected along
          the exact trajectories PACO tests, at nominal S/N set from Fast
          PACO's sigma_alpha, at separations from 3 FWHM to the edge of the
          frame, away from real sources (calibrated S/N > threshold in the
          original map, plus --exclude). Reports detections per nominal S/N
          and separation, and false alarms (groups above threshold) in the
          null and injected maps. --full also runs Full PACO on each
          injected cube (about 40 times the EIDC run time).
  real    A system with known planets (default HR 8799 b, c, d, e): Full,
          Fast and VIP's FastPACO on the full frame, with time and memory;
          peak S/N at each planet, raw and calibrated by each map's own
          per-annulus robust std (sources masked), which puts VIP's wider
          S/N scale on the same footing; every other group above the
          threshold; a figure of the three calibrated maps. Maps are saved
          and reused, so a rerun only recomputes what is missing.

Inputs are FITS files: --cube (T,H,W) or (C,T,H,W) with --channel, --angles
in degrees (T,), --psf (h,w), (T,h,w), or (C,[T,]h,w); the star is assumed at
the frame center (H//2, W//2), as in standard SPHERE pipeline products.

Usage
-----
    python3 benchmarks/e10_full_frame.py scale --cube center_im.fits \\
        --angles rotnth.fits --psf median_unsat.fits --channel 0 \\
        --crops 256,512,1024 --cpu 32 --repeats 2 --memory-backend pss
    # add VIP (about 9 h at 1024 px for T ~ 230; needs ~125 GB):
    python3 benchmarks/e10_full_frame.py scale ... --methods full,fast,vip
    python3 benchmarks/e10_full_frame.py inject --cube ... --channel 0 \\
        --cpu 32 --null 20 --cubes 4 --per-cube 12
    python3 benchmarks/e10_full_frame.py real --cube data/HR8799/center_im.fits \\
        --angles data/HR8799/rotnth.fits --psf data/HR8799/median_unsat.fits \\
        --channel 0 --cpu 32 --memory-backend pss
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
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import numpy as np
import psutil
from astropy.io import fits
from scipy import ndimage

from _common import make_memory_monitor, mem_columns, mem_str, print_system_info
from challenge.make_contrast_curve_sph3 import compute_fwhm
from review_experiments import inject, null_angles, peak_near, wilson, write_csv
from src.fast_paco import FastPACO
from src.full_paco import PACO

OUT = os.path.join(ROOT, "challenge", "outputs", "e10_full_frame")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(args):
    cube = fits.getdata(args.cube).astype(np.float64)
    if cube.ndim == 4:
        cube = cube[args.channel]
    pa = fits.getdata(args.angles).astype(np.float64).ravel()
    psf = fits.getdata(args.psf).astype(np.float64)
    if psf.ndim == 4:
        psf = psf[args.channel]
    if psf.ndim == 3 and psf.shape[0] != len(pa) and psf.shape[0] <= 4:
        psf = psf[args.channel]  # (C,h,w): one PSF per channel
    if psf.ndim == 3:
        psf = np.nanmedian(psf, axis=0)
    if args.psf_size and psf.shape[0] > args.psf_size:
        c, h = psf.shape[0] // 2, args.psf_size // 2
        psf = psf[c - h:c + h + 1, c - h:c + h + 1]
    psf = psf / psf.max()
    if len(pa) != cube.shape[0]:
        sys.exit(f"angles ({len(pa)}) do not match the cube's frames ({cube.shape[0]})")
    if args.frames:
        cube, pa = cube[:args.frames], pa[:args.frames]
    cube = np.nan_to_num(cube)
    fwhm = args.fwhm if args.fwhm else compute_fwhm(psf)
    print(f"cube {cube.shape}, rotation {np.ptp(pa):.1f} deg, psf {psf.shape}, "
          f"fwhm {fwhm:.2f} px", flush=True)
    return cube, pa, psf, fwhm


def crop(cube, size):
    T, H, W = cube.shape
    size = min(size, H, W)
    cy, cx = H // 2, W // 2
    h = size // 2
    return np.ascontiguousarray(cube[:, cy - h:cy - h + size, cx - h:cx - h + size])


def run_ours(method, cube, pa, psf, fwhm, cpu):
    cls = PACO if method == "full" else FastPACO
    obj = cls(cube, pa, psf, fwhm=fwhm, pixscale=1.0, verbose=False)
    snr = obj.run(cpu=cpu)
    snr = snr[0] if isinstance(snr, tuple) else snr
    return np.asarray(snr), obj


def vip_pixscale(fwhm):
    # Same matched-K choice as bench_vs_vippaco.py: VIP's mask radius
    # int(fwhm / pixscale) equals ours, int(2 * fwhm) // 2.
    return fwhm / (int(2 * fwhm) // 2)


def vip_peak_estimate_bytes(T, H, W, K):
    # Whole-field patches, means and inverse covariances, twice (E9).
    return 2 * 8 * H * W * (T * K + K * K + K)


def run_vip(cube, pa, psf, fwhm, cpu):
    from vip_hci.invprob.paco import FastPACO as VIP_FastPACO
    obj = VIP_FastPACO(cube, pa, psf, fwhm=fwhm, pixscale=vip_pixscale(fwhm), verbose=False)
    snr, _ = obj.run(cpu=cpu)  # full native grid
    return np.asarray(snr)


def vip_fits(T, H, W, fwhm):
    K = int(np.sum(np.hypot(*np.mgrid[-10:11, -10:11]) <= int(2 * fwhm) // 2))
    need = vip_peak_estimate_bytes(T, H, W, K)
    return need, need <= 0.9 * psutil.virtual_memory().available


# ---------------------------------------------------------------------------
# scale
# ---------------------------------------------------------------------------

def cmd_scale(args):
    print_system_info()
    cube, pa, psf, fwhm = load(args)
    rows = []
    for size in args.crops:
        c = crop(cube, size)
        T, H, W = c.shape
        for method in args.methods:
            if method == "vip":
                need, ok = vip_fits(T, H, W, fwhm)
                if not ok and not args.force_vip:
                    print(f"  {H}px vip: skipped, needs ~{need / 1e9:.0f} GB, "
                          f"{psutil.virtual_memory().available / 1e9:.0f} GB available",
                          flush=True)
                    rows.append({"crop": H, "T": T, "method": method, "repeat": 0,
                                 "time_s": "", "mem_real_mb": "",
                                 "note": f"skipped: est {need / 1e9:.0f} GB"})
                    continue
            for rep in range(args.repeats if method != "vip" else 1):
                with make_memory_monitor(args) as mon:
                    t0 = time.perf_counter()
                    if method == "vip":
                        run_vip(c, pa, psf, fwhm, args.cpu)
                    else:
                        run_ours(method, c, pa, psf, fwhm, args.cpu)
                    dt = time.perf_counter() - t0
                print(f"  {H}px {method:4s} repeat {rep + 1}: {dt:9.1f} s  {mem_str(mon)}",
                      flush=True)
                rows.append({"crop": H, "T": T, "method": method, "repeat": rep,
                             "time_s": f"{dt:.2f}", **mem_columns(mon), "note": ""})
                os.makedirs(OUT, exist_ok=True)
                write_csv(os.path.join(OUT, args.csv), rows)


# ---------------------------------------------------------------------------
# inject
# ---------------------------------------------------------------------------

def annuli_edges(fwhm, size):
    # Inner edge 2 FWHM as on sph3, then annuli doubling in width.
    inner, outer = 2 * fwhm, size / 2 - 1
    return [inner] + [x for x in (15, 30, 60, 120, 240, 480) if inner < x < outer] + [outer]


def radius(shape):
    ny, nx = shape
    yy, xx = np.mgrid[:ny, :nx]
    return np.hypot(yy - ny // 2, xx - nx // 2)


def calibrate(snr, std_by_annulus, edges):
    r = radius(snr.shape)
    z = np.full(snr.shape, np.nan)
    for (a, b), sd in zip(zip(edges[:-1], edges[1:]), std_by_annulus):
        sel = (r >= a) & (r < b) & np.isfinite(snr)
        z[sel] = snr[sel] / sd
    return z


def groups_above(z, thr, min_px=2):
    lab, n = ndimage.label(np.nan_to_num(z) > thr)
    if n == 0:
        return []
    sizes = ndimage.sum(np.ones_like(z), lab, range(1, n + 1))
    peaks = ndimage.maximum_position(np.nan_to_num(z), lab, range(1, n + 1))
    return [(int(p[0]), int(p[1]), int(s)) for p, s in zip(peaks, sizes) if s >= min_px]


def cmd_inject(args):
    print_system_info()
    cube, pa, psf, fwhm = load(args)
    cube = crop(cube, args.size)
    T, H, W = cube.shape
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    r = radius((H, W))

    # Null calibration (Fast PACO, rolled angles).
    edges = annuli_edges(fwhm, H)
    null_rows, stds = [], []
    for k in range(args.null):
        ang, tag = null_angles(pa, "roll", rng)
        t0 = time.perf_counter()
        snr, _ = run_ours("fast", cube, ang, psf, fwhm, args.cpu)
        dt = time.perf_counter() - t0
        per = []
        for a, b in zip(edges[:-1], edges[1:]):
            v = snr[(r >= a) & (r < b) & np.isfinite(snr)]
            per.append(float(v.std(ddof=1)))
        stds.append(per)
        v = snr[(r >= edges[0]) & (r <= edges[-1]) & np.isfinite(snr)]
        null_rows.append({"realization": k, "construction": tag, "time_s": round(dt, 2),
                          "std": round(float(v.std(ddof=1)), 4),
                          "frac_gt3": float(np.mean(np.abs(v) > 3)),
                          "n_pix": int(v.size),
                          **{f"std_{a:.0f}_{b:.0f}": round(s, 4)
                             for (a, b), s in zip(zip(edges[:-1], edges[1:]), per)}})
        print(f"  null {k + 1}/{args.null} ({tag}): std {null_rows[-1]['std']:.3f}, "
              f"{dt:.1f} s", flush=True)
    calib = np.median(np.array(stds), axis=0).tolist()
    json.dump({"edges": edges, "std": calib, "fwhm": fwhm, "shape": [T, H, W]},
              open(os.path.join(OUT, "null_calibration.json"), "w"), indent=1)

    # False alarms on further realizations, with the calibration fixed (so
    # the maps that set the threshold are not the ones it is tested on).
    fa_rows = []
    for k in range(args.null_fa):
        ang, tag = null_angles(pa, "roll", rng)
        snr, _ = run_ours("fast", cube, ang, psf, fwhm, args.cpu)
        g = groups_above(calibrate(snr, calib, edges), args.threshold)
        fa_rows.append({"realization": k, "construction": tag, "groups": len(g),
                        "where": ";".join(f"{y},{x}({n}px)" for y, x, n in g)})
        print(f"  false-alarm map {k + 1}/{args.null_fa}: {len(g)} groups", flush=True)
    write_csv(os.path.join(OUT, "null_realizations.csv"), null_rows)
    if fa_rows:
        write_csv(os.path.join(OUT, "null_false_alarms.csv"), fa_rows)

    # Baseline: real sources and sigma_alpha.
    snr0, obj = run_ours("fast", cube, pa, psf, fwhm, args.cpu)
    sigma = np.asarray(obj.std)
    # sigma_alpha is in units of the PSF as Fast PACO normalizes it, so the
    # companions are injected with that same template.
    psf_inj = obj.normalised_psf
    z0 = calibrate(snr0, calib, edges)
    real = [(y, x) for y, x, _ in groups_above(z0, args.threshold, min_px=1)]
    real += [(int(b), int(a)) for a, b in (map(float, e.split(",")) for e in args.exclude)]
    print(f"  {len(real)} real sources / excluded positions: {real[:10]}", flush=True)
    np.save(os.path.join(OUT, "baseline_fast_snr.npy"), snr0)

    yy, xx = np.mgrid[:H, :W]
    rmin, rmax = 3 * fwhm, H / 2 - psf.shape[0] / 2 - 2
    rows = []
    for c in range(args.cubes):
        pos, chosen = [], []
        seps = np.linspace(rmin, rmax, args.per_cube)
        rng.shuffle(seps)
        for sep in seps:
            for _ in range(200):
                th = rng.uniform(0, 2 * np.pi)
                row = int(round(H // 2 + sep * np.sin(th)))
                col = int(round(W // 2 + sep * np.cos(th)))
                if not np.isfinite(sigma[row, col]):
                    continue
                if all(np.hypot(row - a, col - b) > 4 * fwhm for a, b in chosen + real):
                    chosen.append((row, col))
                    break
        targets = [args.targets[(c * args.per_cube + j) % len(args.targets)]
                   for j in range(len(chosen))]
        amps = [t * sigma[row, col] for (row, col), t in zip(chosen, targets)]
        cube_k = inject(cube, pa, psf_inj, chosen, amps)
        maps = {}
        for m in (["fast", "full"] if args.full else ["fast"]):
            t0 = time.perf_counter()
            maps[m], _ = run_ours(m, cube_k, pa, psf, fwhm, args.cpu)
            print(f"  cube {c + 1}/{args.cubes} {m}: {time.perf_counter() - t0:.1f} s", flush=True)
        for (row, col), t in zip(chosen, targets):
            rec = {"cube": c, "row": row, "col": col,
                   "sep_px": round(float(r[row, col]), 1), "target": t}
            # Fast PACO is calibrated by its own null; Full PACO's null is
            # not measured on the full frame (too costly), so its detection
            # flag uses the raw S/N.
            for m, snr in maps.items():
                rec[f"{m}_snr"] = round(peak_near(snr, col, row), 3)
                if m == "fast":
                    rec["fast_cal"] = round(peak_near(calibrate(snr, calib, edges), col, row), 3)
                    rec["fast_det"] = int(rec["fast_cal"] >= args.threshold)
                else:
                    rec["full_det_raw"] = int(rec["full_snr"] >= args.threshold)
            rows.append(rec)
        # False alarms in the injected Fast map: groups away from injected
        # and real sources.
        g = groups_above(calibrate(maps["fast"], calib, edges), args.threshold)
        fps = [p for p in g if all(np.hypot(p[0] - a, p[1] - b) > 2 * fwhm
                                   for a, b in chosen + real)]
        rows[-1]["cube_false_alarms"] = len(fps)
        write_csv(os.path.join(OUT, "injection_recovery.csv"), rows)

    print("\n=== Fast PACO detections at calibrated threshold "
          f"{args.threshold:g} ===")
    for t in sorted(set(x["target"] for x in rows)):
        sub = [x for x in rows if x["target"] == t]
        k = sum(x["fast_det"] for x in sub)
        lo, hi = wilson(k, len(sub))
        print(f"  nominal S/N {t:g}: {k}/{len(sub)}  [{lo:.2f}, {hi:.2f}]")
    print(f"  false alarms (null maps): "
          f"{sum(x['groups'] for x in fa_rows)} in {len(fa_rows)} maps of {H}x{W}")


# ---------------------------------------------------------------------------
# real
# ---------------------------------------------------------------------------

def self_calibrate(snr, fwhm, mask_xy, width):
    """Divide each annulus by its own robust std (1.4826 MAD), with circles
    of 3 FWHM around mask_xy (sources) left out of the estimate."""
    H, W = snr.shape
    r = radius((H, W))
    yy, xx = np.mgrid[:H, :W]
    keep = np.isfinite(snr)
    for x, y in mask_xy:
        keep &= np.hypot(xx - x, yy - y) > 3 * fwhm
    z = np.full(snr.shape, np.nan)
    edges = np.arange(2 * fwhm, H / 2, width)
    for a, b in zip(edges[:-1], edges[1:]):
        ring = (r >= a) & (r < b)
        v = snr[ring & keep]
        if v.size < 50:
            continue
        sd = 1.4826 * np.median(np.abs(v - np.median(v)))
        z[ring & np.isfinite(snr)] = (snr[ring & np.isfinite(snr)] - np.median(v)) / sd
    return z


def cmd_real(args):
    print_system_info()
    cube, pa, psf, fwhm = load(args)
    # SPHERE rotnth.fits stores the angles with the opposite sign to the
    # one PACO and VIP expect (see challenge/run_real_system_comparison.py);
    # used as stored, the planets smear out instead of adding up.
    if not args.keep_angle_sign:
        pa = -pa
    cube = crop(cube, args.size)
    T, H, W = cube.shape
    out = os.path.join(OUT, args.name)
    os.makedirs(out, exist_ok=True)
    planets = [(n, float(v)) for n, v in (e.split(":") for e in args.planets.split(","))]
    tol = args.tol / args.pixscale
    print(f"planets (px): {[(n, round(v / args.pixscale, 1)) for n, v in planets]}, "
          f"tolerance {tol:.1f} px", flush=True)

    # Keep the timings of methods run earlier (maps loaded from disk).
    maps, rows = {}, []
    tpath = os.path.join(out, "timing.csv")
    if os.path.exists(tpath):
        with open(tpath) as f:
            rows = list(csv.DictReader(f))
    for m in args.methods:
        c = cube
        if m == "vip":
            need, ok = vip_fits(T, H, W, fwhm)
            if not ok and not args.force_vip:
                if not args.vip_size:
                    print(f"  vip: skipped, needs ~{need / 1e9:.0f} GB; pass --vip-size 512 "
                          f"to run it on a centered crop", flush=True)
                    continue
                c = crop(cube, args.vip_size)
        path = os.path.join(out, f"{m}_snr.npy")
        if os.path.exists(path) and not args.rerun:
            maps[m] = np.load(path)
            print(f"  {m}: loaded {path}", flush=True)
            continue
        with make_memory_monitor(args) as mon:
            t0 = time.perf_counter()
            snr = run_vip(c, pa, psf, fwhm, args.cpu) if m == "vip" else \
                run_ours(m, c, pa, psf, fwhm, args.cpu)[0]
            dt = time.perf_counter() - t0
        print(f"  {m}: {c.shape[1]}px, {dt:.1f} s, {mem_str(mon)}", flush=True)
        rows = [x for x in rows if x["method"] != m]
        rows.append({"method": m, "T": T, "size": c.shape[1], "cpu": args.cpu,
                     "time_s": f"{dt:.2f}", **mem_columns(mon)})
        write_csv(os.path.join(out, "timing.csv"), rows)
        if snr.shape != (H, W):  # VIP on a crop: pad back to the frame
            pad = np.full((H, W), np.nan)
            o = (H - snr.shape[0]) // 2
            pad[o:o + snr.shape[0], o:o + snr.shape[1]] = snr
            snr = pad
        np.save(path, snr)
        maps[m] = snr

    # Locate each planet: the brightest point of the summed, normalized
    # maps in its separation annulus (methods agree on position).
    r = radius((H, W))
    yy, xx = np.mgrid[:H, :W]
    stack = np.nansum([v / np.nanstd(v) for v in maps.values()], axis=0)
    found = {}
    for n, sep in planets:
        ring = np.abs(r - sep / args.pixscale) <= tol
        i = np.nanargmax(np.where(ring, stack, -np.inf))
        found[n] = (int(xx.flat[i]), int(yy.flat[i]))
    cal = {m: self_calibrate(v, fwhm, list(found.values()), args.annulus_width)
           for m, v in maps.items()}

    det_rows = []
    for n, (x, y) in found.items():
        rec = {"planet": n, "x": x, "y": y, "sep_px": round(float(r[y, x]), 1),
               "sep_arcsec": round(float(r[y, x]) * args.pixscale, 3)}
        for m in maps:
            rec[f"{m}_raw"] = round(peak_near(maps[m], x, y), 2)
            rec[f"{m}_cal"] = round(peak_near(cal[m], x, y), 2)
        det_rows.append(rec)
    write_csv(os.path.join(out, "planets.csv"), det_rows)

    # Every other group above threshold on the calibrated maps.
    other = []
    for m, z in cal.items():
        for y, x, npx in groups_above(z, args.threshold):
            if all(np.hypot(x - a, y - b) > 2 * fwhm for a, b in found.values()):
                other.append({"method": m, "x": x, "y": y, "n_px": npx,
                              "sep_px": round(float(r[y, x]), 1),
                              "cal": round(float(z[y, x]), 2),
                              "raw": round(float(maps[m][y, x]), 2)})
    if other:
        write_csv(os.path.join(out, "other_groups.csv"), other)

    print(f"\n=== {args.name}: peak S/N at the planets, raw / self-calibrated ===")
    print(f"{'planet':>7} {'sep':>6} " + "".join(f"{m:>16}" for m in maps))
    for d in det_rows:
        print(f"{d['planet']:>7} {d['sep_arcsec']:>5.2f}\" " + "".join(
            f"{d[m + '_raw']:>7.1f} / {d[m + '_cal']:<6.1f}" for m in maps))
    for m in maps:
        print(f"  {m}: {sum(1 for o in other if o['method'] == m)} other groups "
              f">= {args.threshold:g} (calibrated, >= 2 px)")
    print("  (other groups may be real background sources; see other_groups.csv)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    half = min(int(max(v for _, v in planets) / args.pixscale + 25), H // 2)
    sl = slice(H // 2 - half, H // 2 + half)
    fig, axs = plt.subplots(1, len(cal), figsize=(3.3 * len(cal), 3.4))
    for ax, (m, z) in zip(np.atleast_1d(axs), cal.items()):
        im = ax.imshow(z[sl, sl], origin="lower", cmap="inferno", vmin=-3, vmax=10)
        for n, (x, y) in found.items():
            ax.add_patch(plt.Circle((x - H // 2 + half, y - H // 2 + half), 3 * fwhm,
                                    fill=False, color="cyan", lw=0.8))
            ax.text(x - H // 2 + half + 3 * fwhm, y - H // 2 + half, n, color="cyan",
                    fontsize=8, clip_on=True)
        ax.set_title({"full": "Full PACO (ours)", "fast": "Fast PACO (ours)",
                      "vip": "VIP FastPACO"}[m], fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axs, shrink=0.8, label="calibrated S/N")
    fig.savefig(os.path.join(out, f"{args.name}_maps.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, f"{args.name}_maps.png"), dpi=150, bbox_inches="tight")
    print(f"Wrote {out}/{args.name}_maps.pdf")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("scale", "inject", "real"):
        s = sub.add_parser(name)
        s.add_argument("--cube", required=True)
        s.add_argument("--angles", required=True, help="parallactic angles, degrees")
        s.add_argument("--psf", required=True)
        s.add_argument("--channel", type=int, default=0, help="for (C,T,H,W) cubes")
        s.add_argument("--fwhm", type=float, default=None, help="default: fit to the PSF")
        s.add_argument("--psf-size", type=int, default=41,
                       help="crop the PSF to this odd size (default 41, as sph3)")
        s.add_argument("--frames", type=int, default=None, help="use only the first T frames")
        s.add_argument("--cpu", type=int, default=os.cpu_count() // 2)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--memory-backend", choices=["psutil", "pss", "cgroup"],
                       default="psutil")
    s = sub.choices["scale"]
    s.add_argument("--crops", type=lambda v: [int(x) for x in v.split(",")],
                   default=[256, 512, 1024])
    s.add_argument("--methods", type=lambda v: v.split(","), default=["full", "fast"])
    s.add_argument("--repeats", type=int, default=1)
    s.add_argument("--force-vip", action="store_true")
    s.add_argument("--csv", default="scale.csv")
    s = sub.choices["inject"]
    s.add_argument("--size", type=int, default=1024, help="centered crop (default full frame)")
    s.add_argument("--null", type=int, default=20, help="calibration realizations")
    s.add_argument("--null-fa", type=int, default=10,
                   help="further realizations to count false alarms")
    s.add_argument("--cubes", type=int, default=4)
    s.add_argument("--per-cube", type=int, default=12)
    s.add_argument("--targets", type=lambda v: [float(x) for x in v.split(",")],
                   default=[4, 5, 6, 7, 8, 9])
    s.add_argument("--threshold", type=float, default=5.0)
    s.add_argument("--exclude", nargs="*", default=[], help="x,y of known sources")
    s.add_argument("--full", action="store_true", help="also run Full PACO per cube")
    s = sub.choices["real"]
    s.add_argument("--name", default="HR8799")
    s.add_argument("--size", type=int, default=1024, help="centered crop (default full frame)")
    s.add_argument("--methods", type=lambda v: v.split(","), default=["full", "fast", "vip"])
    s.add_argument("--planets", default="b:1.72,c:0.95,d:0.68,e:0.41",
                   help="name:separation[arcsec] list (default HR 8799, approximate)")
    s.add_argument("--tol", type=float, default=0.08, help="separation tolerance, arcsec")
    s.add_argument("--pixscale", type=float, default=0.01225, help="arcsec/px (IRDIS)")
    s.add_argument("--annulus-width", type=float, default=10.0,
                   help="px, for the per-annulus self-calibration")
    s.add_argument("--threshold", type=float, default=5.0)
    s.add_argument("--keep-angle-sign", action="store_true",
                   help="use the angles as stored (default: negated, the "
                        "rotnth.fits convention)")
    s.add_argument("--vip-size", type=int, default=None,
                   help="run VIP on this centered crop if the full frame does not fit")
    s.add_argument("--force-vip", action="store_true")
    s.add_argument("--rerun", action="store_true", help="ignore saved maps")
    args = p.parse_args()
    {"scale": cmd_scale, "inject": cmd_inject, "real": cmd_real}[args.cmd](args)


if __name__ == "__main__":
    main()
