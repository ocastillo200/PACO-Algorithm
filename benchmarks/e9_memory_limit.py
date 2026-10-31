#!/usr/bin/env python3
"""
E9 -- does a run complete within a fixed amount of RAM?

The claim tested (Sect. "Memory-aware batching" of the paper): this
implementation sizes every large allocation from one budget, so a field
whose statistics do not fit in memory takes longer but still completes,
while VIP allocates the statistics of the whole field up front and needs
memory proportional to the field size whatever the machine has.

Running out of real RAM would make the machine swap or freeze, so each run
is a child process whose whole process tree is sampled every 50 ms and
killed as soon as it exceeds a limit L ("exceeded L"). This emulates a
machine with L of RAM, safely, on Linux and macOS.

Memory accounting:
  - Linux: sum of PSS (/proc/<pid>/smaps_rollup): pages shared between
    processes are split between them. Accurate; use this for the paper.
  - macOS: sum of RSS, which counts shared pages once per process. This
    over-counts this project's shared-memory cache and fork-shared cube, so
    it is biased against our implementation, not against VIP's private
    copies: a run of ours that fits under L here fits under L on Linux.

For each limit L, runs (each in its own child process):
  - VIP FastPACO (vip_hci.invprob.paco), fwhm in px, pixscale=1.0 (same
    patch as ours, as in Appendix B),
  - our PACO and FastPACO, given a budget of `--budget-share` x L
    through memory_fraction (the rest of L is left for the interpreter, the
    cube and the worker processes).
Our S/N maps are compared bit-for-bit with an unconstrained run of the same
class, so a run that completes is also checked to be correct.

Data: the EIDC SPHERE/IRDIS cube (default), or a synthetic cube
(`--synthetic H T`) to make the field's statistics larger than the limit.
The script prints the size of the full-field statistics
HW(TK + K^2 + K) x 8 bytes, which is what VIP holds at once.

Outputs (challenge/outputs/e9_memory/): a CSV with one row per (L, method):
completed / exceeded / failed, peak memory, wall time, identical-to-
reference; and a two-panel figure (peak memory and time vs L).

Usage
-----
    python3 benchmarks/e9_memory_limit.py --cpu 8                 # sph3, L = 4,2,1,0.5 GB
    python3 benchmarks/e9_memory_limit.py --cpu 8 --limits 16 8 4 2 --synthetic 400 200
    python3 benchmarks/e9_memory_limit.py --cpu 8 --skip-vip --limits 1 0.5
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import numpy as np
import psutil

OUT_DIR = os.path.join(ROOT, "challenge", "outputs", "e9_memory")
GiB = 1024 ** 3
METHODS = ["vip_fast", "ours_full", "ours_fast"]
LABELS = {"vip_fast": "VIP FastPACO", "ours_full": "PACO (this work)",
          "ours_fast": "FastPACO (this work)"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(synthetic):
    """(cube, pa, psf, fwhm, pxscale, name)."""
    if synthetic:
        from _common import make_synthetic_dataset
        size, T = synthetic
        fwhm = 4.0
        cube, pa, psf = make_synthetic_dataset(T, size, size, fwhm=fwhm)
        return cube, pa, psf, fwhm, 0.01225, f"synthetic_{size}x{size}x{T}"
    from challenge.make_contrast_curve_sph3 import load_sph3, compute_fwhm
    cube, pa, psf, pxs = load_sph3()
    return cube, pa, psf, compute_fwhm(psf), pxs, "sph3"


def full_field_stats_bytes(cube, fwhm):
    """HW(TK + K^2 + K) x 8: the patches, inverse covariances and means of
    every pixel, which VIP's compute_statistics holds at once."""
    from src.utils import create_boolean_circular_mask
    T, H, W = cube.shape
    K = int(create_boolean_circular_mask((H, W), radius=int(2 * fwhm) // 2).sum())
    return H * W * (T * K + K * K + K) * 8, K


# ---------------------------------------------------------------------------
# Child process: one run
# ---------------------------------------------------------------------------

def worker(args):
    cube, pa, psf, fwhm, pxs, _ = load_data(args.synthetic)
    if args.method == "vip_fast":
        from vip_hci.invprob.paco import FastPACO as VIP_FastPACO
        obj = VIP_FastPACO(cube, pa, psf, fwhm=fwhm, pixscale=1.0, verbose=False)
        t0 = time.perf_counter()
        snr, _flux = obj.run(cpu=args.cpu)
    else:
        from src.fast_paco import FastPACO
        from src.full_paco import PACO
        cls = PACO if args.method == "ours_full" else FastPACO
        kwargs = {}
        if args.budget_gb is not None:
            avail = psutil.virtual_memory().available
            kwargs["memory_fraction"] = min(1.0, args.budget_gb * GiB / avail)
        obj = cls(cube, pa, psf, fwhm=fwhm, pixscale=pxs, verbose=False, **kwargs)
        t0 = time.perf_counter()
        snr = obj.run(cpu=args.cpu)
    elapsed = time.perf_counter() - t0
    np.save(args.out_npy, np.asarray(snr))
    info = {"time_s": elapsed}
    stats = getattr(obj, "last_cache_stats", None)
    if stats:
        info["hit_rate"] = stats.get("hit_rate")
    print("E9_RESULT " + json.dumps(info), flush=True)


# ---------------------------------------------------------------------------
# Parent: run one child under a memory limit
# ---------------------------------------------------------------------------

def _pss_bytes(pid):
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def tree_bytes(proc, use_pss):
    try:
        procs = [proc] + proc.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for p in procs:
        try:
            total += _pss_bytes(p.pid) if use_pss else p.memory_info().rss
        except psutil.Error:
            pass
    return total


def kill_tree(proc):
    try:
        procs = proc.children(recursive=True) + [proc]
    except psutil.NoSuchProcess:
        return
    for p in procs:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=10)


def run_limited(method, limit_gb, budget_gb, args, out_npy):
    cmd = [sys.executable, os.path.abspath(__file__), "--worker",
           "--method", method, "--cpu", str(args.cpu), "--out-npy", out_npy]
    if budget_gb is not None:
        cmd += ["--budget-gb", str(budget_gb)]
    if args.synthetic:
        cmd += ["--synthetic", *map(str, args.synthetic)]
    use_pss = sys.platform.startswith("linux")
    limit = None if limit_gb is None else limit_gb * GiB

    t0 = time.perf_counter()
    child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, cwd=ROOT)
    proc = psutil.Process(child.pid)
    peak, status = 0, None
    while child.poll() is None:
        used = tree_bytes(proc, use_pss)
        peak = max(peak, used)
        if limit is not None and used > limit:
            kill_tree(proc)
            status = "exceeded"
            break
        if args.timeout and time.perf_counter() - t0 > args.timeout:
            kill_tree(proc)
            status = "timeout"
            break
        time.sleep(0.05)
    output = child.communicate()[0] or ""
    wall = time.perf_counter() - t0

    info = {}
    for line in output.splitlines():
        if line.startswith("E9_RESULT "):
            info = json.loads(line[len("E9_RESULT "):])
    if status is None:
        status = "completed" if info and child.returncode == 0 else "failed"
    if status == "failed":
        tail = "\n".join(output.strip().splitlines()[-5:])
        print(f"    child failed (exit {child.returncode}):\n{tail}")
    return {"status": status, "peak_gb": peak / GiB, "wall_s": wall,
            "run_s": info.get("time_s"), "hit_rate": info.get("hit_rate")}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_figure(rows, stats_gb, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    limits = sorted({r["limit_gb"] for r in rows})
    ax1.plot(limits, limits, "k:", lw=1, label="limit $L$")
    ax1.axhline(stats_gb, color="0.6", ls="--", lw=1, label="full-field statistics")
    for i, m in enumerate(METHODS):
        rs = sorted([r for r in rows if r["method"] == m], key=lambda r: r["limit_gb"])
        if not rs:
            continue
        x = [r["limit_gb"] for r in rs]
        ax1.plot(x, [r["peak_gb"] for r in rs], "o-", color=f"C{i}", label=LABELS[m])
        bad = [r for r in rs if r["status"] != "completed"]
        ax1.plot([r["limit_gb"] for r in bad], [r["peak_gb"] for r in bad], "x",
                 color=f"C{i}", ms=10, mew=2)
        ok = [r for r in rs if r["status"] == "completed"]
        ax2.plot([r["limit_gb"] for r in ok], [r["run_s"] for r in ok], "o-",
                 color=f"C{i}", label=LABELS[m])
    for ax in (ax1, ax2):
        if len(limits) > 1:
            ax.set_xscale("log")
        ax.set_xlabel("Memory limit $L$ [GB]")
    ax1.set_yscale("log")
    ax1.set_ylabel("Peak memory [GB] (x: killed at $L$)")
    ax1.legend(fontsize=8)
    ax2.set_ylabel("Run time [s] (completed runs)")
    if any(r["status"] == "completed" and r["run_s"] for r in rows):
        ax2.set_yscale("log")
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "no run completed", ha="center", va="center",
                 transform=ax2.transAxes)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.replace(".png", ".pdf"))
    print(f"Wrote {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu", type=int, default=4)
    ap.add_argument("--limits", type=float, nargs="+", default=[4, 2, 1, 0.5],
                    help="Memory limits L in GB (default: 4 2 1 0.5).")
    ap.add_argument("--budget-share", type=float, default=0.5,
                    help="Share of L given to our implementation as its budget "
                         "(default 0.5; the rest covers interpreter, cube, workers).")
    ap.add_argument("--synthetic", type=int, nargs=2, metavar=("SIZE", "T"),
                    help="Use a synthetic SIZE x SIZE x T cube instead of sph3.")
    ap.add_argument("--only", nargs="+", choices=METHODS, default=METHODS)
    ap.add_argument("--skip-vip", action="store_true")
    ap.add_argument("--timeout", type=float, default=0,
                    help="Kill a run after this many seconds (0 = no timeout).")
    # child-process mode
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    ap.add_argument("--budget-gb", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--out-npy", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        worker(args)
        return

    methods = [m for m in args.only if not (args.skip_vip and m == "vip_fast")]
    cube, _pa, _psf, fwhm, _pxs, name = load_data(args.synthetic)
    stats_bytes, K = full_field_stats_bytes(cube, fwhm)
    T, H, W = cube.shape
    accounting = "PSS (Linux)" if sys.platform.startswith("linux") else \
        "RSS sum (macOS; over-counts shared memory, biased against our implementation)"
    print(f"Data: {name}, (T,H,W)=({T},{H},{W}), K={K}")
    print(f"Full-field statistics HW(TK+K^2+K)x8 = {stats_bytes / GiB:.2f} GB")
    print(f"Available RAM now: {psutil.virtual_memory().available / GiB:.1f} GB; "
          f"memory accounting: {accounting}")
    del cube

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{name}_cpu{args.cpu}"

    # Unconstrained reference maps for the bit-for-bit check.
    refs = {}
    for m in methods:
        if m == "vip_fast":
            continue
        path = os.path.join(OUT_DIR, f"{tag}_{m}_reference.npy")
        print(f"\n[reference] {LABELS[m]}, no limit ...")
        r = run_limited(m, None, None, args, path)
        print(f"    {r['status']}, peak {r['peak_gb']:.2f} GB, {r['run_s'] or 0:.1f} s")
        if r["status"] == "completed":
            refs[m] = np.load(path)

    rows = []
    for L in sorted(args.limits, reverse=True):
        for m in methods:
            budget = None if m == "vip_fast" else args.budget_share * L
            path = os.path.join(OUT_DIR, f"{tag}_{m}_L{L:g}.npy")
            print(f"\n[L={L:g} GB] {LABELS[m]}"
                  + (f", budget {budget:.2f} GB" if budget else "") + " ...")
            r = run_limited(m, L, budget, args, path)
            identical = ""
            if r["status"] == "completed" and m in refs:
                snr = np.load(path)
                identical = bool(np.array_equal(np.nan_to_num(snr, nan=-999),
                                                np.nan_to_num(refs[m], nan=-999)))
            if r["status"] == "completed" and os.path.exists(path) and m != "vip_fast":
                os.remove(path)  # keep only references; the check is recorded
            row = {"data": name, "limit_gb": L, "method": m, "budget_gb": budget,
                   "status": r["status"], "peak_gb": round(r["peak_gb"], 3),
                   "run_s": None if r["run_s"] is None else round(r["run_s"], 1),
                   "wall_s": round(r["wall_s"], 1),
                   "hit_rate": None if r["hit_rate"] is None else round(r["hit_rate"], 3),
                   "identical_to_reference": identical}
            rows.append(row)
            print(f"    {row['status']}, peak {row['peak_gb']:.2f} GB"
                  + (f", {row['run_s']} s" if row["run_s"] is not None else "")
                  + (f", hit rate {row['hit_rate']}" if row["hit_rate"] is not None else "")
                  + (f", identical={identical}" if identical != "" else ""))

    print(f"\n=== E9 summary: {name}, cpu={args.cpu}, "
          f"full-field statistics {stats_bytes / GiB:.2f} GB, accounting: {accounting} ===")
    print(f"{'L [GB]':>7}  {'method':<24}{'status':>10}{'peak [GB]':>11}{'time [s]':>10}"
          f"{'hit rate':>10}{'identical':>11}")
    for r in rows:
        print(f"{r['limit_gb']:7g}  {LABELS[r['method']]:<24}{r['status']:>10}"
              f"{r['peak_gb']:11.2f}{'' if r['run_s'] is None else r['run_s']:>10}"
              f"{'' if r['hit_rate'] is None else r['hit_rate']:>10}"
              f"{str(r['identical_to_reference']):>11}")

    csv_path = os.path.join(OUT_DIR, f"{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")
    make_figure(rows, stats_bytes / GiB, os.path.join(OUT_DIR, f"{tag}.png"))


if __name__ == "__main__":
    main()
