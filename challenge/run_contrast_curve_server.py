#!/usr/bin/env python3
"""
Server-side entry point for Sect. 5.2 ("Contrast curves with injected
companions") of paco_paper.tex.

Runs the two-stage contrast-curve pipeline end to end, in one command, on
whatever machine it is invoked on (paths are resolved relative to this
file, not hardcoded to any one workstation):

  1. challenge/calibrate_contrast_units.py -- measures the empirical
     alpha-to-true-contrast calibration factor via a controlled
     single-companion injection/recovery test. This has to be re-measured
     here rather than reusing the previously-hardcoded value (3.51942)
     because that number was measured before the 0.5px coordinate-center
     fix and the patch-extraction fixes made to src/ this session -- both
     touch the exact flux computation this factor calibrates.
  2. challenge/make_contrast_curve_sph3.py -- runs VIP's injection-recovery
     contrast_curve machinery against FullPACOv2 on the full field, using
     the calibration factor measured in step 1, and writes the CSV/figure
     this section of the paper is built from.

Each stage is run as its own subprocess (not imported) so that this
script stays a thin orchestrator and each stage remains independently
runnable/debuggable exactly as before.

Usage
-----
    # sensible default: use (all cores - 1) for both stages
    python3 challenge/run_contrast_curve_server.py

    # pin core count explicitly (e.g. to leave headroom on a shared box)
    python3 challenge/run_contrast_curve_server.py --cpu 32

    # skip stage 1 and reuse a previously-measured factor
    python3 challenge/run_contrast_curve_server.py --calibration-factor 3.51942

    # widen the contrast-curve sampling (more injected branches/annuli)
    python3 challenge/run_contrast_curve_server.py --cpu 48 --nbranch 3 --fc-rad-sep 2

Outputs (unchanged from running the two scripts by hand)
----------------------------------------------------------
    challenge/figures/contrast_curve_sph3.csv
    challenge/figures/fig_contrast_curve_sph3.pdf
    Astronomy_and_Astrophysics_Template/figures/fig_contrast_curve_sph3.pdf

Send back contrast_curve_sph3.csv (and the pdf, if useful to look at) so
Sect. 5.2 of the paper can be updated with the measured numbers.
"""
import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHALLENGE_DIR = os.path.join(ROOT, "challenge")

CALIBRATION_FACTOR_RE = re.compile(
    r"implied calibration factor A_eff.*?=\s*([0-9.eE+-]+)\s*\(compare")


def run_streamed(cmd, cwd):
    """Run a subprocess, streaming its stdout live, and return the full
    captured text (so it can be parsed afterward) -- long PACO runs print
    their own progress and we do not want that buffered/hidden for
    minutes at a time on a server."""
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): "
                           f"{' '.join(cmd)}")
    return "".join(lines)


def main():
    default_cpu = max(1, (os.cpu_count() or 4) - 1)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu", type=int, default=default_cpu,
                        help=f"Worker processes for both stages "
                             f"(default: cpu_count-1 = {default_cpu} on this "
                             f"machine).")
    parser.add_argument("--calibration-factor", type=float, default=None,
                        help="Skip stage 1 (calibration measurement) and "
                             "use this factor directly for stage 2.")
    parser.add_argument("--crop", type=int, default=120,
                        help="Passed through to make_contrast_curve_sph3.py "
                             "(see its --help for why cropping is needed).")
    parser.add_argument("--nbranch", type=int, default=1,
                        help="Passed through to make_contrast_curve_sph3.py. "
                             "Each additional branch multiplies the number "
                             "of full FullPACOv2 runs dispatched "
                             "(1 + nbranch*fc_rad_sep total).")
    parser.add_argument("--fc-rad-sep", type=int, default=3,
                        help="Passed through to make_contrast_curve_sph3.py.")
    parser.add_argument("--fc-snr", type=float, default=0.001,
                        help="Passed through to make_contrast_curve_sph3.py.")
    args = parser.parse_args()

    t_start = time.perf_counter()

    if args.calibration_factor is not None:
        factor = args.calibration_factor
        print(f"Skipping calibration stage; using provided factor = {factor}")
    else:
        print(f"=== Stage 1/2: calibration (cpu={args.cpu}) ===")
        out = run_streamed(
            [sys.executable, "-u",
             os.path.join(CHALLENGE_DIR, "calibrate_contrast_units.py"),
             "--cpu", str(args.cpu)],
            cwd=ROOT)
        m = CALIBRATION_FACTOR_RE.search(out)
        if not m:
            print("\nERROR: could not parse the calibration factor from "
                  "calibrate_contrast_units.py's output. Re-run it manually "
                  "and pass the printed 'A_eff' value via "
                  "--calibration-factor.", file=sys.stderr)
            sys.exit(1)
        factor = float(m.group(1))
        print(f"\n=== Measured calibration factor: {factor:.6g} ===")

    print(f"\n=== Stage 2/2: contrast curve (cpu={args.cpu}, "
          f"crop={args.crop}, nbranch={args.nbranch}, "
          f"fc_rad_sep={args.fc_rad_sep}, fc_snr={args.fc_snr}) ===")
    run_streamed(
        [sys.executable, "-u",
         os.path.join(CHALLENGE_DIR, "make_contrast_curve_sph3.py"),
         "--cpu", str(args.cpu), "--crop", str(args.crop),
         "--nbranch", str(args.nbranch), "--fc-rad-sep", str(args.fc_rad_sep),
         "--fc-snr", str(args.fc_snr), "--calibration-factor", str(factor)],
        cwd=ROOT)

    dt = time.perf_counter() - t_start
    print(f"\n=== Done in {dt/60:.1f} min ===")
    print(f"Calibration factor used: {factor:.6g}")
    print("Outputs:")
    print(f"  {os.path.join(CHALLENGE_DIR, 'figures', 'contrast_curve_sph3.csv')}")
    print(f"  {os.path.join(CHALLENGE_DIR, 'figures', 'fig_contrast_curve_sph3.pdf')}")
    print("\nSend contrast_curve_sph3.csv back to update Sect. 5.2 of the paper.")


if __name__ == "__main__":
    main()
