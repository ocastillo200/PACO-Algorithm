#!/usr/bin/env python3
"""
Full and Fast PACO S/N maps of the EIDC sph3 dataset (sphere_irdis,
idx=3), computed with the released classes (src.full_paco.PACO and
src.fast_paco.FastPACO), for Sect. 5.3 and Table 6 of the paper; plotted
by plot_challenge_results.py.

Writes challenge/outputs/snr_maps/sphere_irdis_3_{full,fast}_snr.npy
(and the _flux_ counterparts).

Usage
-----
    python3 challenge/rerun_sph3_challenge.py --cpu 32
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

from challenge.sph3_data import compute_fwhm, load_sph3
from src.fast_paco import FastPACO
from src.full_paco import PACO

SNR_DIR = os.path.join(ROOT, "challenge", "outputs", "snr_maps")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu", type=int, default=os.cpu_count())
    args = parser.parse_args()

    cube, pa, psf, pxscale = load_sph3()
    fwhm = compute_fwhm(psf)
    print(f"sph3: shape={cube.shape}  fwhm={fwhm:.3f}px  pxscale={pxscale:.5f}\"")

    os.makedirs(SNR_DIR, exist_ok=True)
    for name, cls in (("fast", FastPACO), ("full", PACO)):
        t0 = time.time()
        obj = cls(cube, pa, psf, fwhm=fwhm, pixscale=pxscale, full_output=True)
        snr, flux = obj.run(cpu=args.cpu, full_output=True)
        print(f"{name}: {time.time() - t0:.1f} s, max S/N {np.nanmax(snr):.2f}")
        np.save(os.path.join(SNR_DIR, f"sphere_irdis_3_{name}_snr.npy"), snr)
        np.save(os.path.join(SNR_DIR, f"sphere_irdis_3_{name}_flux_snr.npy"), flux)
    print(f"Wrote S/N and flux maps to {SNR_DIR}")


if __name__ == "__main__":
    main()
