#!/usr/bin/env python3
"""
Empirical calibration: inject ONE known-contrast companion directly (not
through vip_hci.metrics.contrast_curve's automatic fc_snr*noise sizing,
which turned out to size the injection nonsensically for our flux map's
scale -- see make_contrast_curve_sph3.py's run: throughput/sensitivity
came back all-NaN), recover it with FullPACOv2, and measure the actual
ratio between recovered alpha and the known injected contrast. This
replaces the analytical unit derivation in make_contrast_curve_sph3.py's
docstring with a direct empirical check, since the derivation's
assumption (that alpha is already ~O(1e-3)-scale contrast) is contradicted
by the observed flux-map noise level (~45-100) from that run.

Usage
-----
    python3 challenge/calibrate_contrast_units.py --cpu 32
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
from vip_hci.fm import cube_inject_companions, normalize_psf
from vip_hci.metrics import aperture_flux
from vip_hci.var import frame_center

from src.full_paco import FullPACOv2
from challenge.make_contrast_curve_sph3 import load_sph3, compute_fwhm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, default=4)
    args = parser.parse_args()

    cube, pa, psf, pxscale = load_sph3()
    fwhm = compute_fwhm(psf)
    cy, cx = frame_center(cube[0])
    psf_cy, psf_cx = psf.shape[0] / 2.0, psf.shape[1] / 2.0
    A = float(aperture_flux(psf, [psf_cy], [psf_cx], fwhm)[0])

    rad = 20.0
    theta = 0.0
    target_contrast = 1e-3
    flux_vip_units = target_contrast * A

    size = int(round(3 * fwhm))
    if size % 2 == 0:
        size += 1
    size = min(size, psf.shape[1])
    norm_psf = normalize_psf(psf, fwhm=fwhm, verbose=False, size=size)

    cube_inj = cube_inject_companions(cube.copy(), norm_psf, pa, flux_vip_units,
                                      rad_dists=[rad], theta=theta, verbose=False)

    y_inj = cy + rad * np.sin(np.deg2rad(theta))
    x_inj = cx + rad * np.cos(np.deg2rad(theta))

    print(f"A (aperture_flux of raw psf, 1 FWHM) = {A}")
    print(f"target_contrast = {target_contrast}, flux_vip_units (injected, "
          f"normalized-psf convention) = {flux_vip_units}")
    print(f"Injected at (y,x) = ({y_inj:.2f}, {x_inj:.2f}), fwhm={fwhm:.3f}")

    t0 = time.time()
    obj_nofc = FullPACOv2(cube, pa, psf, fwhm=fwhm, pixscale=pxscale, verbose=False)
    snr0, flux0 = obj_nofc.run(cpu=args.cpu, use_gpu=False, full_output=True)
    print(f"no-fc run: {time.time() - t0:.1f}s (cpu={args.cpu})")

    t0 = time.time()
    obj_fc = FullPACOv2(cube_inj, pa, psf, fwhm=fwhm, pixscale=pxscale, verbose=False)
    snr1, flux1 = obj_fc.run(cpu=args.cpu, use_gpu=False, full_output=True)
    print(f"fc run: {time.time() - t0:.1f}s (cpu={args.cpu})")

    recovered = aperture_flux(flux1 - flux0, [y_inj], [x_inj], fwhm)[0]
    print(f"\nrecovered alpha (aperture flux of flux1-flux0 at injection) = {recovered}")
    print(f"ratio recovered / target_contrast      = {recovered / target_contrast:.6g}")
    print(f"ratio recovered / flux_vip_units        = {recovered / flux_vip_units:.6g}")
    # NOTE: this exact line is parsed by challenge/run_contrast_curve_server.py
    # to feed make_contrast_curve_sph3.py's --calibration-factor -- keep the
    # "implied calibration factor A_eff = " prefix and the trailing number
    # stable if this print is ever reworded.
    print(f"implied calibration factor A_eff = recovered/target_contrast = "
          f"{recovered / target_contrast:.6g}  (compare to A = {A:.6g})")

    r_int, c_int = int(round(y_inj)), int(round(x_inj))
    print(f"\npixel-center check (no aperture averaging):")
    print(f"  flux0[{r_int},{c_int}] = {flux0[r_int, c_int]}")
    print(f"  flux1[{r_int},{c_int}] = {flux1[r_int, c_int]}")
    print(f"  diff                   = {flux1[r_int, c_int] - flux0[r_int, c_int]}")
    print(f"  snr0[{r_int},{c_int}]  = {snr0[r_int, c_int]}")
    print(f"  snr1[{r_int},{c_int}]  = {snr1[r_int, c_int]}")

    print(f"\nflux0 field stats (no-fc): "
          f"nanmedian={np.nanmedian(flux0):.4g}, "
          f"nanstd={np.nanstd(flux0):.4g}, "
          f"nan_frac={np.mean(np.isnan(flux0)):.4f}")
    print(f"flux1 field stats (with-fc): "
          f"nanmedian={np.nanmedian(flux1):.4g}, "
          f"nanstd={np.nanstd(flux1):.4g}, "
          f"nan_frac={np.mean(np.isnan(flux1)):.4f}")


if __name__ == "__main__":
    main()
