#!/usr/bin/env python3
"""
Re-run just the sph3 (sphere_irdis, idx=3) Full/Fast PACO challenge results,
after the get_rotated_pixel_coords centre-offset fix (src/utils.py -- see
the "0.5px offset" investigation). Does NOT import run_paco_challenge.py
(which has no __main__ guard and would execute the full 9-dataset pipeline
as an import side effect) -- the needed functions (load_dataset,
compute_fwhm, run_fast_paco, run_full_paco_efficient, _compute_stats_one)
are copied verbatim from there instead, so this exercises the exact same
code path the original results were generated with, just with the fixed
coordinate function underneath.

Overwrites challenge/outputs/snr_maps/sphere_irdis_3_{full,fast}_snr.npy
(and the _flux_ counterparts) -- the pre-fix versions were backed up first
as *_pre_centerfix_backup.npy for comparison.

Usage
-----
    python3 challenge/rerun_sph3_challenge.py
"""
import sys
import os
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from astropy.io import fits
from vip_hci.fm import normalize_psf
from vip_hci.config.utils_conf import iterable, pool_map
from scipy.ndimage import map_coordinates

from src.paco_base import PACO
from src.fast_paco import FastPACOv2
from src.math_utils import compute_statistics_at_pixel, wfl, al, bl
from src.utils import create_boolean_circular_mask, get_rotated_pixel_coords

DATA_DIR = os.path.join(ROOT, "challenge", "subchallenge_1")
OUT_DIR = os.path.join(ROOT, "challenge", "outputs")
SNR_DIR = os.path.join(OUT_DIR, "snr_maps")

PXSCALE_OVERRIDE = {"nirc2": 0.020}
N_CPU = 6


def load_dataset(inst, idx):
    def fp(kind):
        return os.path.join(DATA_DIR, f"{inst}_{kind}_{idx}.fits")
    cube = fits.getdata(fp("cube")).astype(np.float64)
    pa = fits.getdata(fp("pa")).astype(np.float64).ravel()
    psf = fits.getdata(fp("psf")).astype(np.float64)
    pxs_d = fits.getdata(fp("pxscale"))
    pxs = float(pxs_d.flat[0])
    if np.isnan(pxs) or pxs <= 0:
        pxs = PXSCALE_OVERRIDE.get(inst, 0.020)
    if psf.ndim == 3:
        psf = np.nanmedian(psf, axis=0)
    return cube, pa, psf, pxs


def compute_fwhm(psf):
    from scipy.optimize import curve_fit

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
        profile = np.nan_to_num(profile, nan=peak)
        pn = profile / max(float(profile[0]), 1e-10)
        n_fit = max(5, r_max // 2)
        popt, _ = curve_fit(_gauss1d, r_vals[:n_fit], pn[:n_fit], p0=[2.0],
                            maxfev=2000, bounds=([0.5], [r_max]))
        sigma = float(abs(popt[0]))
        return max(2.355 * sigma, 2.0)
    except Exception:
        try:
            cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
            peak = psf[cy, cx]
            half = peak / 2.0
            ny, nx = psf.shape
            yy, xx = np.ogrid[:ny, :nx]
            r_arr = np.sqrt((yy - cy)**2 + (xx - cx)**2)
            r_max = int(min(cy, cx, ny - cy, nx - cx))
            r_vals = np.arange(r_max)
            profile = np.array([np.mean(psf[(r_arr >= rv - 0.5) & (r_arr < rv + 0.5)])
                                for rv in r_vals])
            for i in range(1, len(profile)):
                if profile[i] <= half:
                    r_half = r_vals[i - 1] + (half - profile[i - 1]) / (profile[i] - profile[i - 1])
                    return max(2.0 * r_half, 2.0)
        except Exception:
            pass
        return 5.0


def run_fast_paco(cube, pa, psf, fwhm, pxscale):
    t0 = time.time()
    fpaco = FastPACOv2(cube, pa, psf, fwhm=fwhm, pixscale=pxscale,
                       full_output=True, verbose=True)
    snr, flux = fpaco.run(cpu=N_CPU, full_output=True)
    return snr, flux, time.time() - t0


def _compute_stats_one(patch):
    return compute_statistics_at_pixel(patch)


def run_full_paco_efficient(cube, pa, psf, fwhm, pxscale, patch_radius_dec=0):
    t0 = time.time()

    T, H, W = cube.shape
    patch_width = int(2 * fwhm) - patch_radius_dec
    half = patch_width // 2

    cy, cx = H // 2, W // 2
    ref_mask = create_boolean_circular_mask((H, W), radius=half, center=(cy, cx))
    K = int(ref_mask.sum())

    ref_ys, ref_xs = np.where(ref_mask)
    dy = (ref_ys - cy).astype(np.int32)
    dx = (ref_xs - cx).astype(np.int32)

    norm_psf = normalize_psf(psf, verbose=False, force_odd=False)
    psf_cy, psf_cx = norm_psf.shape[0] // 2, norm_psf.shape[1] // 2
    psf_h, psf_w = norm_psf.shape

    r_min = int(-dy.min())
    r_max = H - 1 - int(dy.max())
    c_min = int(-dx.min())
    c_max = W - 1 - int(dx.max())

    print(f"  [FullPACO] Stage 1: stats (K={K}, cpu={N_CPU}) …")
    sys.stdout.flush()

    x_range = np.arange(0, H)
    y_range = np.arange(0, W)
    xx, yy = np.meshgrid(x_range, y_range)
    phi0s = np.column_stack((xx.ravel(), yy.ravel()))

    _paco = PACO(cube, pa, psf, fwhm=fwhm, pixscale=pxscale,
                patch_radius_dec=patch_radius_dec, verbose=False)

    raw_patches = list(pool_map(N_CPU, _paco.get_patch, iterable(phi0s)))
    stats = list(pool_map(N_CPU, _compute_stats_one, iterable(raw_patches)))
    del raw_patches

    m_arr = np.zeros((H, W, K), dtype=np.float64)
    Cinv_arr = np.zeros((H, W, K, K), dtype=np.float64)
    valid = np.zeros((H, W), dtype=bool)
    for idx_p, p0 in enumerate(phi0s):
        r, c = int(p0[0]), int(p0[1])
        mi, Ci = stats[idx_p]
        if mi is not None and Ci is not None and not np.any(np.isnan(Ci)):
            sz = min(len(mi), K)
            m_arr[r, c, :sz] = mi[:sz]
            Cinv_arr[r, c, :sz, :sz] = Ci[:sz, :sz]
            valid[r, c] = True
    del stats

    print(f"  [FullPACO] Stage 1 done in {time.time()-t0:.1f}s  "
          f"(valid: {valid.sum()}/{H*W})")
    sys.stdout.flush()

    print(f"  [FullPACO] Stage 2: sweeping {len(phi0s)} test positions …")
    sys.stdout.flush()
    t1 = time.time()

    grid_x, grid_y = np.meshgrid(np.arange(-W / 2, W / 2),
                                  np.arange(-H / 2, H / 2))
    a_arr = np.full(len(phi0s), np.nan)
    b_arr = np.full(len(phi0s), np.nan)

    for i, p0 in enumerate(phi0s):
        if i % 5000 == 0:
            elapsed = time.time() - t1
            frac = i / len(phi0s) if i > 0 else 1e-9
            eta = elapsed / frac * (1 - frac)
            print(f"    {i}/{len(phi0s)}  ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
            sys.stdout.flush()

        angles_px = get_rotated_pixel_coords(grid_x, grid_y, p0, pa)
        if (np.any(angles_px[:, 0] < 0) or np.any(angles_px[:, 0] >= W) or
                np.any(angles_px[:, 1] < 0) or np.any(angles_px[:, 1] >= H)):
            continue

        Cinlst, mlst, hlst, patch_lst = [], [], [], []
        ok = True
        for l, ang in enumerate(angles_px):
            row_i, col_i = int(ang[1]), int(ang[0])
            if not valid[row_i, col_i]:
                ok = False
                break
            if not (r_min <= row_i <= r_max and c_min <= col_i <= c_max):
                ok = False
                break

            row_frac = ang[1] - row_i
            col_frac = ang[0] - col_i
            psf_rows_f = psf_cy + dy - row_frac
            psf_cols_f = psf_cx + dx - col_frac
            h_l = np.zeros(K, dtype=np.float64)
            in_psf = ((psf_rows_f >= 0) & (psf_rows_f <= psf_h - 1) &
                      (psf_cols_f >= 0) & (psf_cols_f <= psf_w - 1))
            if np.any(in_psf):
                h_l[in_psf] = map_coordinates(
                    norm_psf, [psf_rows_f[in_psf], psf_cols_f[in_psf]],
                    order=1, mode="nearest")

            Cinlst.append(Cinv_arr[row_i, col_i])
            mlst.append(m_arr[row_i, col_i])
            hlst.append(h_l)
            rows_k = row_i + dy
            cols_k = col_i + dx
            raw = cube[l, rows_k, cols_k]
            patch_lst.append(raw)

        if ok and len(Cinlst) == T:
            wlst = wfl(hlst, Cinlst)
            a_arr[i] = al(hlst, wlst)
            b_arr[i] = bl(wlst, patch_lst, mlst)

    print(f"  [FullPACO] Stage 2 done in {time.time()-t1:.1f}s")

    a = a_arr.reshape(W, H).T
    b = b_arr.reshape(W, H).T
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = b / np.sqrt(a)
        flux = b / a

    return snr, flux, time.time() - t0


def main():
    inst, idx = "sphere_irdis", 3
    cube, pa, psf, pxscale = load_dataset(inst, idx)
    fwhm = compute_fwhm(psf)
    print(f"Dataset: {inst}_{idx}  shape={cube.shape}  fwhm={fwhm:.3f}px  "
          f"pxscale={pxscale:.5f}\"")

    print("\n--- FastPACO ---")
    snr_fast, flux_fast, dt_fast = run_fast_paco(cube, pa, psf, fwhm, pxscale)
    print(f"FastPACO done in {dt_fast:.1f}s")

    print("\n--- FullPACO (efficient) ---")
    snr_full, flux_full, dt_full = run_full_paco_efficient(cube, pa, psf, fwhm, pxscale)
    print(f"FullPACO done in {dt_full:.1f}s")

    np.save(os.path.join(SNR_DIR, f"{inst}_{idx}_fast_snr.npy"), snr_fast)
    np.save(os.path.join(SNR_DIR, f"{inst}_{idx}_fast_flux_snr.npy"), flux_fast)
    np.save(os.path.join(SNR_DIR, f"{inst}_{idx}_full_snr.npy"), snr_full)
    np.save(os.path.join(SNR_DIR, f"{inst}_{idx}_full_flux_snr.npy"), flux_full)
    print(f"\nWrote updated SNR/flux maps to {SNR_DIR}")


if __name__ == "__main__":
    main()
