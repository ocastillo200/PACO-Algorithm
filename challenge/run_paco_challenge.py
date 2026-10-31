#!/usr/bin/env python3
"""
Run FullPACO and FastPACO on all 9 ADI subchallenge datasets.

FullPACO uses a memory-efficient streaming approach (no pre-stored patch cube).
FastPACO uses the standard two-stage batch approach.
Both are compared side-by-side.
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
from vip_hci.metrics import compute_binary_map
from vip_hci.preproc.rescaling import frame_shift
from vip_hci.config.utils_conf import iterable, pool_map
from scipy.ndimage import maximum_filter, label, find_objects, map_coordinates

from src.paco_base import PACOBase
from src.fast_paco import FastPACO
from src.math_utils import compute_statistics_at_pixel, wfl, al, bl
from src.utils import create_boolean_circular_mask, get_rotated_pixel_coords

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(ROOT, "challenge", "subchallenge_1")
OUT_DIR  = os.path.join(ROOT, "challenge", "outputs")
SNR_DIR  = os.path.join(OUT_DIR, "snr_maps")
REF_DIR  = os.path.join(OUT_DIR, "submission", "ref")
RES_DIR  = os.path.join(OUT_DIR, "submission", "res")

PXSCALE_OVERRIDE = {"nirc2": 0.020}  # nirc2 pxscale files contain nan
N_CPU = 6
THRESHOLD = 5.0

DATASETS = [
    ("sphere_irdis", 1),
    ("sphere_irdis", 2),
    ("sphere_irdis", 3),
    ("nirc2",        1),
    ("nirc2",        2),
    ("nirc2",        3),
    # LMIRCam skipped by user request: ~4800-frame cubes make FastPACO's
    # per-pixel statistics stage impractically slow on this machine.
]

# LMIRCam: T ≈ 4000+ frames → FullPACO sweep impractical
# NIRC2: K ≈ 197 (FWHM≈8px, H=W=321) → Cinv_arr would be (321,321,197,197) ≈ 32 GB
# Both instrument groups run FastPACO only.
FULL_PACO_ONLY_FAST = {("lmircam", 1), ("lmircam", 2), ("lmircam", 3),
                        ("nirc2",    1), ("nirc2",    2), ("nirc2",    3)}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(inst, idx):
    def fp(kind):
        return os.path.join(DATA_DIR, f"{inst}_{kind}_{idx}.fits")
    cube  = fits.getdata(fp("cube")).astype(np.float64)
    pa    = fits.getdata(fp("pa")).astype(np.float64).ravel()
    psf   = fits.getdata(fp("psf")).astype(np.float64)
    pxs_d = fits.getdata(fp("pxscale"))
    pxs   = float(pxs_d.flat[0])
    if np.isnan(pxs) or pxs <= 0:
        pxs = PXSCALE_OVERRIDE.get(inst, 0.020)
    if psf.ndim == 3:
        psf = np.nanmedian(psf, axis=0)
    return cube, pa, psf, pxs


def compute_fwhm(psf):
    """
    Estimate FWHM by fitting a 1-D Gaussian to the PSF radial profile.

    Avoids VIP's fit_2dgaussian which hangs in Python 3.13 multiprocessing
    environments. Takes <10 ms per call.
    """
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
        # Fit 1-D Gaussian up to the first minimum (avoid ring artefacts)
        pn = profile / max(float(profile[0]), 1e-10)
        n_fit = max(5, r_max // 2)
        popt, _ = curve_fit(_gauss1d, r_vals[:n_fit], pn[:n_fit], p0=[2.0],
                             maxfev=2000, bounds=([0.5], [r_max]))
        sigma = float(abs(popt[0]))
        return max(2.355 * sigma, 2.0)
    except Exception:
        # Fallback: linear interpolation at half-maximum
        try:
            cy, cx   = psf.shape[0] // 2, psf.shape[1] // 2
            peak     = psf[cy, cx]
            half     = peak / 2.0
            ny, nx   = psf.shape
            yy, xx   = np.ogrid[:ny, :nx]
            r_arr    = np.sqrt((yy - cy)**2 + (xx - cx)**2)
            r_max    = int(min(cy, cx, ny - cy, nx - cx))
            r_vals   = np.arange(r_max)
            profile  = np.array([np.mean(psf[(r_arr >= rv - 0.5) & (r_arr < rv + 0.5)])
                                 for rv in r_vals])
            for i in range(1, len(profile)):
                if profile[i] <= half:
                    r_half = r_vals[i - 1] + (half - profile[i - 1]) / (profile[i] - profile[i - 1])
                    return max(2.0 * r_half, 2.0)
        except Exception:
            pass
        return 5.0


def snr_path(inst, idx, variant):
    return os.path.join(SNR_DIR, f"{inst}_{idx}_{variant}_snr.npy")


# ─────────────────────────────────────────────────────────────────────────────
# FastPACO wrapper
# ─────────────────────────────────────────────────────────────────────────────

def run_fast_paco(cube, pa, psf, fwhm, pxscale):
    t0 = time.time()
    fpaco = FastPACO(cube, pa, psf, fwhm=fwhm, pixscale=pxscale,
                       full_output=True, verbose=True)
    snr, flux = fpaco.run(cpu=N_CPU, full_output=True)
    return snr, flux, time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Memory-efficient FullPACO
# (Does not pre-store patches; Cinv/m stored, patches resampled per test pixel)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_stats_one(patch):
    """Return (m, Cinv) for one pixel's patch stack."""
    return compute_statistics_at_pixel(patch)


def run_full_paco_efficient(cube, pa, psf, fwhm, pxscale, patch_radius_dec=0):
    """
    Full PACO without patch pre-storage.

    Stage 1 : compute Cinv[y,x] and m[y,x] for all pixels in parallel, using
              the same full-image circular mask as PACOBase.get_patch so K matches.
    Stage 2 : sweep test positions; patches are extracted from the cube at runtime
              using pre-computed relative pixel offsets (consistent with Stage 1).

    patch_radius_dec : int
        patch_width = int(2*fwhm) - patch_radius_dec, matching PACO's own
        constructor parameter (negative values grow the patch beyond the
        2*fwhm default).
    """
    t0 = time.time()

    T, H, W = cube.shape
    patch_width = int(2 * fwhm) - patch_radius_dec
    half        = patch_width // 2

    # K is defined by a full-image mask (matching PACOBase.get_patch behaviour).
    # Use image center as reference to count K.
    cy, cx   = H // 2, W // 2
    ref_mask = create_boolean_circular_mask((H, W), radius=half, center=(cy, cx))
    K        = int(ref_mask.sum())

    # Relative row/col offsets of the K pixels around any interior pixel
    ref_ys, ref_xs = np.where(ref_mask)
    dy = (ref_ys - cy).astype(np.int32)   # shape (K,)
    dx = (ref_xs - cx).astype(np.int32)   # shape (K,)

    # PSF template, sampled at the same relative (dy, dx) offsets. Unlike a
    # single static h_ref, this must be re-sampled per-frame at the *exact*
    # sub-pixel rotated position (Eq. 15-16 of [FLA18] require h evaluated at
    # the continuous phi_tl, not at the nearest integer pixel) — otherwise
    # the matched filter is mis-centered by up to 0.5px on every frame, which
    # silently costs several percent of S/N on every test position.
    norm_psf = normalize_psf(psf, verbose=False, force_odd=False)
    psf_cy, psf_cx = norm_psf.shape[0] // 2, norm_psf.shape[1] // 2
    psf_h, psf_w = norm_psf.shape

    # Boundary: a pixel (r, c) is patchable if all K neighbors are in image
    r_min = int(-dy.min());  r_max = H - 1 - int(dy.max())
    c_min = int(-dx.min());  c_max = W - 1 - int(dx.max())

    # ---------- Stage 1: statistics for all interior pixels ----------
    print(f"  [FullPACO] Stage 1: stats (K={K}, cpu={N_CPU}) …")
    sys.stdout.flush()

    x_range = np.arange(0, H)
    y_range = np.arange(0, W)
    xx, yy  = np.meshgrid(x_range, y_range)
    phi0s   = np.column_stack((xx.ravel(), yy.ravel()))  # (W×H, 2) as (row, col)

    _paco = PACOBase(cube, pa, psf, fwhm=fwhm, pixscale=pxscale,
                 patch_radius_dec=patch_radius_dec, verbose=False)

    raw_patches = list(pool_map(N_CPU, _paco.get_patch, iterable(phi0s)))
    stats       = list(pool_map(N_CPU, _compute_stats_one, iterable(raw_patches)))
    del raw_patches

    m_arr    = np.zeros((H, W, K), dtype=np.float64)
    Cinv_arr = np.zeros((H, W, K, K), dtype=np.float64)
    valid    = np.zeros((H, W), dtype=bool)
    for idx_p, p0 in enumerate(phi0s):
        r, c   = int(p0[0]), int(p0[1])
        mi, Ci = stats[idx_p]
        if mi is not None and Ci is not None and not np.any(np.isnan(Ci)):
            sz = min(len(mi), K)
            m_arr[r, c, :sz]         = mi[:sz]
            Cinv_arr[r, c, :sz, :sz] = Ci[:sz, :sz]
            valid[r, c]              = True
    del stats

    print(f"  [FullPACO] Stage 1 done in {time.time()-t0:.1f}s  "
          f"(valid: {valid.sum()}/{H*W})")
    sys.stdout.flush()

    # ---------- Stage 2: sweep ----------
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
            frac    = i / len(phi0s) if i > 0 else 1e-9
            eta     = elapsed / frac * (1 - frac)
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
                ok = False; break
            if not (r_min <= row_i <= r_max and c_min <= col_i <= c_max):
                ok = False; break

            # Sub-pixel PSF astrometry: sample the PSF at the true continuous
            # rotated position (row_i+row_frac, col_i+col_frac), not just at
            # the nearest integer pixel used for the background patch.
            row_frac = ang[1] - row_i
            col_frac = ang[0] - col_i
            # frame_shift(img, sy, sx)[i,j] = img[i-sy, j-sx] (verified empirically),
            # so sampling the *unshifted* PSF at the equivalent offset requires
            # subtracting the fractional shift, not adding it.
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
            # extract K pixels using pre-computed relative offsets
            rows_k = row_i + dy
            cols_k = col_i + dx
            raw    = cube[l, rows_k, cols_k]   # shape (K,)
            patch_lst.append(raw)

        if ok and len(Cinlst) == T:
            wlst     = wfl(hlst, Cinlst)
            a_arr[i] = al(hlst, wlst)
            b_arr[i] = bl(wlst, patch_lst, mlst)

    print(f"  [FullPACO] Stage 2 done in {time.time()-t1:.1f}s")

    # reshape to (H, W)
    a = a_arr.reshape(W, H).T
    b = b_arr.reshape(W, H).T
    with np.errstate(invalid="ignore", divide="ignore"):
        snr  = b / np.sqrt(a)
        flux = b / a

    return snr, flux, time.time() - t0


def refine_subpixel_peaks(cube, pa, psf, fwhm, pxscale, snr_map,
                          patch_radius_dec=0, lower_thresh=3.5,
                          search_radius=1.5, step=0.1, verbose=True):
    """
    Sub-pixel refinement of near-threshold candidates (Sec. 4.3 of [FLA18]:
    "the source location is also improved by testing subpixel locations on
    a refined grid around the location provided by PACO detection").

    `run_full_paco_efficient` (and PACOBase.run()'s `oversampling` argument)
    only evaluate candidate positions phi0 on the *native* integer pixel
    grid. A real point source whose true position falls between two grid
    nodes is penalised there — Table 2 of [FLA18] quantifies this as up to
    ~9% S/N loss for a source located exactly between two adjacent pixels
    (s=1 sampling). That is enough, by itself, to push a marginal ~5.2-5.4
    sigma source below the 5 sigma detection threshold.

    This function fixes that specifically for candidates that are already
    close to the threshold (`lower_thresh`) rather than re-running the
    full field at higher oversampling (which PACOBase.run(oversampling=s>1)
    would require, at a cost that scales as s^2): for each local maximum
    of `snr_map` above `lower_thresh`, it re-evaluates the exact PACO
    statistic (Eqs. 15-16) on a fine (`step`-spaced) sub-pixel grid within
    `search_radius` pixels, using the true continuous rotated position at
    every frame (relies on the `get_rotated_pixel_coords` fix that removed
    the int() truncation of the candidate position — verified to make
    `oversampling` a genuine, non-degenerate sub-pixel search). The pixel
    value is only ever replaced by a *higher* S/N, since a real source's
    matched-filter response is smooth around its true peak; a properly
    detected candidate whose true peak is already on-grid is unaffected.

    Parameters
    ----------
    cube, pa, psf, fwhm, pxscale : as elsewhere in this module.
    snr_map : np.ndarray (H, W)
        Native-grid S/N map to refine (e.g. output of run_full_paco_efficient).
    lower_thresh : float
        Only candidates already at or above this S/N are refined (keeps the
        cost proportional to the number of near-threshold candidates, not
        to the full field).
    search_radius : float
        Half-width (pixels) of the local sub-pixel search window.
    step : float
        Grid spacing (pixels) of the local sub-pixel search.

    Returns
    -------
    refined_snr : np.ndarray (H, W)
        Copy of `snr_map` with near-threshold peaks replaced by their true
        (higher) sub-pixel S/N where found.
    report : list of dict
        One entry per refined candidate: native position, old/new S/N.
    """
    T, H, W = cube.shape
    patch_width = int(2 * fwhm) - patch_radius_dec
    half        = patch_width // 2

    cy, cx   = H // 2, W // 2
    ref_mask = create_boolean_circular_mask((H, W), radius=half, center=(cy, cx))
    K        = int(ref_mask.sum())
    ref_ys, ref_xs = np.where(ref_mask)
    dy = (ref_ys - cy).astype(np.int32)
    dx = (ref_xs - cx).astype(np.int32)

    norm_psf = normalize_psf(psf, verbose=False, force_odd=False)
    psf_cy, psf_cx = norm_psf.shape[0] // 2, norm_psf.shape[1] // 2
    psf_h, psf_w = norm_psf.shape

    r_min = int(-dy.min());  r_max = H - 1 - int(dy.max())
    c_min = int(-dx.min());  c_max = W - 1 - int(dx.max())

    if verbose:
        print(f"  [Refine] Stage 1: stats (K={K}) …")
    _paco = PACOBase(cube, pa, psf, fwhm=fwhm, pixscale=pxscale,
                 patch_radius_dec=patch_radius_dec, verbose=False)
    stats_cache = {}

    def get_stats(row_i, col_i):
        key = (row_i, col_i)
        if key not in stats_cache:
            apatch = _paco.get_patch((row_i, col_i))
            stats_cache[key] = compute_statistics_at_pixel(apatch)
        return stats_cache[key]

    grid_x, grid_y = np.meshgrid(np.arange(-W / 2, W / 2),
                                 np.arange(-H / 2, H / 2))

    def snr_at(p0_row, p0_col):
        angles_px = get_rotated_pixel_coords(grid_x, grid_y, (p0_row, p0_col), pa)
        a_tot, b_tot = 0.0, 0.0
        for l in range(T):
            col_f, row_f = angles_px[l]
            if not (r_min <= row_f <= r_max and c_min <= col_f <= c_max):
                return np.nan
            row_i, col_i = int(row_f), int(col_f)
            m, Cinv = get_stats(row_i, col_i)
            if m is None or Cinv is None or np.any(np.isnan(Cinv)):
                return np.nan
            row_frac, col_frac = row_f - row_i, col_f - col_i
            psf_rows_f = psf_cy + dy - row_frac
            psf_cols_f = psf_cx + dx - col_frac
            in_psf = ((psf_rows_f >= 0) & (psf_rows_f <= psf_h - 1) &
                      (psf_cols_f >= 0) & (psf_cols_f <= psf_w - 1))
            h_l = np.zeros(K, dtype=np.float64)
            if np.any(in_psf):
                h_l[in_psf] = map_coordinates(
                    norm_psf, [psf_rows_f[in_psf], psf_cols_f[in_psf]],
                    order=1, mode="nearest")
            raw = cube[l, row_i + dy, col_i + dx]
            w_l = Cinv @ h_l
            a_tot += float(h_l @ w_l)
            b_tot += float(w_l @ (raw - m))
        return b_tot / np.sqrt(a_tot) if a_tot > 1e-8 else np.nan

    frame = np.nan_to_num(snr_map, nan=-100.0)
    data_max = maximum_filter(frame, size=max(1, int(round(fwhm))))
    peak_mask = (frame == data_max) & (frame >= lower_thresh)
    candidates = list(zip(*np.where(peak_mask)))

    if verbose:
        print(f"  [Refine] {len(candidates)} candidate(s) >= {lower_thresh} sigma "
              f"to refine (radius={search_radius}px, step={step}px)")

    offsets = np.arange(-search_radius, search_radius + 1e-9, step)
    refined = snr_map.copy()
    report = []
    for (iy, ix) in candidates:
        best_snr, best_off = frame[iy, ix], (0.0, 0.0)
        for doff_y in offsets:
            for doff_x in offsets:
                if doff_y == 0.0 and doff_x == 0.0:
                    continue
                s = snr_at(iy + doff_y, ix + doff_x)
                if np.isfinite(s) and s > best_snr:
                    best_snr, best_off = s, (doff_y, doff_x)
        if best_snr > frame[iy, ix]:
            refined[iy, ix] = best_snr
            report.append({"x": ix, "y": iy, "old_snr": float(frame[iy, ix]),
                           "new_snr": float(best_snr), "offset": best_off})
            if verbose:
                print(f"    ({ix},{iy}): {frame[iy, ix]:.2f} -> {best_snr:.2f}  "
                      f"(refined offset dy={best_off[0]:+.2f}, dx={best_off[1]:+.2f})")
    return refined, report


# ─────────────────────────────────────────────────────────────────────────────
# Self-injection helpers (for metric test on injection-free datasets)
# ─────────────────────────────────────────────────────────────────────────────

def inject_companion_into_cube(cube, psf, pa, sep_px, pa_deg_inj, contrast):
    """
    Inject a scaled PSF at (derotated) separation sep_px and position angle
    pa_deg_inj, sky-fixed across the sequence.

    Sign/convention note: a raw-frame position must be rotated by -pa[t]
    relative to the derotated (sky) position to match src.utils.
    get_rotated_pixel_coords (used by both FastPACO/FullPACO) and VIP's own
    cube_derotate (which rotates by -angle_list[i]). Verified empirically:
    derotating (cube_inj - cube) with vip_hci.preproc.cube_derotate and
    median-combining collapses to a single sharp peak only with this sign;
    the previous '+pa[t]' convention spreads the injected flux across the
    field with no coherent peak, i.e. it never created a real, recoverable
    companion regardless of contrast.
    """
    T, H, W = cube.shape
    cy, cx  = H // 2, W // 2
    norm_psf = normalize_psf(psf, force_odd=True, verbose=False)
    norm_psf = norm_psf / norm_psf.max()
    ph = norm_psf.shape[0] // 2

    cube_inj = cube.copy()
    for t in range(T):
        theta  = np.deg2rad(pa_deg_inj - pa[t])
        dx     = sep_px * np.sin(theta)
        dy     = -sep_px * np.cos(theta)
        cx_t   = cx + dx
        cy_t   = cy + dy
        ix     = int(round(cx_t))
        iy     = int(round(cy_t))
        psf_s  = frame_shift(norm_psf, cy_t - iy, cx_t - ix)
        x0, x1 = ix - ph, ix + ph + 1
        y0, y1 = iy - ph, iy + ph + 1
        px0, px1 = max(0, -x0), min(W, W - max(0, x1 - W))
        py0, py1 = max(0, -y0), min(H, H - max(0, y1 - H))
        ix0, ix1 = max(0, x0), min(W, x1)
        iy0, iy1 = max(0, y0), min(H, y1)
        sl_cube = (t, slice(iy0, iy1), slice(ix0, ix1))
        sl_psf  = (slice(py0, py1), slice(px0, px1))
        h = min(iy1-iy0, py1-py0)
        w = min(ix1-ix0, px1-px0)
        cube_inj[t, iy0:iy0+h, ix0:ix0+w] += contrast * psf_s[py0:py0+h, px0:px0+w]

    # Report the position in the DEROTATED (sky/output) frame, i.e. purely
    # pa_deg_inj with no per-frame pa[t] term — this is what a detection map
    # (which is itself already in the derotated frame) should be scored
    # against. Verified to match the peak location found by an independent
    # cube_derotate + median-combine of the injected signal alone.
    theta0 = np.deg2rad(pa_deg_inj)
    x_inj  = cx + sep_px * np.sin(theta0)
    y_inj  = cy - sep_px * np.cos(theta0)
    return cube_inj, (x_inj, y_inj)


def compute_metric_sweep(snr_map, positions_xy, fwhm):
    """Compute TPR, FDR, FPR over a threshold grid."""
    ny, nx   = snr_map.shape
    thr_grid = np.linspace(0, 2 * THRESHOLD, 80)
    n_inj    = len(positions_xy)
    # resolution elements in field (annulus between 2*fwhm and border)
    r_inner  = 2 * fwhm
    r_outer  = min(ny, nx) / 2 - 1
    yy, xx   = np.ogrid[:ny, :nx]
    r_arr    = np.sqrt((yy - ny//2)**2 + (xx - nx//2)**2)
    field_px = np.sum((r_arr >= r_inner) & (r_arr <= r_outer))
    n_res    = max(1, int(field_px / (np.pi * (fwhm / 2)**2)))

    tpr_arr, fdr_arr, fpr_arr = [], [], []
    for thr in thr_grid:
        try:
            res = compute_binary_map(
                snr_map, thresholds=[thr],
                injections=tuple((p[0], p[1]) for p in positions_xy),
                fwhm=fwhm, npix=2, overlap_threshold=0.7,
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

    # at submitted threshold
    try:
        rt = compute_binary_map(
            snr_map, thresholds=[THRESHOLD],
            injections=tuple((p[0], p[1]) for p in positions_xy),
            fwhm=fwhm, npix=2, overlap_threshold=0.7,
            max_blob_fact=2, debug=False, plot=False)
        nd, nfp, _ = rt; nd, nfp = nd[0], nfp[0]
    except Exception:
        nd, nfp = 0, 0
    prec = nd / (nd + nfp) if (nd + nfp) > 0 else 0.0
    rec  = nd / n_inj if n_inj > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    dt   = thr_grid[1] - thr_grid[0]
    auc_tpr = float(np.trapezoid(tpr_arr, thr_grid) / (thr_grid[-1] - thr_grid[0] + 1e-10))
    auc_fdr = float(np.trapezoid(fdr_arr, thr_grid) / (thr_grid[-1] - thr_grid[0] + 1e-10))

    return {"thr": thr_grid, "tpr": tpr_arr, "fdr": fdr_arr, "fpr": fpr_arr,
            "f1": f1, "auc_tpr": auc_tpr, "auc_fdr": auc_fdr,
            "n_det": nd, "n_fp": nfp}


# ═════════════════════════════════════════════════════════════════════════════
# Main run loop
# ═════════════════════════════════════════════════════════════════════════════

run_meta = {}  # keyed by (inst, idx, variant)

print("=" * 70)
print("PACO Challenge Run – FullPACO vs FastPACO")
print("=" * 70)

for inst, idx in DATASETS:
    label = f"{inst}_{idx}"
    print(f"\n{'─'*60}")
    print(f"Dataset: {label}")
    sys.stdout.flush()

    cube, pa, psf, pxscale = load_dataset(inst, idx)
    print(f"  Loading FWHM …")
    sys.stdout.flush()
    fwhm = compute_fwhm(psf)
    print(f"  Shape: {cube.shape}  FWHM: {fwhm:.2f}px  pxscale: {pxscale:.5f}\"")
    sys.stdout.flush()

    # ── FastPACO ─────────────────────────────────────────────────────────
    fp_path = snr_path(inst, idx, "fast")
    if os.path.exists(fp_path):
        print("  [FastPACO] cached – loading")
        snr_fast = np.load(fp_path)
    else:
        print("  [FastPACO] running …")
        sys.stdout.flush()
        snr_fast, flux_fast, dt = run_fast_paco(cube, pa, psf, fwhm, pxscale)
        np.save(fp_path, snr_fast)
        np.save(snr_path(inst, idx, "fast_flux"), flux_fast)
        print(f"  [FastPACO] done in {dt:.1f}s")
        sys.stdout.flush()
    run_meta[(inst, idx, "fast")] = {"fwhm": fwhm, "pxscale": pxscale}

    # ── FullPACO ──────────────────────────────────────────────────────────
    if (inst, idx) in FULL_PACO_ONLY_FAST:
        print("  [FullPACO] skipped (memory infeasible: LMIRCam T≈4000 or NIRC2 K≈197)")
        sys.stdout.flush()
    else:
        fp2_path = snr_path(inst, idx, "full")
        if os.path.exists(fp2_path):
            print("  [FullPACO] cached – loading")
            snr_full = np.load(fp2_path)
        else:
            print("  [FullPACO] running …")
            sys.stdout.flush()
            snr_full, flux_full, dt = run_full_paco_efficient(
                cube, pa, psf, fwhm, pxscale)
            np.save(fp2_path, snr_full)
            np.save(snr_path(inst, idx, "full_flux"), flux_full)
            print(f"  [FullPACO] done in {dt:.1f}s")
            sys.stdout.flush()
        run_meta[(inst, idx, "full")] = {"fwhm": fwhm, "pxscale": pxscale}

print("\n" + "=" * 70)
print("All algorithm runs complete.")
sys.stdout.flush()

# ═════════════════════════════════════════════════════════════════════════════
# Self-injection test (injection-free datasets: sph2, nirc2_2)
# ═════════════════════════════════════════════════════════════════════════════

INJECT_DSETS = [("sphere_irdis", 2), ("nirc2", 2)]

# (separation_arcsec, pa_deg, contrast). Contrast is instrument-specific and
# calibrated empirically (see challenge/outputs/run_log.txt investigation):
# contrast is measured directly against the per-pixel TEMPORAL NOISE STD at
# the injection separation (not relative to stellar peak flux), because
# nirc2's raw-count calibration differs enormously from sphere_irdis's
# ~unity-normalised cube. Direct FastPACO sweeps at 0.35" on sphere_irdis_2
# showed contrast=1.0 (~0.4x local std=2.4) -> SNR~7.3, contrast=5.0 (~2x std)
# -> SNR~13.5; contrast=10 (~4x std) is used here for margin. nirc2_2's local
# std at 0.35" is ~5536 (much noisier data, consistent with nirc2 scoring
# lower than sphere_irdis for nearly every algorithm in the paper's own
# tables), so a proportionally larger contrast is used.
INJECT_CONFIG_BY_INST = {
    "sphere_irdis": [(0.35, 45, 10.0),    (0.55, 155, 10.0),    (0.75, 270, 10.0)],
    "nirc2":        [(0.35, 45, 60000.0), (0.55, 155, 60000.0), (0.75, 270, 60000.0)],
}

inject_results = {}  # keyed by (inst, idx, variant)

print("\n" + "=" * 70)
print("Self-injection metric test")
print("=" * 70)

for inst, idx in INJECT_DSETS:
    label = f"{inst}_{idx}"
    cube, pa, psf, pxscale = load_dataset(inst, idx)
    fwhm = compute_fwhm(psf)
    T, H, W = cube.shape
    cy, cx = H // 2, W // 2

    # Inject companions
    cube_inj = cube.copy()
    positions_xy = []
    for sep_as, pa_deg, contrast in INJECT_CONFIG_BY_INST[inst]:
        sep_px = sep_as / pxscale
        if sep_px >= min(H, W) / 2 - 2 * fwhm:
            sep_px = 0.3 / pxscale
        cube_inj, pos = inject_companion_into_cube(
            cube_inj, psf, pa, sep_px, pa_deg, contrast)
        positions_xy.append(pos)

    print(f"\n  {label}: injected {len(positions_xy)} companions")
    for p in positions_xy:
        print(f"    ({p[0]:.1f}, {p[1]:.1f}) px")
    sys.stdout.flush()

    for variant in ["fast", "full"]:
        if variant == "full" and (inst, idx) in FULL_PACO_ONLY_FAST:
            continue
        inj_snr_path = os.path.join(SNR_DIR, f"{label}_{variant}_inject_snr.npy")
        inj_met_path = os.path.join(SNR_DIR, f"{label}_{variant}_inject_metrics.npy")

        if os.path.exists(inj_snr_path):
            print(f"  [{variant.upper()}] injection SNR cached")
            snr_inj = np.load(inj_snr_path)
        else:
            print(f"  [{variant.upper()}PACO + injection] running …")
            sys.stdout.flush()
            if variant == "fast":
                snr_inj, _, dt = run_fast_paco(cube_inj, pa, psf, fwhm, pxscale)
            else:
                snr_inj, _, dt = run_full_paco_efficient(cube_inj, pa, psf, fwhm, pxscale)
            np.save(inj_snr_path, snr_inj)
            print(f"  [{variant.upper()}PACO + injection] done in {dt:.1f}s")
            sys.stdout.flush()

        print(f"  Computing metrics for {variant.upper()}PACO …")
        sys.stdout.flush()
        metrics = compute_metric_sweep(snr_inj, positions_xy, fwhm)
        np.save(inj_met_path, {"metrics": metrics, "positions": positions_xy,
                               "fwhm": fwhm}, allow_pickle=True)
        inject_results[(inst, idx, variant)] = {**metrics,
                                                 "positions": positions_xy,
                                                 "fwhm": fwhm}
        print(f"    F1={metrics['f1']:.3f}  AUC_TPR={metrics['auc_tpr']:.3f}  "
              f"AUC_FDR={metrics['auc_fdr']:.3f}  n_det={metrics['n_det']}  "
              f"n_fp={metrics['n_fp']}")
        sys.stdout.flush()

# Save meta for plotting script
np.save(os.path.join(OUT_DIR, "run_meta.npy"),
        {str(k): v for k, v in run_meta.items()}, allow_pickle=True)
np.save(os.path.join(OUT_DIR, "inject_meta.npy"),
        {str(k): {kk: vv for kk, vv in v.items()
                  if kk not in ("thr", "tpr", "fdr", "fpr")}
         for k, v in inject_results.items()},
        allow_pickle=True)

print("\nAll done! Run plot_challenge_results.py to generate figures.")
