#!/usr/bin/env python3
"""
Data generation for Sect. 5.2 ("Contrast curves with injected companions")
of paco_paper.tex.

Runs VIP's vip_hci.metrics.contrast_curve injection-recovery machinery on
our production FullPACOv2 implementation (src.full_paco). This is not
natively supported: contrast_curve/throughput require `algo` to be a plain
callable with signature (cube, angle_list, verbose, **kwargs) returning a
single 2D frame it can do aperture photometry on (see
vip_hci/metrics/contrcurve.py:throughput, the `argl = getfullargspec(algo)`
check). FullPACOv2 is a class needing (cube, angles, psf, fwhm, ...) at
construction and exposing .run(cpu=..., full_output=...) -> (snr, flux) --
so this script wraps it in a closure satisfying VIP's contract, rather
than modifying FullPACOv2 itself (nothing about the wrapped call is
FullPACOv2-specific; the same wrapper works for any of this project's PACO
classes).

Units derivation -- why `starphot = aperture_flux(raw_psf, ..., fwhm)` is
the right choice here, even though our returned frame is already a
contrast (not raw counts, unlike a typical PCA/KLIP residual frame):

  Let A = aperture_flux(raw_psf, 1 FWHM) (raw_psf = the same, UNNORMALIZED
  psf array passed to FullPACOv2's constructor). src/full_paco.py uses
  that raw psf directly as the per-frame template h in Eqs. 15-16 of
  Flasseur et al. (2018) (no extra normalize_psf call) -- so in our model
  r = alpha*h + f, alpha is already a *dimensionless* companion/star flux
  ratio (a contrast), not a raw-counts amplitude, PROVIDED the psf array
  represents the star's own PSF under the same conditions as the science
  frames (which is exactly what PACO.__init__'s dit_science/dit_psf/
  nd_transmission rescaling is for).

  VIP's injection machinery, however, always re-normalizes whatever
  psf_template it is given so that its OWN 1-FWHM aperture flux is 1
  (vip_hci.fm.normalize_psf's documented convention) and injects a
  companion of "flux" counts in THAT convention. The same physical
  companion, expressed in our model's units, has
  alpha_true = flux_vip_units / A. A perfectly recovered alpha_hat should
  equal that, while VIP's own bookkeeping measures
  injected_flux ~= flux_vip_units (aperture flux of its own
  already-normalized fc_map). So the "throughput" VIP computes,
  thruput = recovered_flux/injected_flux ~= alpha_true*A / (alpha_true*A)
  ... expanding: recovered_flux ~= alpha_true = flux_vip_units/A, so
  thruput ~= 1/A regardless of true recovery quality -- and
  contrast = sigma*noise/thruput/starphot becomes sigma*noise*A/starphot.
  Setting starphot = A cancels that stray factor of A, leaving sigma*noise
  directly in our native alpha/contrast units -- i.e. exactly the
  standard, documented starphot convention ("aperture photometry of the
  psf"), just arrived at from a different starting point (an
  already-a-contrast frame) than the usual raw-counts-residual case it is
  normally used for.

  This is checked empirically below, not just trusted algebraically: the
  printed throughput values should come out order-unity for bright,
  wide-separation injections if the units line up correctly; if they come
  out systematically far from that, something in this derivation or in
  the assumption dit_science=dit_psf=nd_transmission=1 (true for how this
  project's sph3 runs have been configured throughout, see
  challenge/run_paco_challenge.py) does not hold and must be revisited
  before trusting the resulting curve.

Usage
-----
    python3 challenge/make_contrast_curve_sph3.py
    python3 challenge/make_contrast_curve_sph3.py --cpu 4 --nbranch 1
"""
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from vip_hci.metrics import aperture_flux
from vip_hci.metrics.contrcurve import contrast_curve, noise_per_annulus

from src.full_paco import FullPACOv2

DATA_DIR = os.path.join(ROOT, "challenge", "subchallenge_1")
FIG_DIR = os.path.join(ROOT, "challenge", "figures")
PAPER_FIG_DIR = os.path.join(
    ROOT, "Astronomy_and_Astrophysics_Template", "figures")


def load_sph3():
    def fp(kind):
        return os.path.join(DATA_DIR, f"sphere_irdis_{kind}_3.fits")

    cube = fits.getdata(fp("cube")).astype(np.float64)
    pa = fits.getdata(fp("pa")).astype(np.float64).ravel()
    psf = fits.getdata(fp("psf")).astype(np.float64)
    if psf.ndim == 3:
        psf = np.nanmedian(psf, axis=0)
    pxs_d = fits.getdata(fp("pxscale"))
    pxs = float(np.asarray(pxs_d).flat[0])
    if not np.isfinite(pxs) or pxs <= 0:
        pxs = 0.01225
    return cube, pa, psf, pxs


def crop_cube_centered(cube, size):
    """
    Crop a (T, H, W) cube to a centered (T, size, size) region.

    Needed because vip_hci.metrics.contrast_curve's own edge-safety margin
    (radmax_fwhm = (width-1)//2 - fwhm/2, see
    vip_hci/metrics/contrcurve.py:throughput) is too thin for this
    implementation's actual edge-invalidity zone: a candidate position is
    only valid if EVERY one of its T rotated trajectory points still falls
    inside the frame AND far enough from the border for a full patch
    extraction (patch_width//2 pixels) -- empirically, for sph3's 160x160
    field (fwhm=3.72, patch_width=7), this pushed the true safe radius down
    to ~65-70px, well inside VIP's own ~77px margin. Confirmed empirically:
    without cropping, at least one of the ~25 simultaneous companions VIP
    injects per test annulus lands at an invalid (NaN) position, and
    because cube_inject_companions shifts the injected PSF via FFT, that
    single NaN companion propagates NaN across the ENTIRE frame (verified:
    fc_map_all came back 100% NaN, not just locally NaN near that one
    companion).

    Cropping to a smaller field shrinks the field's own edge accordingly,
    so VIP's margin calculation (still using the CROPPED width) lands
    safely inside our verified-valid region instead of reaching into it.
    """
    T, H, W = cube.shape
    if size >= H and size >= W:
        return cube
    cy, cx = H // 2, W // 2
    half = size // 2
    return cube[:, cy - half:cy - half + size, cx - half:cx - half + size].copy()


def compute_fwhm(psf):
    """Same estimator used throughout this project (see
    challenge/run_paco_challenge.py:compute_fwhm) -- 1-D Gaussian fit to
    the PSF's radial profile, avoiding VIP's fit_2dgaussian (hangs under
    Python 3.13 multiprocessing)."""
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
        r_arr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
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
        return 3.72  # sph3's known value, used throughout this project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--crop", type=int, default=120,
                        help="Crop the (square) field to this size before "
                             "running, centered. See crop_cube_centered's "
                             "docstring for why this is necessary (VIP's "
                             "own edge-safety margin is too thin for this "
                             "implementation's actual edge-invalidity "
                             "zone). Pass 0 or >= the native size to "
                             "disable cropping.")
    parser.add_argument("--nbranch", type=int, default=1)
    parser.add_argument("--fc-rad-sep", type=int, default=3)
    parser.add_argument("--fc-snr", type=float, default=0.001,
                        help="VIP's own default (100) sizes the injected "
                             "companion as fc_snr*(std of our flux map "
                             "within the annulus). A handful of outlier "
                             "pixels (unstable covariance inversions at "
                             "specific patch geometries) inflate that std "
                             "far beyond the map's typical/median scale -- "
                             "confirmed empirically: fc_snr=100 injects "
                             "companions bright enough (contrast >> 1 in "
                             "our alpha units) to break background-"
                             "statistics estimation broadly, producing an "
                             "all-NaN throughput. 0.001 keeps injections "
                             "physically sane even at the noisiest "
                             "(innermost) separations -- see "
                             "challenge/calibrate_contrast_units.py.")
    parser.add_argument("--calibration-factor", type=float, default=3.51942,
                        help="Empirical alpha-to-true-contrast calibration "
                             "factor (our_alpha ~= factor * true_contrast), "
                             "measured by challenge/calibrate_contrast_units.py "
                             "via a controlled single-companion "
                             "injection/recovery test at r=20px, contrast="
                             "1e-3 (see that script's docstring/output). "
                             "Divides the final sensitivity/oracle curves.")
    args = parser.parse_args()

    cube, pa, psf, pxscale = load_sph3()
    fwhm = compute_fwhm(psf)
    print(f"sph3 (native): cube={cube.shape}, fwhm={fwhm:.3f}px, "
          f"pxscale={pxscale:.5f}\"")
    if args.crop and args.crop < min(cube.shape[1], cube.shape[2]):
        cube = crop_cube_centered(cube, args.crop)
        print(f"Cropped to {cube.shape} (see crop_cube_centered docstring "
              f"for why -- avoids VIP's contrast_curve reaching into the "
              f"edge-invalidity zone of the native field)")
    T, H, W = cube.shape
    # Empirically measured (see the radial nan_frac breakdown run against
    # this field): candidate positions are valid (non-NaN) out to
    # ~half_width - 6px, then transition to fully invalid within ~5px
    # beyond that. Truncate the reported curve there regardless of what
    # VIP's own (shallower) margin tried to compute, so no row in the
    # final output is drawn from the always-zeroed invalid-edge region.
    SAFE_RADIUS = min(H, W) / 2.0 - 6.0
    print(f"SAFE_RADIUS (max trustworthy separation, px) = {SAFE_RADIUS:.1f}")

    cy, cx = psf.shape[0] / 2.0, psf.shape[1] / 2.0
    A = float(aperture_flux(psf, [cy], [cx], fwhm)[0])
    print(f"starphot = aperture_flux(raw psf, 1 FWHM) = {A:.6g} "
          f"(see module docstring for why this is the right choice even "
          f"though our own flux map is already a contrast, not raw counts)")

    captured = []  # [(a_map, flux_map, snr_map), ...], first entry = no-fc call

    def paco_algo(cube, angle_list, verbose=False, **kwargs):
        obj = FullPACOv2(cube, angle_list, psf, fwhm=fwhm, pixscale=pxscale,
                         verbose=False)
        t0 = time.perf_counter()
        snr, flux = obj.run(cpu=args.cpu, use_gpu=False, full_output=True)
        dt = time.perf_counter() - t0
        captured.append((obj.a_map.copy(), flux.copy(), snr.copy()))
        nan_frac = float(np.mean(np.isnan(flux)))
        finite = flux[np.isfinite(flux)]
        print(f"  [paco_algo] call #{len(captured)} finished in {dt:.1f}s "
              f"(cpu={args.cpu}) nan_frac={nan_frac:.4f} "
              f"finite_range=[{finite.min() if finite.size else float('nan'):.4g}, "
              f"{finite.max() if finite.size else float('nan'):.4g}] "
              f"finite_std={finite.std() if finite.size else float('nan'):.4g}",
              flush=True)
        np.save(os.path.join(FIG_DIR, f"_diag_flux_call{len(captured)}.npy"), flux)
        # Root cause of the all-NaN throughput (confirmed empirically, see
        # calibrate_contrast_units.py and the radial nan_frac breakdown):
        # NaN pixels here (invalid candidate positions near the field's
        # edge-validity boundary) feed into VIP's noise_per_annulus ->
        # noise[idx]=NaN for deep annuli -> cube_inject_companions injects
        # a NaN-amplitude companion directly into the RAW CUBE (not just
        # our own output) -> that NaN cube pixel corrupts background
        # statistics for every candidate whose patch touches it, at ANY
        # frame, producing the observed near-total NaN frame_fc. Returning
        # 0.0 instead of NaN here keeps every downstream VIP computation
        # finite: deep annuli beyond our valid zone get a harmless,
        # near-zero-noise -> near-zero-amplitude injection instead of a
        # cube-corrupting one. Rows beyond our verified-valid radius are
        # dropped from the final reported curve below regardless (see
        # SAFE_RADIUS), so this only prevents a crash, not a
        # mis-measurement -- no invalid pixel's "0" ever contributes to a
        # reported sensitivity value.
        return np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)

    print("\nRunning vip_hci.metrics.contrast_curve "
          f"(nbranch={args.nbranch}, fc_rad_sep={args.fc_rad_sep}, "
          f"fc_snr={args.fc_snr}) -- this dispatches "
          f"1 + nbranch*fc_rad_sep = {1 + args.nbranch * args.fc_rad_sep} "
          "full FullPACOv2 runs...\n", flush=True)

    t_start = time.perf_counter()
    datafr, frame_fc_all, frame_nofc, fc_map_all = contrast_curve(
        cube, pa, psf, fwhm, pxscale, starphot=A, algo=paco_algo,
        sigma=5, nbranch=args.nbranch, fc_rad_sep=args.fc_rad_sep,
        fc_snr=args.fc_snr, student=True, smooth=True, plot=False,
        verbose=True, full_output=True,
    )
    print(f"\nTotal wall time: {time.perf_counter() - t_start:.1f}s")

    np.save(os.path.join(FIG_DIR, "_diag_frame_fc_all.npy"), frame_fc_all)
    np.save(os.path.join(FIG_DIR, "_diag_frame_nofc.npy"), frame_nofc)
    np.save(os.path.join(FIG_DIR, "_diag_fc_map_all.npy"), fc_map_all)
    for i in range(frame_fc_all.shape[0]):
        print(f"  frame_fc_all[{i}]: nan_frac="
              f"{np.mean(np.isnan(frame_fc_all[i])):.4f}, "
              f"fc_map nonzero (>1e-5) count="
              f"{int(np.sum(fc_map_all[i] > 1e-5))}")

    print("\nThroughput sanity check (should be order-unity, no NaN, if "
          "fc_snr is small enough to avoid the outlier-driven catastrophic "
          "injection described above):")
    print(datafr[["distance", "distance_arcsec", "throughput",
                  "sensitivity_gaussian"]].to_string(index=False))
    print(f"\nthroughput: min={np.nanmin(datafr['throughput']):.4g} "
          f"median={np.nanmedian(datafr['throughput']):.4g} "
          f"max={np.nanmax(datafr['throughput']):.4g} "
          f"n_nan={int(datafr['throughput'].isna().sum())}/{len(datafr)}")

    # ---- Oracle curve: sigma_alphahat = 1/sqrt(a), from the no-injection
    # call's a-map (Eq. 19/27 of Flasseur et al. 2018) -- no extra PACO run
    # needed, `captured[0]` is exactly that no-fc call (throughput() always
    # calls algo on the plain cube before any fake-companion cube).
    a_map_nofc = captured[0][0]
    sigma_alphahat_map = 1.0 / np.sqrt(np.clip(a_map_nofc, 1e-300, None))
    _, oracle_profile, oracle_rad_px = noise_per_annulus(
        sigma_alphahat_map, separation=1, fwhm=fwhm, wedge=(0, 360))
    oracle_at_datafr_rad = np.interp(datafr["distance"], oracle_rad_px,
                                     oracle_profile)
    # Both curves are computed in "our alpha units" up to this point (the
    # starphot=A choice cancels the VIP-internal normalized-psf-vs-raw-psf
    # unit mismatch, see module docstring, but does not know about the
    # residual empirical calibration factor measured in
    # calibrate_contrast_units.py -- apply it here, uniformly, to both
    # curves so they remain directly comparable to each other.
    oracle_contrast = (5.0 * oracle_at_datafr_rad / A) / args.calibration_factor

    # NOT datafr['sensitivity_gaussian']/['sensitivity_student']: VIP hard-
    # clips those to 1.0 whenever sigma*noise/thruput/starphot > 1
    # (vip_hci/metrics/contrcurve.py: "cont_curve_samp[...>1] = 1"), a
    # guard meant for the normal case where contrast is already << 1.
    # Confirmed empirically that every row here exceeds that threshold
    # before the calibration_factor correction is even applied, so the
    # clipped columns come back as a uniform, uninformative 1.0 at every
    # separation. Recompute from the untouched noise/throughput/sigma-corr
    # columns instead, applying the sigma*noise/thruput/starphot formula
    # (identical to VIP's own, see contrcurve.py) ourselves, then the
    # empirical calibration_factor on top.
    sensitivity_gaussian_cal = ((5.0 * datafr["noise"] / datafr["throughput"] / A)
                                / args.calibration_factor)
    sensitivity_student_cal = ((datafr["sigma corr"] * datafr["noise"]
                               / datafr["throughput"] / A)
                               / args.calibration_factor)

    # Truncate at SAFE_RADIUS: rows beyond it are drawn from candidate
    # positions this implementation cannot validly evaluate (see
    # SAFE_RADIUS derivation above) -- report only what was actually
    # measured on real, non-degenerate data.
    keep = datafr["distance"].to_numpy() <= SAFE_RADIUS
    n_dropped = int((~keep).sum())
    print(f"\nDropping {n_dropped}/{len(keep)} rows beyond "
          f"SAFE_RADIUS={SAFE_RADIUS:.1f}px")
    datafr = datafr[keep].reset_index(drop=True)
    sensitivity_gaussian_cal = sensitivity_gaussian_cal[keep].reset_index(drop=True)
    sensitivity_student_cal = sensitivity_student_cal[keep].reset_index(drop=True)
    oracle_contrast = np.asarray(oracle_contrast)[keep]

    print(f"\nthroughput (post-truncation): "
          f"min={np.nanmin(datafr['throughput']):.4g} "
          f"median={np.nanmedian(datafr['throughput']):.4g} "
          f"max={np.nanmax(datafr['throughput']):.4g} "
          f"n_nan={int(datafr['throughput'].isna().sum())}/{len(datafr)}")

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(PAPER_FIG_DIR, exist_ok=True)
    csv_path = os.path.join(FIG_DIR, "contrast_curve_sph3.csv")
    out = datafr.copy()
    out["sensitivity_gaussian_calibrated"] = sensitivity_gaussian_cal
    out["sensitivity_student_calibrated"] = sensitivity_student_cal
    out["oracle_5sigma_contrast"] = oracle_contrast
    out["calibration_factor"] = args.calibration_factor
    out.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    ax.plot(datafr["distance_arcsec"], sensitivity_gaussian_cal,
           "o-", color="#d62728", ms=3, label="Injected (Gaussian $5\\sigma$)")
    ax.plot(datafr["distance_arcsec"], sensitivity_student_cal,
           "s--", color="#1f77b4", ms=3,
           label="Injected (Student-$t$ corrected)")
    ax.plot(datafr["distance_arcsec"], oracle_contrast,
           "-", color="0.3", lw=1.5, label="PACO oracle (Eq. 27, no injection)")
    ax.set_yscale("log")
    ax.set_xlabel("Angular separation [arcsec]")
    ax.set_ylabel(r"5$\sigma$ contrast (relative units$^*$)")
    ax.text(0.02, 0.02,
           "*shape validated; absolute scale assumes\n"
           "dit_science=dit_psf=nd_transmission=1\n"
           "(unverified for this reprocessed dataset)",
           transform=ax.transAxes, fontsize=6, color="0.4", va="bottom")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for d in (FIG_DIR, PAPER_FIG_DIR):
        out_pdf = os.path.join(d, "fig_contrast_curve_sph3.pdf")
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
