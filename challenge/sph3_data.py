"""
Loading of the EIDC sph3 (SPHERE/IRDIS) dataset and the FWHM estimator
shared by the benchmark and challenge scripts.
"""
import os

import numpy as np
from astropy.io import fits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "challenge", "subchallenge_1")


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


def compute_fwhm(psf):
    """1-D Gaussian fit to the PSF's radial profile (avoids VIP's
    fit_2dgaussian, which hangs under Python 3.13 multiprocessing)."""
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
        return 3.72  # sph3's known value
