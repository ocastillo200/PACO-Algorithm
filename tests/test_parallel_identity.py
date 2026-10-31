"""The S/N map must not depend on the number of worker processes, and the
deprecated class names must give the same map as the current ones.

Run: python -m pytest tests/  (or: python tests/test_parallel_identity.py)
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import FastPACO, FastPACOv2, FullPACOv2, PACO

FWHM = 3.0


def _data():
    rng = np.random.default_rng(0)
    T, N = 16, 32
    pa = np.linspace(0.0, 60.0, T)
    y, x = np.mgrid[-5:6, -5:6]
    psf = np.exp(-(x ** 2 + y ** 2) / (2 * (FWHM / 2.355) ** 2))
    cube = rng.standard_normal((T, N, N))
    # A companion, so that the map is not pure noise.
    th = np.deg2rad(pa)
    for t in range(T):
        cy = int(round(N // 2 + 8 * np.sin(th[t])))
        cx = int(round(N // 2 + 8 * np.cos(th[t])))
        cube[t, cy - 5:cy + 6, cx - 5:cx + 6] += 5 * psf
    return cube, pa, psf


def _snr(cls, cpu):
    cube, pa, psf = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        obj = cls(cube, pa, psf, fwhm=FWHM, pixscale=1.0, verbose=False)
    out = obj.run(cpu=cpu)
    return np.asarray(out[0] if isinstance(out, tuple) else out)


def _assert_same(a, b):
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True)


def test_full_paco_serial_equals_parallel():
    _assert_same(_snr(PACO, 1), _snr(PACO, 2))


def test_fast_paco_serial_equals_parallel():
    _assert_same(_snr(FastPACO, 1), _snr(FastPACO, 2))


def test_map_is_finite_somewhere():
    assert np.isfinite(_snr(FastPACO, 1)).any()


def test_deprecated_aliases_warn_and_agree():
    cube, pa, psf = _data()
    for old, new in [(FullPACOv2, PACO), (FastPACOv2, FastPACO)]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = old(cube, pa, psf, fwhm=FWHM, pixscale=1.0, verbose=False)
        assert any(issubclass(x.category, DeprecationWarning) for x in w)
        assert isinstance(obj, new)
        _assert_same(_snr(old, 1), _snr(new, 1))


def test_small_dev_shm_shrinks_cache_not_result():
    # A /dev/shm with room for a few slots only (as in a container): the
    # cache shrinks and the map is unchanged.
    import src.full_paco as fp
    cube, pa, psf = _data()
    obj = PACO(cube, pa, psf, fwhm=FWHM, pixscale=1.0, verbose=False)
    T, K = obj.num_frames, obj.patch_area_pixels
    slot = (T * K + K + K * K) * 8
    tiny = int((5 * slot + obj.height * obj.width * 4) / fp.SHM_SAFETY) + 1
    ref = _snr(PACO, 2)
    orig = fp.shm_headroom
    try:
        fp.shm_headroom = lambda: tiny
        _assert_same(_snr(PACO, 2), ref)
        assert fp._G["capacity"] == 5
        fp.shm_headroom = lambda: 1000
        try:
            _snr(PACO, 2)
        except MemoryError:
            pass
        else:
            raise AssertionError("expected MemoryError for a full /dev/shm")
    finally:
        fp.shm_headroom = orig

def _paco_ab(cpu, **kw):
    cube, pa, psf = _data()
    init = {k: kw.pop(k) for k in ("memory_fraction",) if k in kw}
    obj = PACO(cube, pa, psf, fwhm=FWHM, pixscale=1.0, verbose=False, **init)
    N = cube.shape[1]
    y, x = np.mgrid[:N, :N]
    phi0s = np.column_stack((y.ravel(), x.ravel())).astype(float)
    a, b = obj.PACOCalc(phi0s, cpu=cpu, use_gpu=False, **kw)
    return a, b, obj


def test_ring_order_equals_single_pass():
    # One bin wider than the field = no ring resets: same a, b.
    for cpu in (1, 2):
        a1, b1, _ = _paco_ab(cpu)
        a2, b2, _ = _paco_ab(cpu, radius_bin_width=1e9)
        _assert_same(a1, a2)
        _assert_same(b1, b2)


def test_paco_matches_serial_reference():
    # The cached, ring-ordered sweep against the base class, which
    # precomputes every pixel's statistics.
    from src import PACOBase
    cube, pa, psf = _data()
    ref = PACOBase(cube, pa, psf, fwhm=FWHM, pixscale=1.0, verbose=False).run(cpu=1)
    _assert_same(_snr(PACO, 2), np.asarray(ref[0] if isinstance(ref, tuple) else ref))


def test_tiny_budget_bounds_cache_not_result():
    # A budget too small for more than one cache slot: the cache shrinks,
    # misses rise, and a, b are unchanged.
    import src.full_paco as fp
    a1, b1, big = _paco_ab(2)
    a2, b2, small = _paco_ab(2, memory_fraction=1e-9)
    assert fp._G["capacity"] == 1
    assert small.last_cache_stats["misses"] > big.last_cache_stats["misses"]
    _assert_same(a1, a2)
    _assert_same(b1, b2)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
