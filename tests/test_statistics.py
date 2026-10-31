"""Tests for the patch statistics of src.math_utils against the published
estimator (Flasseur et al. 2018, Eqs. 7-12; Eq. 1 of the paper).

Run: python -m pytest tests/  (or: python tests/test_statistics.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.math_utils import (compute_statistics_at_pixel, compute_statistics_batch,
                            diagsample_covariance, shrinkage_factor)


def _rho_reference(S, T):
    """Eq. 1 written out term by term, with sum_i S_ii^2 over the diagonal."""
    K = S.shape[0]
    tr_S2 = sum(S[i, j] * S[j, i] for i in range(K) for j in range(K))
    tr_S = sum(S[i, i] for i in range(K))
    diag_sq = sum(S[i, i] ** 2 for i in range(K))
    rho = (tr_S2 + tr_S ** 2 - 2 * diag_sq) / ((T + 1) * (tr_S2 - diag_sq))
    return min(max(rho, 0.0), 1.0)


def _statistics_reference(patch):
    """Mean, ddof=0 sample covariance (1/T), shrinkage, inverse."""
    T, K = patch.shape
    m = patch.sum(axis=0) / T
    S = np.zeros((K, K))
    for t in range(T):
        d = patch[t] - m
        S += np.outer(d, d)
    S /= T
    rho = _rho_reference(S, T)
    C = (1 - rho) * S + rho * np.diag(np.diag(S))
    return m, np.linalg.inv(C), rho


def _patches(rng, n, T, K, corr=0.6):
    # Correlated columns, so that rho is strictly inside (0, 1).
    A = np.eye(K) + corr * rng.standard_normal((K, K)) / np.sqrt(K)
    return rng.standard_normal((n, T, K)) @ A.T


def test_rho_matches_eq1():
    rng = np.random.default_rng(1)
    for T, K in [(40, 9), (120, 29), (20, 29)]:
        for patch in _patches(rng, 5, T, K):
            S = np.cov(patch, rowvar=False, ddof=0)
            got = shrinkage_factor(S, T)
            want = _rho_reference(S, T)
            assert abs(got - want) < 1e-12, (T, K, got, want)


def test_rho_strictly_inside_unit_interval():
    # Guards against a test that passes only because of the clipping.
    rng = np.random.default_rng(2)
    S = np.cov(_patches(rng, 1, 120, 29)[0], rowvar=False, ddof=0)
    assert 0.0 < shrinkage_factor(S, 120) < 1.0


def test_rho_differs_from_full_matrix_sum():
    # The sum in Eq. 1 is over the diagonal only; summing S_ij^2 over the
    # whole matrix (as VIP does) gives a different weight.
    rng = np.random.default_rng(3)
    T = 120
    S = np.cov(_patches(rng, 1, T, 29)[0], rowvar=False, ddof=0)
    tr_S2, tr_S, full_sq = np.trace(S @ S), np.trace(S), np.sum(S ** 2)
    wrong = (tr_S2 + tr_S ** 2 - 2 * full_sq) / ((T + 1) * (tr_S2 - full_sq))
    assert abs(shrinkage_factor(S, T) - min(max(wrong, 0), 1)) > 1e-3


def test_rho_clipping_and_diagonal_case():
    T = 30
    # Diagonal S: denominator zero, rho = 1 (C = F = S).
    assert shrinkage_factor(np.diag([1.0, 2.0, 3.0]), T) == 1.0
    # Few frames, strong correlation: raw value above 1 is clipped.
    rng = np.random.default_rng(4)
    S = np.cov(rng.standard_normal((3, 25)), rowvar=False, ddof=0)
    assert 0.0 <= shrinkage_factor(S, 3) <= 1.0


def test_shrinkage_target_is_diagonal():
    S = np.arange(9.0).reshape(3, 3)
    assert np.array_equal(diagsample_covariance(S), np.diag([0.0, 4.0, 8.0]))


def test_statistics_at_pixel_matches_reference():
    rng = np.random.default_rng(5)
    for patch in _patches(rng, 5, 60, 13):
        m, Cinv = compute_statistics_at_pixel(patch)
        m_ref, Cinv_ref, _ = _statistics_reference(patch)
        np.testing.assert_allclose(m, m_ref, rtol=0, atol=1e-12)
        np.testing.assert_allclose(Cinv, Cinv_ref, rtol=1e-10, atol=1e-10)


def test_sample_covariance_is_not_halved():
    # C must scale with the data variance as the 1/T sample covariance does:
    # for white noise of variance s2, diag(C) ~ s2, not s2 / 2.
    rng = np.random.default_rng(6)
    s2 = 4.0
    patch = np.sqrt(s2) * rng.standard_normal((20000, 5))
    _, Cinv = compute_statistics_at_pixel(patch)
    np.testing.assert_allclose(np.diag(np.linalg.inv(Cinv)), s2, rtol=0.05)


def test_batch_matches_per_pixel():
    rng = np.random.default_rng(7)
    patches = _patches(rng, 8, 50, 13)
    patches[3] = np.nan  # an out-of-bounds patch
    m, Cinv = compute_statistics_batch(patches)
    for i, patch in enumerate(patches):
        if i == 3:
            assert np.all(np.isnan(m[i])) and np.all(np.isnan(Cinv[i]))
            continue
        m_i, Cinv_i = compute_statistics_at_pixel(patch)
        np.testing.assert_allclose(m[i], m_i, rtol=0, atol=1e-12)
        np.testing.assert_allclose(Cinv[i], Cinv_i, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
