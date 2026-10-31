"""
Statistical and linear-algebra helpers used by the PACO algorithms.

All functions are pure (no class state) and can be used independently.
"""
from typing import Optional, Tuple, Union

import numpy as np

from .gpu import GPU_AVAILABLE, cp


# ---------------------------------------------------------------------------
# Patch statistics
# ---------------------------------------------------------------------------

def compute_statistics_at_pixel(patch: np.ndarray) -> Tuple[Optional[np.ndarray],
                                                              Optional[np.ndarray]]:
    """
    Compute mean and shrinkage-regularised inverse covariance for a temporal
    patch column (Equations 7 and 12 of Flasseur et al. 2018).

    Parameters
    ----------
    patch : np.ndarray, shape (T, K)
        Stack of T flattened patch pixels across time frames.

    Returns
    -------
    m : np.ndarray of shape (K,) or None
        Temporal mean of the patch column.
    Cinv : np.ndarray of shape (K, K) or None
        Inverse of the shrinkage-regularised sample covariance. Returns None
        when the patch is all-NaN or has fewer than 2 frames.
    """
    if patch is None or np.all(np.isnan(patch)):
        return None, None

    T, K = patch.shape

    if T < 2:
        return np.mean(patch, axis=0), np.full((K, K), np.nan)

    m = np.mean(patch, axis=0)
    S = np.cov(patch, rowvar=False, ddof=0)

    if np.any(np.isnan(S)):
        return m, np.full((K, K), np.nan)

    rho = shrinkage_factor(S, T)
    F = diagsample_covariance(S)
    C = (1.0 - rho) * S + rho * F

    try:
        Cinv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        Cinv = np.linalg.pinv(C)

    return m, Cinv


def compute_statistics_batch(patches: Union[np.ndarray, "cp.ndarray"],
                             use_gpu: bool = False
                             ) -> Tuple[Union[np.ndarray, "cp.ndarray"],
                                       Union[np.ndarray, "cp.ndarray"]]:
    """
    Vectorised batch version of :func:`compute_statistics_at_pixel`: computes
    (m, Cinv) for N independent temporal patch columns in one call instead of
    N Python-level calls.

    Same equations (7 and 12 of Flasseur et al. 2018) as the per-pixel
    version, just batched over a leading N axis. This is what makes the
    computation worth moving to a GPU: dispatching one (T, K)-sized
    covariance + inversion per pixel is dominated by Python/kernel-launch
    overhead, not by the actual FLOPs (K is typically a few dozen). Batching
    N pixels turns that into a handful of large BLAS/cuBLAS calls
    (einsum-based batched covariance, batched matrix inversion), which is
    where a GPU's throughput advantage actually shows up. On CPU
    (use_gpu=False) this is also a legitimate, independent speedup over the
    per-pixel Python loop (fewer, larger NumPy calls) — used to validate the
    vectorised math locally without needing CUDA hardware, and usable as a
    plain CPU batching optimisation on its own.

    Parameters
    ----------
    patches : array, shape (N, T, K)
        N independent temporal patch columns (T frames, K pixels each). Can
        already be a CuPy array (e.g. after a caller-managed
        `cp.asarray(...)` transfer) when use_gpu=True — this function does
        not perform any host<->device transfer itself (see
        `src.gpu.GPUStatisticsBatcher` for batch-wise transfer + VRAM
        budgeting).
    use_gpu : bool
        Use CuPy array operations. Falls back to NumPy automatically if CuPy
        isn't available (see `src.gpu.GPU_AVAILABLE`).

    Returns
    -------
    m : array, shape (N, K)
    Cinv : array, shape (N, K, K)
        Rows corresponding to a patch containing ANY NaN (out-of-bounds
        edge pixels, see PACOBase.get_patch) are filled with NaN, matching
        compute_statistics_at_pixel's None-for-invalid convention pixel by
        pixel (an all-finite patch there always implies an all-finite (m,
        Cinv); a patch with any NaN there always propagates to a NaN mean/
        covariance — so "any NaN in the patch" and "invalid" are exactly
        equivalent for every input compute_statistics_at_pixel is actually
        called with in this codebase, i.e. output of PACOBase.get_patch, which
        never returns a *partially* NaN patch).
    """
    xp = cp if use_gpu and GPU_AVAILABLE else np
    N, T, K = patches.shape

    m = xp.full((N, K), xp.nan, dtype=xp.float64)
    Cinv = xp.full((N, K, K), xp.nan, dtype=xp.float64)

    if N == 0:
        return m, Cinv

    flat = patches.reshape(N, T * K)
    valid = ~xp.any(xp.isnan(flat), axis=1)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return m, Cinv

    vp = patches[valid]  # (n_valid, T, K)

    if T < 2:
        m[valid] = xp.mean(vp, axis=1)
        return m, Cinv  # Cinv stays NaN, matching the per-pixel T < 2 case

    mv = xp.mean(vp, axis=1)  # (n_valid, K)
    centered = vp - mv[:, None, :]
    # Batched sample covariance, ddof=0 (matches np.cov(..., ddof=0)).
    S = xp.einsum('ntk,ntl->nkl', centered, centered) / T

    diag_S = xp.diagonal(S, axis1=1, axis2=2)  # (n_valid, K)
    trace_S_sq = xp.einsum('nkl,nlk->n', S, S)
    trace_S = xp.sum(diag_S, axis=1)
    sum_diag_S_sq = xp.sum(diag_S ** 2.0, axis=1)

    top = trace_S_sq + trace_S ** 2 - 2.0 * sum_diag_S_sq
    bot = (T + 1.0) * (trace_S_sq - sum_diag_S_sq)
    rho = xp.where(bot == 0, xp.ones_like(top), top / xp.where(bot == 0, xp.ones_like(bot), bot))
    rho = xp.clip(rho, 0.0, 1.0)

    eye = xp.eye(K, dtype=S.dtype)
    F = S * eye[None, :, :]  # diag(diag(S)), batched (Eq. 12's shrinkage target)
    C = (1.0 - rho)[:, None, None] * S + rho[:, None, None] * F

    Cinv_valid = _batched_inv_with_pinv_fallback(C, xp)

    m[valid] = mv
    Cinv[valid] = Cinv_valid
    return m, Cinv


def _batched_inv_with_pinv_fallback(C: Union[np.ndarray, "cp.ndarray"], xp) -> Union[np.ndarray, "cp.ndarray"]:
    """
    Batched matrix inverse, falling back to a per-matrix pinv only for
    matrices that are actually singular (mirrors compute_statistics_at_pixel's
    inv-then-pinv-on-failure behaviour, without paying pinv's SVD cost for
    every matrix in the — common — well-conditioned case).

    Deliberately catches *any* exception from the batched xp.linalg.inv call,
    not just LinAlgError: this also covers a CuPy version too old to support
    N-D batched inv (falls back to inverting one matrix at a time, still
    correct, just without the batched speedup).
    """
    try:
        return xp.linalg.inv(C)
    except Exception:
        pass
    Cinv = xp.empty_like(C)
    for i in range(C.shape[0]):
        try:
            Cinv[i] = xp.linalg.inv(C[i])
        except Exception:
            Cinv[i] = xp.linalg.pinv(C[i])
    return Cinv


def shrinkage_factor(S: Union[np.ndarray, "cp.ndarray"],
                     T: int,
                     use_gpu: bool = False) -> float:
    """
    Oracle approximating shrinkage (OAS) regularisation factor rho.

    Clips the result to [0, 1] to guarantee a valid convex combination.

    Parameters
    ----------
    S : array of shape (K, K)
        Sample covariance matrix.
    T : int
        Number of temporal frames (samples).
    use_gpu : bool
        Use CuPy instead of NumPy for matrix operations.

    Returns
    -------
    float
        Shrinkage coefficient rho in [0, 1].
    """
    xp = cp if use_gpu and GPU_AVAILABLE else np

    trace_S_sq = xp.trace(xp.dot(S, S))
    trace_S = xp.trace(S)
    sum_diag_S_sq = xp.sum(xp.diag(S) ** 2.0)

    top = trace_S_sq + trace_S ** 2 - 2.0 * sum_diag_S_sq
    bot = (T + 1.0) * (trace_S_sq - sum_diag_S_sq)

    p = 1.0 if bot == 0 else top / bot

    if use_gpu and GPU_AVAILABLE:
        p = float(cp.asnumpy(p))

    return max(min(p, 1.0), 0.0)


def diagsample_covariance(S: Union[np.ndarray, "cp.ndarray"],
                          use_gpu: bool = False) -> Union[np.ndarray, "cp.ndarray"]:
    """
    Return the diagonal matrix whose entries equal the diagonal of S.

    This is the shrinkage target F in Equation 12 of Flasseur et al. 2018:
    replacing off-diagonal entries with zero to reduce estimation error.

    Parameters
    ----------
    S : array of shape (K, K)
        Input covariance matrix.
    use_gpu : bool
        Use CuPy instead of NumPy.

    Returns
    -------
    array of shape (K, K)
        Diagonal matrix diag(diag(S)).
    """
    xp = cp if use_gpu and GPU_AVAILABLE else np
    return xp.diag(xp.diag(S))


# ---------------------------------------------------------------------------
# PACO inner-product helpers  (w, a, b terms from Equations 13–16 of [FLA18])
# ---------------------------------------------------------------------------

def wfl(hfl: np.ndarray,
        Cfl_inv: np.ndarray,
        use_gpu: bool = False) -> np.ndarray:
    """
    Compute the whitened template vector w_l = C_l^{-1} h_l for each frame l.

    Parameters
    ----------
    hfl : array-like, length L
        List/array of template vectors h_l, one per frame.
    Cfl_inv : array-like, length L
        List/array of inverse covariance matrices C_l^{-1}, one per frame.
    use_gpu : bool
        Use CuPy for matrix multiplication.

    Returns
    -------
    np.ndarray, length L
        Whitened template vectors.
    """
    xp = cp if use_gpu and GPU_AVAILABLE else np
    return xp.array([Cfl_inv[i] @ hfl[i] for i in range(len(hfl))])


def al(hfl: np.ndarray,
       wfl_arr: np.ndarray,
       use_gpu: bool = False) -> float:
    """
    Compute the normalization scalar a = sum_l h_l^T w_l (Equation 15 of [FLA18]).

    Parameters
    ----------
    hfl : array-like, length L
        Template vectors h_l.
    wfl_arr : array-like, length L
        Whitened template vectors w_l (output of :func:`wfl`).
    use_gpu : bool
        Use CuPy.

    Returns
    -------
    float
        Scalar a value.
    """
    xp = cp if use_gpu and GPU_AVAILABLE else np
    result = xp.sum([hfl[i] @ wfl_arr[i].T for i in range(len(hfl))], axis=0)
    return float(cp.asnumpy(result)) if use_gpu and GPU_AVAILABLE else result


def bl(wfl_arr: np.ndarray,
       r_fl: np.ndarray,
       m_fl: np.ndarray,
       use_gpu: bool = False) -> float:
    """
    Compute the detection statistic b = sum_l w_l^T (r_l - m_l) (Equation 16 of [FLA18]).

    Parameters
    ----------
    wfl_arr : array-like, length L
        Whitened template vectors w_l.
    r_fl : array-like, length L
        Observed patches r_l (one per frame).
    m_fl : array-like, length L
        Background mean patches m_l.
    use_gpu : bool
        Use CuPy.

    Returns
    -------
    float
        Scalar b value.
    """
    xp = cp if use_gpu and GPU_AVAILABLE else np
    result = xp.sum([wfl_arr[i].T @ (r_fl[i] - m_fl[i]) for i in range(len(wfl_arr))], axis=0)
    return float(cp.asnumpy(result)) if use_gpu and GPU_AVAILABLE else result
