from typing import Optional, Tuple
import numpy as np

def compute_statistics_at_pixel(patch: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Calculates the mean and regularized inverse covariance for a temporal patch.

    This function estimates the covariance using a "shrinkage" technique to
    obtain a robust matrix, which is particularly useful when the number of
    samples (frames) is low. It then inverts this regularized matrix.

    Parameters
    ----------
    patch : np.ndarray
        A 2D array of shape (T, K), where T is the number of temporal frames
        and K is the number of pixels in a patch.

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
        A tuple containing the mean vector (m) and the inverse covariance
        matrix (Cinv). Returns (None, None) or values with NaNs if the
        input patch is invalid or the matrix is not invertible.
    """
    if patch is None or np.all(np.isnan(patch)):
        return None, None

    T, K = patch.shape 

    if T < 2:
        m = np.mean(patch, axis=0)
        return m, np.full((K, K), np.nan)
        
    m = np.mean(patch, axis=0)

    # Sample covariance matrix
    S = np.cov(patch, rowvar=False, ddof=0)
    
    if np.any(np.isnan(S)):
         return m, np.full((K, K), np.nan)

    # Data-driven shrinkage factor
    rho = shrinkage_factor(S, T) 

    # Diagonal matrix of S
    F = np.diag(np.diag(S))

    # Regularized covariance matrix
    C = (1.0 - rho) * S + rho * F

    try:
        Cinv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        Cinv = np.full((K, K), np.nan)

    return m, Cinv


def shrinkage_factor(S: np.ndarray, T: int) -> float:
    """
    Calculates the shrinkage factor ρ to regularize the covariance matrix.

    Parameters
    ----------
    S : numpy.ndarray
        The sample covariance matrix.
    T : int
        The number of temporal frames used to calculate S.

    Returns
    -------
    rho : float
        The calculated shrinkage factor, constrained between 0.0 and 1.0.
    """
    trace_S_sq = np.trace(np.dot(S, S))
    trace_S = np.trace(S)
    sum_diag_S_sq = np.sum(np.diag(S)**2.0)

    top = (trace_S_sq + trace_S**2 - 2.0 * sum_diag_S_sq)
    bot = ((T + 1.0) * (trace_S_sq - sum_diag_S_sq))

    if bot == 0:
        p = 1.0
    else:
        p = top / bot

    return max(min(p, 1.0), 0.0)