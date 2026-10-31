"""
Coordinate transformation and geometric utilities for the PACO algorithms.
"""
from typing import Optional, Tuple, Union

import numpy as np
from vip_hci.var import cart_to_pol, pol_to_cart

from .gpu import GPU_AVAILABLE, cp


def get_rotated_pixel_coords(x: np.ndarray,
                             y: np.ndarray,
                             p0: Tuple[int, int],
                             angles: np.ndarray,
                             astro_convention: bool = False) -> np.ndarray:
    """
    Compute the pixel coordinates to which a source at p0 rotates in each frame.

    Given a candidate companion position p0 in the derotated frame, the function
    returns the corresponding pixel positions in each raw science frame by rotating
    backward by the parallactic angles.

    Parameters
    ----------
    x : np.ndarray, shape (H, W)
        x-coordinate grid (columns), centred on the image.
    y : np.ndarray, shape (H, W)
        y-coordinate grid (rows), centred on the image.
    p0 : tuple of int
        (row, col) position of the candidate companion in the centred grid.
    angles : np.ndarray, shape (T,)
        Parallactic angles in degrees for each frame.
    astro_convention : bool
        If True, use the astronomical (North-up) angle convention.

    Returns
    -------
    np.ndarray, shape (T, 2)
        Per-frame rotated pixel coordinates as (row, col) pairs.
    """
    # For non-square images the column and row centers differ.
    # x.shape is (H, W): axis-0 is rows (height), axis-1 is cols (width).
    # Must match the SAME reference used below to build `phi0` (width/2,
    # height/2 -- not (width-1)/2) and the grid's own definition
    # (x[i,j]=j-W/2, y[i,j]=i-H/2, see paco_base.PACOCalc): using (N-1)/2
    # here while subtracting N/2 below introduced a constant, rotation-
    # independent 0.5px offset in the returned coordinates (verified: at
    # angles=0, i.e. zero rotation, this used to return p0 shifted by
    # exactly (-0.5, -0.5) instead of p0 itself).
    center_row = x.shape[0] / 2.0   # H/2
    center_col = x.shape[1] / 2.0   # W/2

    # x/y are centred coordinate grids built as x[i,j]=j-W/2, y[i,j]=i-H/2
    # (see paco_base.PACOCalc). Evaluate that mapping directly instead of
    # indexing with int(p0) — indexing silently floors p0 to the nearest
    # pixel, which makes fractional (sub-pixel) candidate positions
    # collapse onto the same integer cell and makes `oversampling` in
    # PACOBase.run() a no-op (verified: two different sub-pixel p0 values in
    # the same integer cell used to return identical rotated coordinates).
    width = x.shape[1]
    height = x.shape[0]
    phi0 = np.array([p0[1] - width / 2.0, p0[0] - height / 2.0])

    rad, theta = cart_to_pol(phi0[0], phi0[1], astro_convention=astro_convention)
    angles_rad = -1 * angles + theta
    angles_pol = np.array([rad * np.ones_like(angles_rad), angles_rad])

    nx, ny = pol_to_cart(angles_pol[0], angles_pol[1], astro_convention=astro_convention)
    # nx is the column component (horizontal), ny is the row component (vertical)
    nx += center_col
    ny += center_row

    return np.array([nx, ny]).T


def create_boolean_circular_mask(
    shape: Union[np.ndarray, tuple, list],
    radius: Optional[int] = 4,
    center: Optional[Tuple[int, int]] = None,
    use_gpu: bool = False,
) -> Union[np.ndarray, "cp.ndarray"]:
    """
    Create a 2-D boolean circular mask.

    Parameters
    ----------
    shape : array-like of int, length 2
        Shape (height, width) of the output mask.
    radius : int, optional
        Radius of the circular region in pixels.
    center : tuple of int, optional
        (row, col) centre of the circle. Defaults to the image centre.
    use_gpu : bool
        Return a CuPy array instead of NumPy.

    Returns
    -------
    np.ndarray or cupy.ndarray, shape (height, width)
        Boolean mask that is True inside the circle.
    """
    xp = cp if use_gpu and GPU_AVAILABLE else np

    w, h = shape[0], shape[1]
    if center is None:
        center = [int(w / 2), int(h / 2)]
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    X, Y = xp.ogrid[:w, :h]
    dist2 = (X - center[0]) ** 2 + (Y - center[1]) ** 2
    return dist2 <= radius ** 2


def extract_patch_window(cube: np.ndarray,
                         px: Tuple[int, int],
                         patch_width: int,
                         patch_area: int,
                         full_frame_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Extract the temporal column of patch pixels centred at px.

    Shared implementation for what used to be four near-identical copies
    (PACOBase.get_patch and the module-level _g_get_patch in paco_base.py,
    fast_paco.py, and full_paco.py) of the same logic, all built around

        mask = create_boolean_circular_mask(cube[0].shape, radius=k, center=px)
        patch = cube[np.broadcast_to(mask, cube.shape)]

    i.e. a mask the size of a full FRAME, broadcast across the whole cube
    and used for a single boolean-fancy-index over all of it. That forces
    NumPy to scan every one of the cube's T*H*W elements to find the
    T*patch_area True ones -- fine for a small field, but on a 1028x1028
    cube this measured at 109 ms/call (vs 0.05 ms/call for the windowed
    version below, a ~2200x difference), and this function is called on
    every cache miss in every PACO variant. For a large field this was the
    dominant cost behind PACO's radius-ring traversal running
    dramatically slower than expected, since its GPU-prefill step calls
    this once per unique miss *serially* in the parent process (not
    divided across worker processes the way the ordinary lazy-miss path
    is) -- but the same inefficiency was already present, just hidden by
    being split across workers, in the plain per-pixel lazy path used by
    every variant on every cache miss.

    Fix: crop a (patch_width, patch_width) window around px FIRST (O(1)
    relative to field size), then mask only that small window. A circular
    mask's value at a given position depends only on that position's
    offset from the mask's own centre, not on the size of the array it is
    evaluated over -- so cropping the window first and masking it small is
    mathematically identical to masking the full frame and taking the
    corresponding crop, and this was verified bit-for-bit identical
    (including the NaN-fill boundary case) against the full-frame version
    it replaces.

    Parameters
    ----------
    cube : np.ndarray, shape (T, H, W)
    px : tuple of int
        (row, col) position (matches this codebase's (x, y) = (row, col)
        convention at existing call sites).
    patch_width : int
        Patch diameter in pixels.
    patch_area : int
        Number of pixels in the circular mask of the given patch_width
        (precomputed by callers since it's needed for the NaN-fill shape
        regardless of which code path is taken).
    full_frame_mask : np.ndarray, optional
        A pre-computed mask the size of a full frame (cube[0].shape),
        e.g. from an external create_boolean_circular_mask(cube[0].shape,
        ...) call, for API compatibility with callers that pass their own
        mask (PACOBase.get_patch's `mask` parameter). Cropped to the same
        small window as the cube itself (a cheap O(patch_width^2) slice)
        rather than used for a full-frame boolean index -- if not given,
        a small mask is computed directly at the window's own size.

    Returns
    -------
    np.ndarray, shape (T, patch_area)
        Flattened patch pixels across time frames. Filled with NaN if px
        is too close to the cube's spatial edge for a full patch.
    """
    k = patch_width // 2
    k2 = k + (patch_width % 2)
    # px arrives as a plain int at most call sites already, but not all
    # (e.g. PACOBase.compute_statistics passes phi0s entries directly, which
    # are float64 -- integer-VALUED for the s=1 default, but still a float
    # dtype). The mask-based approach this replaces tolerated a fractional
    # centre implicitly (create_boolean_circular_mask's distance formula
    # doesn't require an integer); slicing does. Round explicitly: for
    # integer-valued floats (the case actually exercised anywhere in this
    # codebase) this is a no-op, and it gives a well-defined "nearest
    # pixel" interpretation for a genuinely fractional px rather than
    # slicing on a truncated float and raising a TypeError.
    x, y = int(round(px[0])), int(round(px[1]))
    nx, ny = cube.shape[1:3]
    if not (k <= x < nx - k2 and k <= y < ny - k2):
        return np.full((cube.shape[0], patch_area), np.nan)
    # Window/mask span 2k+1 (the TRUE extent of a radius-k circle: from
    # x-k to x+k inclusive), NOT `patch_width` -- these only coincide when
    # patch_width is odd (k2 = k+1, so the window built from k/k2 happens
    # to be 2k+1 wide too). For EVEN patch_width (k2 = k), a window sized
    # `patch_width` = 2k is one short of the circle's real footprint, so
    # the small mask below silently lost the outermost ring of pixels
    # (confirmed: K=27 instead of 29 for patch_width=6, radius=3 -- a real
    # bug from an earlier version of this function, only ever exercised
    # with sph3's odd patch_width=7 before, which is why it went
    # unnoticed). The pre-existing k/k2 boundary check above is left
    # untouched (still slightly conservative for odd patch_width, exact
    # for even) to preserve exactly which positions are valid vs NaN.
    window = cube[:, x - k:x + k + 1, y - k:y + k + 1]
    if full_frame_mask is None:
        small_mask = create_boolean_circular_mask((2 * k + 1, 2 * k + 1),
                                                  radius=k, center=(k, k))
    else:
        small_mask = full_frame_mask[x - k:x + k + 1, y - k:y + k + 1]
    return window[:, small_mask]
