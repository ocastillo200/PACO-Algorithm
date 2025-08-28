from typing import List, Optional, Tuple, Union
import numpy as np


def get_rotated_pixel_coords(x: np.ndarray,
                            y: np.ndarray,
                            p0: Tuple[int, int],
                            angles: np.ndarray,
                            astro_convention: Optional[bool] = False) -> np.ndarray:
    """Calculates the new coordinates of a point after rotating around the center.

    Parameters
    ----------
    x : np.ndarray
        A 2D array representing the x-coordinates of the grid (e.g., from
        np.meshgrid).
    y : np.ndarray
        A 2D array representing the y-coordinates of the grid (e.g., from
        np.meshgrid).
    p0 : Tuple[int, int]
        The initial (y, x) integer coordinates of the point to be rotated.
    angles : np.ndarray
        A 1D array of rotation angles in degrees.
    astro_convention : bool, optional
        If True, assumes the astronomical convention where North is up (+y)
        and East is left (+x), with angles increasing counter-clockwise from
        North. Defaults to False.

    Returns
    -------
    np.ndarray
        A 2D array of shape (N, 2) where N is the number of angles,
        containing the new floating-point (y, x) coordinates for each rotation.
    """
    px = x[int(p0[0]), int(p0[1])]
    py = y[int(p0[0]), int(p0[1])]

    initial_theta = np.rad2deg(np.arctan2(py, px))
    if astro_convention:
        initial_theta -= 90

    final_angles_deg = -angles + initial_theta
    final_angles_rad = np.deg2rad(final_angles_deg)
    radius = np.sqrt(px**2 + py**2)

    if astro_convention:
        final_angles_rad += np.pi / 2
        nx = -radius * np.cos(final_angles_rad)
        ny = radius * np.sin(final_angles_rad)
    else:
        nx = radius * np.cos(final_angles_rad)
        ny = radius * np.sin(final_angles_rad)

    nx += int(x.shape[0] / 2)
    ny += int(y.shape[0] / 2)

    return np.array([ny, nx]).T


def create_boolean_circular_mask(shape: Union[Tuple[int, int], List[int]],
                                 radius: Optional[int] = 4,
                                 center: Optional[Tuple[int, int]] = None
                                 ) -> np.ndarray:
    """
    Returns a 2D boolean mask given a shape, radius, and center.

    Parameters
    ----------
    shape : tuple[int, int] or list[int]
        The (height, width) shape of the output mask array.
    radius : int, optional
        The radius of the circular mask in pixels.
    center : tuple[int, int], optional
        The (y, x) pixel coordinates for the center of the mask. If None,
        the center of the shape is used.

    Returns
    -------
    mask : np.ndarray
        A 2D boolean array where values inside the circular region are True
        and values outside are False.
    """
    h, w = shape
    if center is None:
        center = (h // 2, w // 2)
    if radius is None:
        radius = min(center[0], center[1], w - center[1], h - center[0])

    Y, X = np.ogrid[:h, :w]
    dist_from_center_sq = (Y - center[0])**2 + (X - center[1])**2

    mask = dist_from_center_sq <= radius**2
    return mask