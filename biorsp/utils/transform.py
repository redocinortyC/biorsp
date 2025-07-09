"""
Utility functions for coordinate transformations.
"""

from typing import Tuple
import numpy as np


def cartesian_to_polar(
    coordinates: np.ndarray,
    *,
    vantage_point: Tuple[float, float] = (0, 0),
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian coordinates to polar coordinates.

    Args:
        coordinates (np.ndarray): Array of shape (n_points, 2) with (x, y) pairs.
        vantage_point (tuple): Origin for polar transformation.
        verbose (bool): If True, print debug information.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Radial (r) and angular (theta) coordinates.
    """
    if coordinates is None or coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must be a 2D array with shape (n_points, 2).")

    dx, dy = coordinates.T - np.asarray(vantage_point)[:, None]

    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)

    if verbose:
        print(
            f"[cartesian_to_polar] Converted {coordinates.shape[0]} points relative to {vantage_point}."
        )

    return r, theta
