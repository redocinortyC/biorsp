"""
Compute the global A1 metric for spatial RSR.
"""

import numpy as np


def compute_A1(
    fg1_curve: np.ndarray,
    fg2_curve: np.ndarray,
) -> float:
    """
    Compute the global A1 metric as the geometric mean of per-angle RSR ratios.

    Args:
        fg1_curve: 1D array of RSR ratios for feature 1 at each scan center.
        fg2_curve: 1D array of RSR ratios for feature 2 (or expected RSR) at each scan center.

    Returns:
        A1: The geometric mean of the squared-ratio curve.
    """
    if fg1_curve.shape != fg2_curve.shape:
        raise ValueError("fg1_curve and fg2_curve must have the same length")

    ratios = (fg1_curve**2) / (fg2_curve**2)
    ratios = np.maximum(ratios, np.finfo(float).tiny)  # avoid log(0)
    return float(np.exp(np.mean(np.log(ratios))))
