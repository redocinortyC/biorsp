"""
Compute the Radar Scan Radius (RSR) for a set of angles within a sliding window.
"""

import numpy as np


def compute_rsr(
    angles: np.ndarray,
    center: float,
    window: float,
    bins: np.ndarray,
    coverage: float,
    mode: str,
) -> float:
    """
    Compute the Radar Scanning Radius (RSR) for a set of angles within
    a sliding window.

    The RSR is the sqrt of the area under the empirical cumulative
    distribution function (ECDF) of the angles relative to center.

    Modes:
        - "absolute": after computing raw area A_fg, scale it by coverage
                    (N_fg/N_bg) so that sparser fg leads to smaller radius.
        - "relative": leave the raw ECDF-area alone.

    Args:
        angles: The angles to compute the RSR for, in radians.
        center: The center angle in radians.
        window: The width of the sliding window in radians.
        bins: The bin edges for the histogram.
        coverage: The coverage ratio (N_fg / N_bg) for scaling in absolute mode.
        mode: Either absolute or relative.

    Returns:
        The computed RSR value, which is the square root of the area under the ECDF
        of the angles within the specified window, scaled by coverage if in absolute mode.
    """
    rel = ((angles - center + np.pi) % (2 * np.pi)) - np.pi

    mask = np.abs(rel) <= window / 2
    counts, _ = np.histogram(rel[mask], bins=bins)
    counts = np.maximum(counts.astype(float), np.finfo(float).eps)
    area, _ = compute_area_under_cdf(counts, bins)

    if mode == "absolute":
        area *= coverage

    return np.sqrt(area)


def compute_area_under_cdf(
    counts: np.ndarray,
    bins: np.ndarray,
) -> tuple:
    """
    Compute the area under the empirical cumulative distribution function.

    Args:
        counts: The (regularized) counts in each bin.
        bins: The bin edges for the histogram.

    Returns:
        area: area under the ECDF
        total_width: sum of bin widths
    """
    widths = np.diff(bins)
    cum_counts = np.cumsum(counts)
    total = cum_counts[-1]
    if total <= 0:
        return 0.0, widths.sum()

    cdf_vals = cum_counts / total
    area = np.inner(cdf_vals, widths)
    return area, widths.sum()
