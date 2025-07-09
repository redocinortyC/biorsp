"""
Compute the Radar Scan Radius (RSR) for a set of angles within a sliding window.
"""

import numpy as np
from statsmodels.stats.multitest import multipletests


def compute_rsr(
    angles: np.ndarray,
    center: float,
    window: float,
    bins: np.ndarray,
) -> float:
    """
    Compute the Radar Scanning Radius (RSR) for a set of angles within
    a sliding window.

    The RSR is computed as the area under the histogram of angles within
    a sliding window centered at a specified angle. The area is computed
    as the sum of the bin heights multiplied by their widths.

    Args:
        angles: The angles to compute the RSR for, in radians.
        center: The center angle in radians.
        window: The width of the sliding window in radians.
        bins: The bin edges for the histogram.

    Returns:
        The computed RSR value.
    """
    rel = ((angles - center + np.pi) % (2 * np.pi)) - np.pi

    mask = np.abs(rel) <= window / 2
    counts, _ = np.histogram(rel[mask], bins=bins)
    counts = np.maximum(counts.astype(float), np.finfo(float).eps)

    area, _ = compute_histogram_area(counts, bins)

    return area


def compute_histogram_area(
    counts: np.ndarray,
    bins: np.ndarray,
) -> tuple:
    """
    Compute the area under the histogram (sum of bin heights × bin widths).

    Args:
        counts: The (regularized) counts in each bin.
        bins: The bin edges for the histogram.

    Returns:
        area: area under the histogram
        total_width: sum of bin widths
    """
    widths = np.diff(bins)
    area = np.inner(counts, widths)
    return area, widths.sum()


def estimate_optimal_block_size(curve: np.ndarray, max_lag: int = 50) -> int:
    """
    Estimate optimal block size using autocorrelation function.

    Rule of thumb: block size ≈ 2 x first lag where autocorr < 0.1

    Args:
        curve: 1D array of values to analyze for autocorrelation.
        max_lag: Maximum lag to consider for autocorrelation.

    Returns:
        Optimal block size for circular bootstrap.
    """
    n = len(curve)
    max_lag = min(max_lag, n // 4)

    for lag in range(1, max_lag):
        shifted = np.roll(curve, lag)
        autocorr = np.corrcoef(curve, shifted)[0, 1]
        if autocorr < 0.1:
            return max(3, 2 * lag)  # Minimum block size of 3

    return max(3, max_lag // 2)  # Fallback


def adjust_pvalues(
    p_values: list[float],
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> list[float]:
    """
    Adjust a list of p-values for multiple testing.

    Args:
        p_values: List of raw p-values.
        method: Correction method ('fdr_bh', 'bonferroni', etc.).
        alpha: Significance level (only used to return boolean reject mask if needed).

    Returns:
        q_values: List of adjusted p-values (q-values).
    """
    p_arr = np.asarray(p_values, dtype=float)
    # multipletests returns: rejected, qvals, alphacSidak, alphacBonf
    _, qvals, _, _ = multipletests(p_arr, alpha=alpha, method=method)
    return qvals.tolist()


def compute_RMSD(
    fg_curve: np.ndarray,
    ref_curve: np.ndarray,
) -> float:
    """
    Root-mean-square deviation between two RSR curves.
    """
    return float(np.sqrt(np.mean((fg_curve - ref_curve) ** 2)))
