"""
Compute the global A1 metric for spatial RSR coverage bias.
"""

import numpy as np
# from .helpers import estimate_optimal_block_size


def compute_A1(
    fg_curve: np.ndarray,
    bg_curve: np.ndarray,
) -> float:
    """
    Compute the global A1 metric for a list of spatial RSR coverage biases.

    A1 quantifies how much a feature's foreground spatial distribution
    deviates from the background distribution across all scanning angles.

    The metric is computed as the geometric mean of squared RSR ratios
    between foreground and background, providing a global measure of
    spatial clustering strength relative to the background distribution.

    Args:
        fg_curve: 1D array of RSR values for feature foregrounds.
        bg_curve: 1D array of RSR values for the background at each scan center.

    Returns:
        A1_scores: A list of global spatial coverage bias metrics.
            - A1 = 1: No spatial bias (foreground matches background distribution)
            - A1 > 1: Positive spatial bias (foreground more clustered than background)
            - A1 < 1: Negative spatial bias (foreground more dispersed than background)
    """

    if fg_curve.shape != bg_curve.shape:
        raise ValueError("Each fg_curve and bg_curve must have the same length")

    if len(fg_curve) == 0:
        raise ValueError("Input curves cannot be empty")

    ratios = (fg_curve**2) / (bg_curve**2)
    ratios = np.maximum(ratios, np.finfo(float).tiny)
    return float(np.exp(np.mean(np.log(ratios))))


def compute_A2(
    fg_curve: np.ndarray,
    bg_curve: np.ndarray,
) -> float:
    """
    Compute the global A2 metric as the Gini coefficient of squared RSR ratios.

    A2 ranges between 0 and 1: values near 0 indicate even distribution across angles;
    values near 1 indicate concentration in few sharp peaks.

    Args:
        fg_curve: 1D array of RSR for the feature foregrounds.
        bg_curve: 1D array of RSR for the background.

    Returns:
        Gini coefficient of the squared RSR ratios.
    """

    if fg_curve.shape != bg_curve.shape:
        raise ValueError("Each fg_curve and bg_curve must have the same length")
    if len(fg_curve) == 0:
        raise ValueError("Input curves cannot be empty")

    # squared‐ratio enrichment values
    ratios = (fg_curve**2) / (bg_curve**2)
    ratios = np.maximum(ratios, np.finfo(float).tiny)

    # sort enrichment values in non-decreasing order
    sorted_ratios = np.sort(ratios)
    N = sorted_ratios.size
    # compute Gini coefficient: sum_{i=1}^N (2i - N - 1) * r_i / (N * sum(r))
    indices = np.arange(1, N + 1)
    numerator = np.sum((2 * indices - N - 1) * sorted_ratios)
    denominator = N * np.sum(sorted_ratios)
    gini = numerator / denominator if denominator != 0 else 0.0
    return float(gini)


# def compute_A1_with_statistics(
#     fg_curve: np.ndarray,
#     bg_curve: np.ndarray,
#     n_bootstrap: int = 1000,
#     block_size: int = None,
# ) -> dict:
#     """
#     Compute A1 and empirical p-value via circular block bootstrap.

#     Args:
#         fg_curve: 1D array of RSR values for the feature foreground at each scan center.
#         bg_curve: 1D array of RSR values for the background at each scan center.
#         n_bootstrap: Number of bootstrap iterations.
#         block_size: Block size for circular bootstrap.

#     Returns:
#         A1: The global spatial coverage bias metric.
#         p_value: Empirical p-value for deviation from null (A1=1).
#     """
#     A1 = compute_A1(fg_curve, bg_curve)
#     n_angles = len(fg_curve)

#     if block_size is None:
#         block_size = estimate_optimal_block_size(fg_curve)

#     bootstrap_A1s = []

#     # Circular block bootstrap
#     for _ in range(n_bootstrap):
#         res = []
#         while len(res) < n_angles:
#             start = np.random.randint(0, n_angles)

#             for i in range(block_size):
#                 if len(res) < n_angles:
#                     res.append((start + i) % n_angles)  # Wrap around

#         sample = fg_curve[res], bg_curve[res]
#         bootstrap_A1s.append(compute_A1(*sample))

#     # empirical p-value: one-sided test against null=1
#     arr = np.array(bootstrap_A1s)
#     if A1 > 1:
#         p_val = np.mean(arr <= 1)
#     else:
#         p_val = np.mean(arr >= 1)

#     return {"A1": A1, "p_value": float(p_val)}


# def compute_A2_with_statistics(
#     fg_curve: np.ndarray,
#     bg_curve: np.ndarray,
#     n_bootstrap: int = 1000,
#     block_size: int = None,
# ) -> dict:
#     """
#     Compute A2 and empirical p-value via circular block bootstrap.

#     Args:
#         fg_curve: 1D array of RSR values for the feature foreground at each scan center.
#         bg_curve: 1D array of RSR values for the background at each scan center.
#         n_bootstrap: Number of bootstrap iterations.
#         block_size: Block size for circular bootstrap.

#     Returns:
#         A2: The global skewness metric.
#         p_value: Empirical p-value for deviation from null (A2=0).
#     """
#     A2 = compute_A2(fg_curve, bg_curve)
#     n_angles = len(fg_curve)

#     if block_size is None:
#         block_size = estimate_optimal_block_size(fg_curve)

#     bootstrap_A2s = []
#     for _ in range(n_bootstrap):
#         res = []
#         while len(res) < n_angles:
#             start = np.random.randint(0, n_angles)
#             for i in range(block_size):
#                 if len(res) < n_angles:
#                     res.append((start + i) % n_angles)
#         sample = fg_curve[res], bg_curve[res]
#         bootstrap_A2s.append(compute_A2(*sample))

#     # empirical p-value: one-sided test against null=0
#     arr2 = np.array(bootstrap_A2s)
#     if A2 > 0:
#         p_val2 = np.mean(arr2 <= 0)
#     else:
#         p_val2 = np.mean(arr2 >= 0)

#     return {"A2": A2, "p_value": float(p_val2)}
