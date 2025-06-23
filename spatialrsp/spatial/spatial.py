"""
Compute angular area RSRs for foreground and background angles.
"""

from typing import Union, Sequence, Tuple, List
import numpy as np
from .helpers import compute_rsr


def compute_rsp(
    theta_fgs: Union[np.ndarray, Sequence[np.ndarray]],
    theta_bg: np.ndarray,
    scanning_window: float,
    resolution: int,
    scanning_range: np.ndarray,
    mode: str = "absolute",
    expected_model: str = "local",
    normalize: bool = True,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[List[np.ndarray], List[np.ndarray], np.ndarray],
]:
    """Compute RSP curves from foreground/background angular distributions.

    Args:
        theta_fgs: Either a single np.ndarray or list of np.ndarrays containing
                          foreground angular values (radians) for one or more features.
        theta_bg (np.ndarray): Background angular values (radians).
        scanning_window (float): Width of the angular scanning window (radians).
        resolution (int): Number of bins within the scanning window.
        scanning_range (np.ndarray): List of center angles to scan over.
        mode (str): Either "absolute" or "relative".
        expected_model (str): 'random' for uniform-circle null, 'local' for background-adapted null.
        normalize (bool): If True, normalize the output areas by the background RSR.

    Returns:
        tuple: When single feature provided:
               - In absolute mode: (fg_curve, expected_fg_curve, bg_curve)
               - In relative mode: (fg_curve, bg_curve)
               When multiple features provided:
               - In absolute mode: (fg_curves, exp_fg_curves, bg_curve)
               - In relative mode: (fg_curves, bg_curve)
               where fg_curves and exp_fg_curves are lists of arrays.
    """
    bins = np.linspace(-scanning_window / 2, scanning_window / 2, resolution + 1)
    single_feature = isinstance(theta_fgs, np.ndarray)
    if single_feature:
        theta_fgs = [theta_fgs]
    elif not isinstance(theta_fgs, (list, tuple)):
        raise ValueError(
            "theta_fgs must be a numpy array or a list/tuple of numpy arrays"
        )

    num_features = len(theta_fgs)
    coverages = [len(theta_fg) / len(theta_bg) for theta_fg in theta_fgs]

    fg_curves = [np.zeros(len(scanning_range)) for _ in range(num_features)]
    exp_fg_curves = (
        [np.zeros(len(scanning_range)) for _ in range(num_features)]
        if mode == "absolute"
        else None
    )
    bg_curve = np.zeros(len(scanning_range))

    for i, center in enumerate(scanning_range):
        bg_rsr = compute_rsr(
            theta_bg,
            center,
            scanning_window,
            bins,
            1.0,  # coverage ignored in relative mode
            "relative",
        )
        bg_curve[i] = 1.0 if normalize else bg_rsr

        for j, theta_fg in enumerate(theta_fgs):
            fg_rsr = compute_rsr(
                theta_fg, center, scanning_window, bins, coverages[j], mode
            )
            fg_curves[j][i] = (fg_rsr / bg_rsr) if normalize else fg_rsr

            if mode == "absolute" and exp_fg_curves is not None:
                if expected_model == "random":
                    raw_exp = np.sqrt(coverages[j] * (scanning_window / 2))
                else:  # 'local'
                    raw_exp = np.sqrt(coverages[j]) * bg_rsr
                exp_fg_curves[j][i] = (raw_exp / bg_rsr) if normalize else raw_exp

    if mode == "absolute":
        if single_feature:
            return fg_curves[0], exp_fg_curves[0], bg_curve
        else:
            return fg_curves, exp_fg_curves, bg_curve
    else:
        if single_feature:
            return fg_curves[0], bg_curve
        else:
            return fg_curves, bg_curve
