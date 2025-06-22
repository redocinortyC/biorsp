"""
Format numbers to a specified number of significant figures.

This module provides a single function, `sigfigs`, which formats a floating-point
number to a specified number of significant figures.
"""

import math


def sigfigs(x: float, n: int) -> str:
    """Format number to n significant figures.

    Args:
        x (float): Value to format.
        n (int): Number of significant figures.

    Returns:
        str: Formatted string.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    if x is None:
        raise ValueError("x must not be None")

    if x == 0:
        return f"{0:.{n - 1}f}"

    exp = int(math.floor(math.log10(abs(x))))
    scaled = round(x / (10**exp), n - 1) * (10**exp)
    if 1e-4 <= abs(scaled) < 1e6:
        return f"{scaled:.{max(n - 1 - exp, 0)}f}"

    try:
        return f"{scaled:.{n - 1}e}"
    except Exception:
        raise RuntimeError("Failed to format number") from None
