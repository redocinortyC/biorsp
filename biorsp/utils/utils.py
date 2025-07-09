"""
Misc. functions for biorsp.
"""

from typing import Optional
import numpy as np
import anndata
from scipy import sparse


def extract_foreground_mask(
    adata: anndata.AnnData,
    feature: str,
    *,
    layer: Optional[str] = None,
    threshold: Optional[float] = None,
    quantile: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute a boolean mask selecting foreground for a given feature.

    Args:
        adata: Annotated data matrix with `.obs`, `.X`, and/or `.layers[...]`.
        feature: The gene/feature name in `adata.var_names` (or in `adata.obs` if not in X).
        layer: If provided, draw expression from `adata.layers[layer][:, feature]`.
            Otherwise, if `feature` in `adata.obs`, use that; else use `adata.X[:, feature]`.
        threshold: Numeric cutoff: foreground = expr > threshold.
        quantile: If `threshold` is None and `quantile` is given (0<q<1), set
            threshold = the q-th quantile of `expr`.
        mask: If provided, must be a boolean array of length `adata.n_obs`; returned as-is.

    Returns:
        fg_mask: Boolean array where True indicates foreground cells.
    """
    n = adata.n_obs

    if mask is not None:
        if mask.dtype != bool or mask.shape[0] != n:
            raise ValueError("`mask` must be a boolean array of length n_obs")
        return mask

    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers")
        idx = adata.var_names.get_loc(feature)
        expr_col = adata.layers[layer][:, idx]
    elif feature in adata.obs:
        expr_col = adata.obs[feature].values
    else:
        idx = adata.var_names.get_loc(feature)
        expr_col = adata.X[:, idx]

    if sparse.issparse(expr_col):
        expr = expr_col.toarray().ravel()
    else:
        expr = np.asarray(expr_col).ravel()
    if expr.shape[0] != n:
        raise ValueError(f"Expression vector length {expr.shape[0]} != n_obs {n}")

    if threshold is None and quantile is not None:
        if not (0 < quantile < 1):
            raise ValueError("`quantile` must be between 0 and 1")
        threshold = np.quantile(expr, quantile)

    if threshold is not None:
        fg_mask = expr > threshold
    else:
        fg_mask = expr > 0

    return fg_mask


def get_polar_angles(
    adata: anndata.AnnData,
    mask: Optional[np.ndarray] = None,
    polar_coord: str = "X_polar",
) -> np.ndarray:
    """
    Return the polar-angle vector (θ) for the cells selected by `mask` (or all cells).

    Args:
        adata: Annotated data matrix with polar coordinates in `.obsm[polar_coord]`.
        mask: Boolean array of length n_obs selecting foreground cells, or None to select all cells
            (background). Non-boolean arrays will be cast to bool if possible.
        polar_coord: Key in `adata.obsm` where your polar coordinates live (shape n_obs×2,
            with column 1 = angle in radians).

    Returns:
        angles: The θ values (radians) for the selected cells.
    """
    n = adata.n_obs

    if mask is None:
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.asarray(mask)
        if mask.shape[0] != n:
            raise ValueError(f"`mask` length {mask.shape[0]} ≠ number of cells {n}")
        if not np.issubdtype(mask.dtype, np.bool_):
            try:
                mask = mask.astype(bool)
            except Exception as e:
                raise ValueError("`mask` must be a boolean array") from e

    if polar_coord not in adata.obsm:
        raise KeyError(f"Polar-coord key '{polar_coord}' not found in .obsm")
    coords = adata.obsm[polar_coord]
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"Expected .obsm['{polar_coord}'] of shape (n_obs, 2), got {coords.shape}"
        )
    angles = coords[:, 1].astype(float)

    return angles[mask]


def percentile_to_threshold(values, percentile):
    """
    Convert percentile threshold to actual threshold.

    Args:
        values: Array of values
        percentile: Threshold percentile (0-1)

    Returns:
        Actual threshold value
    """
    if percentile == 0:
        return -np.inf
    elif percentile == 1:
        return np.inf

    non_zero_values = values[values > 0]
    if non_zero_values.size == 0:
        return np.inf

    return np.percentile(non_zero_values, percentile * 100)
