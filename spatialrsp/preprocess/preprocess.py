"""
Standardized preprocessing pipeline for AnnData objects.

Includes:
  - Quality control filtering
  - Heuristic log-normalization
  - Dimensionality reduction (PCA, UMAP, t-SNE)
  - Optional polar-coordinate transform on embeddings
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from ..utils.formatting import sigfigs
from ..utils.transform import cartesian_to_polar

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _get_dense_array(adata: ad.AnnData) -> np.ndarray:
    """
    Ensure that `adata.X` is a dense NumPy array.

    Converts sparse or other array‐like matrices to `.X.toarray()` or `np.array`.
    """
    matrix = adata.X
    if not isinstance(matrix, np.ndarray):
        dense = matrix.toarray() if hasattr(matrix, "toarray") else np.array(matrix)
        logger.debug("Converted adata.X to dense array of shape %s", dense.shape)
        return dense
    logger.debug("adata.X already a NumPy array; copying")
    return matrix.copy()


def _is_log_normalized(matrix: np.ndarray, cutoff: float = 10.0) -> bool:
    """
    Heuristic check whether data is already log-normalized.

    Assumes that if the 99th percentile is below `cutoff`, it's log-scale.
    """
    percentile_99 = np.percentile(matrix, 99)
    logger.debug("Data 99th percentile = %s", sigfigs(percentile_99, 3))
    return percentile_99 < cutoff


class Preprocessor:
    """
    A configurable preprocessing pipeline for AnnData.

    Methods correspond to sequential pipeline steps; use `.run(...)` for an end-to-end flow.
    """

    def __init__(self) -> None:
        """Initialize the Preprocessor (uses module logger by default)."""
        self.logger = logger

    def quality_control(
        self,
        adata: ad.AnnData,
        min_nonzero: int = 10,
        max_dropout: float = 0.5,
    ) -> ad.AnnData:
        """
        Filter cells by minimum expressed features and maximum dropout rate.

        Args:
            adata: Input AnnData object.
            min_nonzero: Minimum nonzero counts per cell.
            max_dropout: Maximum fraction of zeros per cell.

        Returns:
            Filtered AnnData object.
        """
        matrix = _get_dense_array(adata)
        nonzero_counts = (matrix != 0).sum(axis=1)
        dropout_rates = 1 - (nonzero_counts / matrix.shape[1])
        keep_mask = (nonzero_counts >= min_nonzero) & (dropout_rates <= max_dropout)

        num_kept = int(keep_mask.sum())
        self.logger.info(
            "QC: kept %d/%d cells (min_nonzero=%d, max_dropout=%.2f)",
            num_kept,
            adata.n_obs,
            min_nonzero,
            max_dropout,
        )

        return adata[keep_mask].copy()

    def normalize(
        self,
        adata: ad.AnnData,
        force: bool = False,
    ) -> ad.AnnData:
        """
        Normalize counts to median total and log1p-transform.

        Args:
            adata: Input AnnData object.
            force: If True, always normalize; otherwise skip if already log-normalized.

        Returns:
            AnnData with normalized `.X`.
        """
        matrix = _get_dense_array(adata)
        if not force and _is_log_normalized(matrix):
            self.logger.info("Data seems already log-normalized; skipping step")
            return adata

        totals = matrix.sum(axis=1)
        median_total = np.median(totals)
        self.logger.info(
            "Normalizing counts: median total = %s", sigfigs(median_total, 3)
        )

        scaled = (matrix.T / totals).T * median_total
        adata.X = np.log1p(scaled)
        self.logger.info("Normalization complete; new shape = %s", adata.X.shape)
        return adata

    def reduce_dimensionality(
        self,
        adata: ad.AnnData,
        method: str = "PCA",
        n_components: int = 50,
        **kwargs: Any,
    ) -> ad.AnnData:
        """
        Compute a low-dimensional embedding and store it in `adata.obsm`.

        Args:
            adata: Input AnnData object.
            method: One of "PCA", "UMAP", "TSNE".
            n_components: Number of dimensions in the embedding.
            **kwargs: Extra parameters passed to the reducer.

        Returns:
            AnnData with new embedding in `obsm["X_<method>"]`.
        """
        matrix = _get_dense_array(adata)
        method_name = method.upper()
        self.logger.info("Running %s with %d components", method_name, n_components)

        if method_name == "PCA":
            reducer = PCA(n_components=n_components, **kwargs)
        elif method_name == "UMAP":
            n_neighbors = min(kwargs.get("n_neighbors", 15), adata.n_obs - 1)
            reducer = UMAP(n_components=n_components, n_neighbors=n_neighbors, **kwargs)
        elif method_name in {"TSNE", "T-SNE"}:
            perplexity = min(kwargs.get("perplexity", 30), adata.n_obs - 1)
            reducer = TSNE(n_components=n_components, perplexity=perplexity, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")

        embedding = reducer.fit_transform(matrix)
        key = f"X_{method_name.lower()}"
        adata.obsm[key] = embedding
        self.logger.info(
            "%s embedding stored in obsm['%s'] with shape %s",
            method_name,
            key,
            embedding.shape,
        )
        return adata

    def polar_transform(
        self,
        adata: ad.AnnData,
        embedding_key: str = "X_umap",
        polar_key: str = "X_polar",
        vantage_point: Optional[Tuple[float, float]] = None,
    ) -> ad.AnnData:
        """
        Convert a 2D embedding to polar coordinates.

        Args:
            adata: Input AnnData object.
            embedding_key: Key in `obsm` for the source embedding.
            polar_key: Key to store the polar coordinates.
            vantage_point: Optional (x, y) center; defaults to embedding centroid.

        Returns:
            AnnData with polar coords in `obsm[polar_key]`.
        """
        embedding = adata.obsm.get(embedding_key)
        if embedding is None:
            raise KeyError(f"Embedding '{embedding_key}' not found")

        if vantage_point is None:
            vantage_point = tuple(np.mean(embedding, axis=0))
            self.logger.debug("Computed centroid vantage_point = %s", vantage_point)

        r, theta = cartesian_to_polar(embedding, vantage_point=vantage_point)
        adata.obsm[polar_key] = np.column_stack((r, theta))
        self.logger.info("Polar coordinates stored in obsm['%s']", polar_key)
        return adata

    def run(
        self,
        adata: ad.AnnData,
        qc: Union[bool, Dict[str, Any]] = True,
        normalize: Union[bool, Dict[str, Any]] = True,
        reduction: Optional[Dict[str, Any]] = None,
        polar: bool = False,
    ) -> ad.AnnData:
        """
        Execute the full pipeline on an AnnData object.

        Args:
            adata: Input AnnData.
            qc: False to skip QC, or dict for qc params.
            normalize: False to skip, or dict for normalize params.
            reduction: None to skip, or dict with keys "method", "n_components", etc.
            polar: If True, apply polar_transform on the reduction result.

        Returns:
            The processed AnnData.
        """
        if qc:
            qc_params = qc if isinstance(qc, dict) else {}
            adata = self.quality_control(adata, **qc_params)

        if normalize:
            norm_params = normalize if isinstance(normalize, dict) else {}
            adata = self.normalize(adata, **norm_params)

        if reduction:
            adata = self.reduce_dimensionality(adata, **reduction)

        if polar:
            emb_key = f"X_{(reduction or {}).get('method', 'umap').lower()}"
            adata = self.polar_transform(adata, embedding_key=emb_key)

        return adata
