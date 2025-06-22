"""
I/O utilities for loading and saving single-cell datasets into AnnData format.
Supports multiple formats including CSV, H5AD, 10X, Loom, and MatrixMarket.
"""

import logging
from pathlib import Path
from typing import Union, Optional, Literal

import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse as sp

logger = logging.getLogger(__name__)


def load_data(
    file_path: Union[str, Path, AnnData],
    fmt: Literal["auto", "h5ad", "10x", "loom", "csv", "tsv", "mtx"] = "auto",
    sparse: bool = True,
    memory_map: bool = True,
    chunk_size: Optional[int] = None,
    verbose: bool = True,
    **read_kwargs,
) -> AnnData:
    """
    Load single-cell data from a variety of file formats into AnnData.

    Args:
        file_path: Path to input file or AnnData object.
        fmt: File format. 'auto' infers from file extension.
        sparse: Whether to convert matrix to sparse CSR format.
        memory_map: Enable memory mapping when supported.
        chunk_size: For large CSVs/TSVs, load in chunks.
        verbose: Print or log information.
        **read_kwargs: Extra arguments passed to Scanpy or Pandas.

    Returns:
        AnnData object.
    """
    if isinstance(file_path, AnnData):
        if verbose:
            logger.info("Input is already an AnnData object; returning as-is.")
        return file_path

    path = Path(file_path)
    ext = path.suffix.lower()

    if fmt == "auto":
        fmt = {
            ".h5ad": "h5ad",
            ".loom": "loom",
            ".csv": "csv",
            ".tsv": "csv",
            ".txt": "csv",
            ".mtx": "mtx",
        }.get(ext, "10x" if path.is_dir() else None)

        if not fmt:
            raise ValueError(f"Could not auto-detect format from path: {path}")

    if verbose:
        logger.info("[↓] Loading file: %s (detected format: %s)", path, fmt)

    if fmt == "h5ad":
        adata = sc.read_h5ad(str(path), **read_kwargs)
    elif fmt == "10x":
        adata = sc.read_10x_mtx(str(path), **read_kwargs)
    elif fmt == "loom":
        adata = sc.read_loom(str(path), **read_kwargs)
    elif fmt in {"csv", "tsv"}:
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(
            path,
            sep=sep,
            memory_map=memory_map,
            engine="c",
            dtype="float32",
            na_filter=False,
            chunksize=chunk_size,
            **read_kwargs,
        )
        adata = AnnData(
            X=df.values if not chunk_size else df,
            obs=pd.DataFrame(index=df.index),
            var=pd.DataFrame(index=df.columns),
        )
    elif fmt == "mtx":
        adata = sc.read_mtx(str(path), **read_kwargs)
    else:
        raise ValueError(f"Unsupported format '{fmt}' for path {path}")

    if sparse and not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    if verbose:
        logger.info("[✓] Loaded AnnData with shape %s", adata.shape)

    return adata


def save_data(
    adata: AnnData,
    file_path: Union[str, Path],
    compression: Optional[Literal["gzip", "lzf"]] = "lzf",
    verbose: bool = True,
) -> None:
    """
    Save an AnnData object to disk in H5AD format.

    Args:
        adata: AnnData object to save.
        file_path: Output file path.
        compression: Compression type (gzip, lzf, or None).
        verbose: If True, print saving details.
    """
    path = Path(file_path)
    if verbose:
        logger.info("[↑] Saving AnnData to %s with compression=%s", path, compression)
    adata.write_h5ad(
        str(path), compression=compression, as_dense=["obs", "var"], chunks=True
    )
    if verbose:
        logger.info("[✓] Save complete.")
