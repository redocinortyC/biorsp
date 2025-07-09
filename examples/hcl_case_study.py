#!/usr/bin/env python3
"""Hematopoietic Regulators in the Human Cell Landscape.

This script demonstrates a biorsp workflow on the Human Cell Landscape
(HCL) dataset. It filters fetal hematopoietic cells, computes Radar Scanning
Plot metrics for highly variable genes and classifies them into biorsp
archetypes.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from biorsp.rsp import compute_rsp, compute_A1, compute_A2
from biorsp.rsp.helpers import adjust_pvalues
from biorsp.utils.transform import cartesian_to_polar


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def load_and_subset(path: str, lineage_col: str, lineage_label: str) -> sc.AnnData:
    """Load the HCL dataset and subset to a specific lineage."""
    adata = sc.read_h5ad(path)
    if lineage_col not in adata.obs:
        raise KeyError(f"Column '{lineage_col}' not found in adata.obs")
    mask = adata.obs[lineage_col] == lineage_label
    if mask.sum() == 0:
        raise ValueError(
            f"No cells with {lineage_col} == '{lineage_label}' found"
        )
    return adata[mask].copy()


def preprocess(adata: sc.AnnData, n_pcs: int = 30) -> sc.AnnData:
    """Run standard Scanpy preprocessing."""
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat")
    if adata.var.get("highly_variable", pd.Series([], dtype=bool)).sum() == 0:
        raise ValueError("No highly variable genes were detected")
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    return adata


def compute_hsc_centroid(adata: sc.AnnData, celltype_col: str, hsc_label: str) -> np.ndarray:
    """Compute the UMAP centroid of HSC cells."""
    if celltype_col not in adata.obs:
        raise KeyError(f"Column '{celltype_col}' not found in adata.obs")
    mask = adata.obs[celltype_col] == hsc_label
    if mask.sum() == 0:
        raise ValueError(f"No cells labeled '{hsc_label}' found")
    coords = adata.obsm["X_umap"][mask]
    return coords.mean(axis=0)


def polar_transform(adata: sc.AnnData, origin: np.ndarray) -> None:
    """Add polar coordinates relative to ``origin`` to ``adata.obsm``."""
    r, theta = cartesian_to_polar(adata.obsm["X_umap"], vantage_point=origin)
    adata.obsm["X_polar"] = np.column_stack((r, theta))


def knee_threshold(values: np.ndarray) -> float:
    """Approximate kneedle knee threshold."""
    if values.size == 0:
        return 0.0
    vals = np.sort(values)[::-1]
    x = np.linspace(0, 1, vals.size)
    y = (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)
    line = x
    diff = line - y
    idx = int(np.argmax(diff))
    return float(vals[idx])


def compute_rsp_metrics(adata: sc.AnnData, n_perm: int = 1000, polar_key: str = "X_polar") -> pd.DataFrame:
    """Compute A1/A2 metrics with permutation p-values for all genes."""
    angles = adata.obsm[polar_key][:, 1]
    scanning_range = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    window = np.pi
    resolution = 60

    results = []
    for idx, gene in enumerate(adata.var_names):
        expr = adata[:, idx].X
        expr = expr.toarray().ravel() if hasattr(expr, "toarray") else expr.ravel()
        fg_mask = expr > 0
        if fg_mask.sum() == 0:
            continue
        theta_fg = angles[fg_mask]
        fg_curve, bg_curve = compute_rsp(
            theta_fg,
            angles,
            window,
            resolution,
            scanning_range,
            mode="relative",
        )
        a1 = compute_A1(fg_curve, bg_curve)
        a2 = compute_A2(fg_curve, bg_curve)

        perm_a1 = []
        perm_a2 = []
        for _ in range(n_perm):
            shuffle = np.random.permutation(angles)
            fg_curve_p, bg_curve_p = compute_rsp(
                shuffle[fg_mask],
                angles,
                window,
                resolution,
                scanning_range,
                mode="relative",
            )
            perm_a1.append(compute_A1(fg_curve_p, bg_curve_p))
            perm_a2.append(compute_A2(fg_curve_p, bg_curve_p))
        perm_a1 = np.asarray(perm_a1)
        perm_a2 = np.asarray(perm_a2)
        p_a1 = float((perm_a1 >= a1).mean())
        p_a2 = float((perm_a2 >= a2).mean())
        results.append((gene, a1, a2, p_a1, p_a2))

    df = pd.DataFrame(results, columns=["gene", "A1", "A2", "pval_A1", "pval_A2"]).set_index("gene")
    df["qval_A1"] = adjust_pvalues(df["pval_A1"].tolist())
    df["qval_A2"] = adjust_pvalues(df["pval_A2"].tolist())
    return df


def classify_archetypes(df: pd.DataFrame, alpha: float = 0.05) -> tuple[float, pd.DataFrame]:
    """Assign biorsp archetypes to genes."""
    sig = df[df["qval_A1"] < alpha]
    tau = knee_threshold(sig["A2"].values)
    labels = []
    for _, row in df.iterrows():
        if row["qval_A1"] >= alpha:
            labels.append(pd.NA)
            continue
        if row["A1"] > 1 and row["A2"] > tau:
            labels.append("I")
        elif row["A1"] > 1 and row["A2"] <= tau:
            labels.append("II")
        elif row["A1"] <= 1 and row["A2"] > tau:
            labels.append("III")
        else:
            labels.append("IV")
    df["biorsp_archetype"] = labels
    return tau, df


def plot_top_gene(adata: sc.AnnData, df: pd.DataFrame, origin: np.ndarray, outfile: str) -> str:
    """Plot UMAP colored by the top Archetype I gene."""
    arc1 = df[df["biorsp_archetype"] == "I"]
    if arc1.empty:
        raise ValueError("No Archetype I genes detected")
    top_gene = arc1.sort_values("A1", ascending=False).index[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    sc.pl.umap(adata, color=top_gene, ax=ax, show=False)
    ax.scatter(*origin, c="red", marker="x", s=80, label="HSC centroid")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    return top_gene


# ---------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="biorsp HCL case study")
    parser.add_argument("input", help="Path to HCL h5ad file")
    parser.add_argument("--lineage-col", default="lineage", help="Obs column with lineage annotations")
    parser.add_argument("--lineage-label", default="fetal hematopoietic", help="Value selecting fetal hematopoietic cells")
    parser.add_argument("--celltype-col", default="cell_type", help="Obs column with cell-type labels")
    parser.add_argument("--hsc-label", default="HSC", help="Label identifying hematopoietic stem cells")
    parser.add_argument("--n-pcs", type=int, default=30, help="Number of PCA components")
    parser.add_argument("--n-perm", type=int, default=1000, help="Number of permutations per gene")
    parser.add_argument("--output-h5ad", default="hcl_rsp_results.h5ad", help="Output AnnData file")
    parser.add_argument("--output-image", default="top_gene_umap.png", help="Output figure path")
    args = parser.parse_args(argv)

    adata = load_and_subset(args.input, args.lineage_col, args.lineage_label)
    adata = preprocess(adata, args.n_pcs)
    origin = compute_hsc_centroid(adata, args.celltype_col, args.hsc_label)
    polar_transform(adata, origin)

    metrics = compute_rsp_metrics(adata, args.n_perm)
    tau, metrics = classify_archetypes(metrics)

    adata.var.loc[metrics.index, "A1"] = metrics["A1"]
    adata.var.loc[metrics.index, "A2"] = metrics["A2"]
    adata.var.loc[metrics.index, "qval_A1"] = metrics["qval_A1"]
    adata.var.loc[metrics.index, "qval_A2"] = metrics["qval_A2"]
    adata.var.loc[metrics.index, "biorsp_archetype"] = metrics["biorsp_archetype"]

    top_gene = plot_top_gene(adata, metrics, origin, args.output_image)

    summary = metrics.loc[metrics["biorsp_archetype"] == "I", ["A1", "A2", "qval_A1", "qval_A2"]].sort_values("A1", ascending=False)
    print("Top Archetype I genes:")
    print(summary.head(10))

    adata.write(args.output_h5ad)
    print(f"Results written to {args.output_h5ad}")
    print(f"UMAP plot saved to {args.output_image}")


if __name__ == "__main__":
    main()
