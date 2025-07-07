"""Benchmarking script comparing BioRSP directional enrichment
metrics against traditional cluster-based differential expression.

This script generates a synthetic 2D embedding of single cells with
four directional gene expression patterns. It computes directional
metrics A1/A2 using the SpatialRSP package and compares these metrics
to cluster-wise differential expression rankings.

Outputs include figures, correlation statistics and a CSV summary of
metric values. All random processes use a fixed seed for reproducibility.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scanpy as sc

import spatialrsp as rsp
from spatialrsp.utils.transform import cartesian_to_polar
from importlib.metadata import version, PackageNotFoundError


# --------------------------- Utility functions --------------------------- #

def print_versions():
    """Print versions of major packages used in the benchmark."""
    try:
        rsp_ver = version("spatialrsp")
    except PackageNotFoundError:
        rsp_ver = "unknown"

    versions = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "seaborn": sns.__version__,
        "scipy": scipy.__version__,
        "scanpy": sc.__version__,
        "spatialrsp": rsp_ver,
    }
    print("Package versions:")
    for k, v in versions.items():
        print(f"  {k:<10} {v}")


# -------------------------- Simulation functions ------------------------ #

def generate_embedding(n_cells: int, seed: int = 0) -> np.ndarray:
    """Create a synthetic 2D embedding using Gaussian mixtures."""
    rng = np.random.default_rng(seed)
    weights = [0.4, 0.35, 0.25]
    means = np.array([[0, 0], [2.5, 2.5], [-2.5, 2.5]])
    covs = [np.array([[1.0, 0.3], [0.3, 1.0]]),
            np.array([[1.2, -0.4], [-0.4, 1.2]]),
            np.array([[0.8, 0.2], [0.2, 0.8]])]
    parts = []
    for w, m, c in zip(weights, means, covs):
        size = int(n_cells * w)
        part = rng.multivariate_normal(mean=m, cov=c, size=size)
        parts.append(part)
    data = np.vstack(parts)
    if data.shape[0] < n_cells:
        extra = rng.multivariate_normal(means[0], covs[0],
                                        size=n_cells - data.shape[0])
        data = np.vstack([data, extra])
    return data


def simulate_genes(embedding: np.ndarray, seed: int = 0) -> pd.DataFrame:
    """Simulate four directional gene expression patterns."""
    rng = np.random.default_rng(seed)
    r, theta = cartesian_to_polar(embedding, vantage_point=(0, 0))
    max_r = r.max()

    directions = np.linspace(0, 2 * np.pi, 5)[:-1]  # 0,90,180,270 degrees
    kappa = 8.0  # concentration for von Mises
    genes = {}
    for i, mu in enumerate(directions):
        base = scipy.stats.vonmises.pdf(theta, kappa, loc=mu)
        expr = base * (r / max_r)
        noise = rng.normal(scale=0.2, size=expr.size)
        values = np.clip(expr + noise, a_min=0, a_max=None)
        genes[f"Gene{i+1}"] = values
    return pd.DataFrame(genes)


# -------------------------- Analysis workflow --------------------------- #

def run_scanpy_pipeline(adata: sc.AnnData) -> None:
    """Standard Scanpy preprocessing and clustering."""
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=1.0)
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon",
                            n_genes=adata.n_vars)


def extract_de_pvals(adata: sc.AnnData) -> pd.Series:
    """Return minimal adjusted p-value per gene from Scanpy results."""
    rgg = adata.uns["rank_genes_groups"]
    genes = adata.var_names.tolist()
    pvals = pd.DataFrame(rgg["pvals_adj"], index=genes).min(axis=1)
    return pvals


def compute_biorsp_metrics(adata: sc.AnnData, polar_key: str = "X_polar") -> pd.DataFrame:
    """Compute A1/A2 directional metrics for each gene."""
    theta_bg = adata.obsm[polar_key][:, 1]
    scanning_range = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    window = np.pi
    resolution = 60

    metrics = []
    for idx, gene in enumerate(adata.var_names):
        expr = adata.X[:, idx]
        threshold = np.quantile(expr, 0.75)
        mask = expr > threshold
        theta_fg = theta_bg[mask]
        fg_curve, _, bg_curve = rsp.spatial.compute_rsp(
            theta_fg, theta_bg, window, resolution, scanning_range,
            mode="absolute", normalize=False
        )
        a1 = rsp.spatial.compute_A1(fg_curve, bg_curve)
        a2 = rsp.spatial.compute_A2(fg_curve, bg_curve)
        metrics.append({"gene": gene, "A1": a1, "A2": a2})
    return pd.DataFrame(metrics).set_index("gene")


# ----------------------------- Visualization ---------------------------- #

def plot_gene_umap(adata: sc.AnnData, genes: list[str], outdir: str) -> None:
    """Save UMAP scatter plots colored by gene expression."""
    os.makedirs(outdir, exist_ok=True)
    for gene in genes:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        sc.pl.embedding(
            adata,
            basis="synthetic",
            color=gene,
            ax=ax,
            show=False,
            title=f"{gene} expression",
        )
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"umap_{gene}.pdf"))
        plt.close(fig)


def plot_angular_densities(adata: sc.AnnData, genes: list[str], outdir: str) -> None:
    """Plot angular density for each gene."""
    theta = adata.obsm["X_polar"][:, 1]
    os.makedirs(outdir, exist_ok=True)
    bins = np.linspace(-np.pi, np.pi, 50)
    for gene in genes:
        expr = adata[:, gene].X.ravel()
        threshold = np.quantile(expr, 0.75)
        mask = expr > threshold
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 4), dpi=300)
        ax.hist(theta[mask], bins=bins, alpha=0.8)
        ax.set_title(f"Angular density: {gene}")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"angular_{gene}.pdf"))
        plt.close(fig)


def plot_rank_scatter(df: pd.DataFrame, outdir: str) -> None:
    """Scatter plot comparing BioRSP vs DE rankings."""
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    sns.scatterplot(x=-np.log10(df["de_pval"]), y=df["A1"], ax=axes[0])
    axes[0].set_xlabel("-log10 DE p-value")
    axes[0].set_ylabel("A1")
    sns.scatterplot(x=-np.log10(df["de_pval"]), y=df["A2"], ax=axes[1])
    axes[1].set_xlabel("-log10 DE p-value")
    axes[1].set_ylabel("A2")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "ranking_scatter.pdf"))
    plt.close(fig)


def plot_correlation_bars(corr_a1: float, corr_a2: float, outdir: str) -> None:
    """Bar chart summarizing rank correlation between methods."""
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 5), dpi=300)
    sns.barplot(x=["A1", "A2"], y=[corr_a1, corr_a2], ax=ax, palette="viridis")
    ax.set_ylabel("Spearman correlation")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "correlations.pdf"))
    plt.close(fig)


# ------------------------------ Main logic ------------------------------ #

def main(outdir: str = "results", seed: int = 0):
    np.random.seed(seed)
    print_versions()

    embedding = generate_embedding(10_000, seed)
    gene_df = simulate_genes(embedding, seed)

    adata = sc.AnnData(gene_df)
    adata.obsm["X_synthetic"] = embedding
    adata.obsm["X_polar"] = np.column_stack(
        cartesian_to_polar(embedding, vantage_point=(0, 0))
    )
    adata.var_names = gene_df.columns
    adata.obs_names = [f"cell{i}" for i in range(adata.n_obs)]

    # Register synthetic embedding for plotting convenience
    adata.obsm["X_synthetic"].dtype = float
    sc.pl.embedding = sc.pl.embedding  # hush linting

    run_scanpy_pipeline(adata)
    de_pvals = extract_de_pvals(adata)

    rsp_metrics = compute_biorsp_metrics(adata)
    rsp_metrics["de_pval"] = de_pvals.loc[rsp_metrics.index]

    df = rsp_metrics.copy()
    df["rank_de"] = df["de_pval"].rank(method="average")
    df["rank_A1"] = (-df["A1"]).rank(method="average")
    df["rank_A2"] = (-df["A2"]).rank(method="average")

    corr_a1 = scipy.stats.spearmanr(df["rank_de"], df["rank_A1"]).correlation
    corr_a2 = scipy.stats.spearmanr(df["rank_de"], df["rank_A2"]).correlation

    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "gene_metrics.csv"))
    with open(os.path.join(outdir, "correlations.txt"), "w") as fh:
        fh.write(f"Spearman correlation (DE vs A1): {corr_a1:.3f}\n")
        fh.write(f"Spearman correlation (DE vs A2): {corr_a2:.3f}\n")

    plot_gene_umap(adata, list(df.index), outdir)
    plot_angular_densities(adata, list(df.index), outdir)
    plot_rank_scatter(df, outdir)
    plot_correlation_bars(corr_a1, corr_a2, outdir)


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "results"
    main(outdir)
