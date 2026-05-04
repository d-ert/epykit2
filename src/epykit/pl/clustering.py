from __future__ import annotations

import numpy as np
import polars as pl

from .._style import PALETTE
from ._utils import _get_ax, _save_fig
from ..methyldata import MethylData


def pca(
    md: MethylData,
    *,
    n_sites: int = 10000,
    ax=None,
    figsize=(6, 5),
    save: str | None = None,
):
    """PCA of per-sample methylation profiles.

    Samples sites randomly (or every Nth site if total < n_sites*2)
    to avoid OOM on large datasets.

    Parameters
    ----------
    md : MethylData
        Methylation data object with filtered store.
    n_sites : int
        Number of sites to sample for PCA. Default 10000.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, create new figure.
    figsize : tuple
        Figure size (width, height) in inches.
    save : str, optional
        Path to save figure. If None, don't save.

    Returns
    -------
    fig, ax : (Figure, Axes)
        Matplotlib figure and axes.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError("scikit-learn is required for PCA. Install with: pip install scikit-learn") from exc

    # Get list of samples
    samples = md.obs.get_column("sample_id").to_list()
    if "treatment" not in md.obs.columns and "group" not in md.obs.columns:
        raise ValueError("md.obs must have 'treatment' or 'group' column for coloring. Run ep.tl.qc(md) first.")

    group_col = "group" if "group" in md.obs.columns else "treatment"
    groups = md.obs.get_column(group_col).to_list()

    # Count total sites
    total_sites = (
        pl.scan_parquet(f"{md.store}/sample=*/chrom=*/part-*.parquet")
        .select(pl.count())
        .collect()
    ).item()

    # Determine sampling strategy
    if total_sites <= n_sites * 2:
        # Load all sites
        all_data = pl.scan_parquet(f"{md.store}/sample=*/chrom=*/part-*.parquet").collect()
    else:
        # Sample every Kth site to get ~n_sites
        k = max(1, total_sites // n_sites)
        all_data = (
            pl.scan_parquet(f"{md.store}/sample=*/chrom=*/part-*.parquet")
            .select(["chrom", "pos", "sample", "N_meth", "coverage"])
            .collect()
        )
        # Use row_number to sample every kth row
        all_data = all_data.with_row_count("row_num").filter(pl.col("row_num") % k == 0)

    # Compute beta values
    all_data = all_data.with_columns(
        pl.when(pl.col("coverage") > 0)
        .then(pl.col("N_meth") / pl.col("coverage"))
        .otherwise(None)
        .alias("beta")
    )

    # Pivot to get sites x samples matrix
    pivot = all_data.pivot(values="beta", index=["chrom", "pos"], on="sample", aggregate_function="mean")

    # Ensure all samples present (fill missing with NaN)
    for sample in samples:
        if sample not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(None).alias(sample))

    # Get matrix and remove NaN rows
    matrix = pivot.select(samples).to_numpy()
    matrix = matrix[~np.isnan(matrix).any(axis=1)]

    if matrix.shape[0] < 2:
        raise ValueError("Not enough valid sites for PCA after filtering NaNs")

    # Fit PCA
    pca_fit = PCA(n_components=2)
    coords = pca_fit.fit_transform(matrix)

    # Create color mapping for groups
    unique_groups = sorted(set(groups))
    group_colors = {
        unique_groups[0]: PALETTE["control"] if unique_groups[0] in ["control", "ctrl"] else PALETTE["hypo"],
        unique_groups[1] if len(unique_groups) > 1 else None: PALETTE["treatment"] if len(unique_groups) > 1 and unique_groups[1] in ["treatment", "cd55"] else PALETTE["hyper"],
    }
    if None in group_colors:
        del group_colors[None]

    # Plot
    fig, ax = _get_ax(ax, figsize)
    for group in unique_groups:
        mask = np.array([g == group for g in groups])
        if mask.sum() > 0:
            color = group_colors.get(group, PALETTE["neutral"])
            ax.scatter(coords[mask, 0], coords[mask, 1], s=100, alpha=0.6, label=group, color=color)

    ax.set_xlabel(f"PC1 ({pca_fit.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca_fit.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA of methylation profiles")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save:
        _save_fig(md, fig, save)
    return fig, ax


__all__ = ["pca"]
