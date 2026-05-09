from __future__ import annotations

from ._utils import _get_ax, _save_fig
from ..methyldata import MethylData
import polars as pl
import numpy as np


def coverage_histogram(md: MethylData, bins: int = 100, ax=None, figsize=(6, 4), save: str | None = None):
    """Plot histogram of coverage across all sites.
    
    For large datasets, samples every Kth site to avoid OOM.
    """
    # Count total sites — FIX-11: pl.count() removed in Polars ≥0.20; use pl.len()
    total_sites = (
        pl.scan_parquet(f"{md.store}/sample=*/chrom=*/part-*.parquet")
        .select(pl.len())
        .collect()
    ).item()
    
    # Determine sampling strategy
    if total_sites <= 1_000_000:
        # Small dataset: load all coverage values
        cov = (
            pl.scan_parquet(f"{md.store}/sample=*/chrom=*/part-*.parquet")
            .select("coverage")
            .collect()["coverage"]
            .to_numpy()
        )
    else:
        # Large dataset: sample every Kth site to get ~1M points
        k = max(1, total_sites // 1_000_000)
        cov = (
            pl.scan_parquet(f"{md.store}/sample=*/chrom=*/part-*.parquet")
            .select("coverage")
            .with_row_index("_row_num")
            .filter(pl.col("_row_num") % k == 0)
            .drop("_row_num")
            .collect()["coverage"]
            .to_numpy()
        )

    fig, ax = _get_ax(ax, figsize)
    ax.hist(cov, bins=bins, edgecolor="black")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Count")
    ax.set_title("Coverage histogram")

    if save:
        _save_fig(md, fig, save)
    return fig, ax


def methylation_heatmap(md: MethylData, n_top: int = 1000, ax=None, figsize=(8, 6), save: str | None = None):
    try:
        import seaborn as sns
    except ImportError as exc:
        raise ImportError("seaborn is required for heatmaps. Install with: pip install seaborn") from exc

    dmc = md.dmc
    if dmc is None:
        raise ValueError("No DMC results available. Run ep.tl.dmc(md) first.")

    top = (
        dmc
        .filter(pl.col("meth_diff").is_not_null())
        .with_columns(pl.col("meth_diff").abs().alias("abs_diff"))
        .sort("abs_diff", descending=True)
        .head(n_top)
        .select(["chrom", "pos"])
    )
    if len(top) == 0:
        raise ValueError("No DMC rows available to build heatmap")

    samples = md.obs.get_column("sample_id").to_list()
    
    # top is already a collected DataFrame from md.dmc, no need to call .collect()
    if len(top) == 0:
        raise ValueError("No top DMCs to build heatmap")
    
    # Process each sample separately to reduce memory footprint
    site_dfs = []
    for sample in samples:
        sample_df = (
            pl.scan_parquet(f"{md.store}/sample={sample}/chrom=*/part-*.parquet")
            .select(["chrom", "pos", "N_meth", "coverage"])
            .join(top.lazy(), on=["chrom", "pos"], how="inner")
            .collect()
        )
        if len(sample_df) > 0:
            sample_df = sample_df.with_columns(
                pl.when(pl.col("coverage") > 0)
                .then(pl.col("N_meth") / pl.col("coverage"))
                .otherwise(None)
                .alias("beta"),
                pl.lit(sample).alias("sample")
            )
            site_dfs.append(sample_df)
    
    if not site_dfs:
        raise ValueError("No sites found in store matching top DMCs")
    
    site_df = pl.concat(site_dfs)

    pivot = site_df.pivot(values="beta", index=["chrom", "pos"], on="sample", aggregate_function="mean")
    for sample in samples:
        if sample not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(None).alias(sample))

    matrix = pivot.select(samples).to_numpy()
    fig, ax = _get_ax(ax, figsize)
    sns.heatmap(matrix, cmap="viridis", ax=ax)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Top DMC sites")
    ax.set_title(f"Methylation heatmap (top {n_top})")

    if save:
        _save_fig(md, fig, save)
    return fig, ax


__all__ = ["coverage_histogram", "methylation_heatmap"]