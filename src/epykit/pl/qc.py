from __future__ import annotations

from ._utils import _get_ax, _save_fig
from ..methyldata import MethylData
import polars as pl
import numpy as np


def coverage_histogram(md: MethylData, bins: int = 100, ax=None, figsize=(6, 4), save: str | None = None):
    cov = (
        pl.scan_parquet(f"{md.store}/sample=*/chrom=*/part-*.parquet")
        .select("coverage")
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
    site_df = (
        pl.scan_parquet(f"{md.store}/sample=*/chrom=*/part-*.parquet")
        .join(top.lazy(), on=["chrom", "pos"], how="inner")
        .select(["chrom", "pos", "sample", "N_meth", "coverage"])
        .collect()
        .with_columns(
            pl.when(pl.col("coverage") > 0)
            .then(pl.col("N_meth") / pl.col("coverage"))
            .otherwise(None)
            .alias("beta")
        )
    )

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
