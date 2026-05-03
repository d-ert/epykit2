from __future__ import annotations

import numpy as np
import polars as pl

from .._style import PALETTE
from ._utils import _get_ax, _save_fig
from ..methyldata import MethylData


def volcano(
    md: MethylData,
    *,
    alpha: float = 0.05,
    min_abs_diff: float = 0.1,
    ax=None,
    figsize=(6, 5),
    save: str | None = None,
):
    dmc = md.dmc
    if dmc is None:
        raise ValueError("Run ep.tl.dmc(md) first")

    p_col = "qvalue" if "qvalue" in dmc.columns else "pvalue"
    diff = dmc["meth_diff"].to_numpy()
    pval = dmc[p_col].to_numpy()
    y = -np.log10(np.maximum(pval, 1e-300))

    sig = (pval < alpha) & (np.abs(diff) >= min_abs_diff)
    hyper = sig & (diff > 0)
    hypo = sig & (diff < 0)
    ns = ~sig

    fig, ax = _get_ax(ax, figsize)
    ax.scatter(diff[ns], y[ns], s=4, color=PALETTE["neutral"], alpha=0.4, rasterized=True)
    ax.scatter(diff[hypo], y[hypo], s=4, color=PALETTE["hypo"], alpha=0.7, rasterized=True)
    ax.scatter(diff[hyper], y[hyper], s=4, color=PALETTE["hyper"], alpha=0.7, rasterized=True)

    ax.axhline(-np.log10(alpha), color="grey", lw=0.8, ls="--")
    ax.axvline(min_abs_diff, color="grey", lw=0.8, ls="--")
    ax.axvline(-min_abs_diff, color="grey", lw=0.8, ls="--")

    n_hyper = int(hyper.sum())
    n_hypo = int(hypo.sum())
    ax.set_title(f"DMC volcano  |  hyper={n_hyper:,}  hypo={n_hypo:,}")
    ax.set_xlabel("Methylation difference (treatment − control)")
    ax.set_ylabel(f"−log₁₀({p_col})")

    if save:
        _save_fig(md, fig, save)
    return fig, ax


def ma_plot(
    md: MethylData,
    *,
    alpha: float = 0.05,
    min_abs_diff: float = 0.1,
    ax=None,
    figsize=(7, 5),
    save: str | None = None,
):
    """MA plot: mean beta vs methylation difference.

    x-axis: mean methylation across treatment and control
    y-axis: methylation difference (treatment − control)
    color: hypermethylated (red), hypomethylated (blue), not significant (grey)
    """
    dmc = md.dmc
    if dmc is None:
        raise ValueError("Run ep.tl.dmc(md) first")

    p_col = "qvalue" if "qvalue" in dmc.columns else "pvalue"
    diff = dmc["meth_diff"].to_numpy()
    pval = dmc[p_col].to_numpy()
    mean_beta = dmc["mean_beta"].to_numpy() if "mean_beta" in dmc.columns else (
        dmc.select(pl.col("*").exclude("meth_diff")).mean().to_dicts()[0].get("beta", 0.5)
    )

    sig = (pval < alpha) & (np.abs(diff) >= min_abs_diff)
    hyper = sig & (diff > 0)
    hypo = sig & (diff < 0)
    ns = ~sig

    fig, ax = _get_ax(ax, figsize)
    ax.scatter(mean_beta[ns], diff[ns], s=4, color=PALETTE["neutral"], alpha=0.4, rasterized=True)
    ax.scatter(mean_beta[hypo], diff[hypo], s=4, color=PALETTE["hypo"], alpha=0.7, rasterized=True)
    ax.scatter(mean_beta[hyper], diff[hyper], s=4, color=PALETTE["hyper"], alpha=0.7, rasterized=True)

    ax.axhline(0, color="black", lw=1)
    ax.axhline(min_abs_diff, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(-min_abs_diff, color="grey", lw=0.8, ls="--", alpha=0.5)

    n_hyper = int(hyper.sum())
    n_hypo = int(hypo.sum())
    ax.set_title(f"MA plot  |  hyper={n_hyper:,}  hypo={n_hypo:,}")
    ax.set_xlabel("Mean methylation")
    ax.set_ylabel("Methylation difference (treatment − control)")

    if save:
        _save_fig(md, fig, save)
    return fig, ax


def manhattan(
    md: MethylData,
    *,
    alpha: float = 0.05,
    ax=None,
    figsize=(14, 4),
    save: str | None = None,
):
    """Manhattan plot: genome-wide significance.

    x-axis: chromosome position
    y-axis: −log₁₀(p-value)
    color: alternates by chromosome
    """
    dmc = md.dmc
    if dmc is None:
        raise ValueError("Run ep.tl.dmc(md) first")

    if "chrom" not in dmc.columns or "pos" not in dmc.columns:
        raise ValueError("DMC table must contain 'chrom' and 'pos' columns. Run ep.tl.annotate(md) first.")

    p_col = "qvalue" if "qvalue" in dmc.columns else "pvalue"

    dmc_sorted = dmc.sort(["chrom", "pos"])
    chroms = dmc_sorted["chrom"].unique().to_list()

    chrom_order = (
        [f"chr{i}" for i in range(1, 23)]
        + [f"chr{c}" for c in ["X", "Y", "M"]]
        + [c for c in chroms if c not in [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]]
    )

    fig, ax = _get_ax(ax, figsize)

    cumulative_pos = 0
    chrom_offsets = {}
    colors = [PALETTE["hypo"], PALETTE["hyper"]]

    for chrom_idx, chrom in enumerate(chrom_order):
        if chrom not in chroms:
            continue
        chrom_data = dmc_sorted.filter(pl.col("chrom") == chrom)
        if len(chrom_data) == 0:
            continue

        positions = chrom_data["pos"].to_numpy()
        pvals = chrom_data[p_col].to_numpy()
        y = -np.log10(np.maximum(pvals, 1e-300))

        x_coords = cumulative_pos + positions
        color = colors[chrom_idx % 2]
        ax.scatter(x_coords, y, s=3, color=color, alpha=0.6, rasterized=True)

        chrom_offsets[chrom] = (cumulative_pos, cumulative_pos + positions.max())
        cumulative_pos += positions.max() + 1e7  # add gap between chromosomes

    ax.axhline(-np.log10(alpha), color="red", lw=1, ls="--", label=f"α={alpha}")
    ax.set_xlabel("Chromosome")
    ax.set_ylabel(f"−log₁₀({p_col})")
    ax.set_title("Manhattan plot")
    ax.legend()

    # Set chromosome labels at midpoints
    tick_positions = []
    tick_labels = []
    for chrom in chrom_order:
        if chrom in chrom_offsets:
            start, end = chrom_offsets[chrom]
            mid = (start + end) / 2
            tick_positions.append(mid)
            tick_labels.append(chrom.replace("chr", ""))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)

    if save:
        _save_fig(md, fig, save)
    return fig, ax


__all__ = ["volcano", "ma_plot", "manhattan"]
