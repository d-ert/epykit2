from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ..methyldata import MethylData


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib") from exc
    return plt


def _save_figure(md: MethylData, fig, name: str, out_dir: str | None = None) -> str:
    out = Path(out_dir or "figures")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if "figures" not in md.uns:
        md.uns["figures"] = {}
    md.uns["figures"][name] = str(path)
    return str(path)


def volcano(md: MethylData, out_dir: str | None = None) -> str:
    dmc = md.dmc
    if dmc is None:
        raise ValueError("No DMC results available. Run ep.tl.dmc(md) first.")

    p_col = "qvalue" if "qvalue" in dmc.columns else "pvalue"
    y = -np.log10(np.maximum(dmc[p_col].to_numpy(), 1e-300))
    x = dmc["meth_diff"].to_numpy()

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=5, alpha=0.5)
    ax.set_xlabel("Methylation difference (case - control)")
    ax.set_ylabel(f"-log10({p_col})")
    ax.set_title("DMC volcano")
    return _save_figure(md, fig, "volcano", out_dir=out_dir)


def coverage_histogram(md: MethylData, bins: int = 100, out_dir: str | None = None) -> str:
    cov = (
        pl.scan_parquet(f"{md.store}/sample=*/chrom=*/part-*.parquet")
        .select("coverage")
        .collect()["coverage"]
        .to_numpy()
    )
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(cov, bins=bins)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Count")
    ax.set_title("Coverage histogram")
    return _save_figure(md, fig, "coverage_histogram", out_dir=out_dir)


def methylation_heatmap(md: MethylData, n_top: int = 1000, out_dir: str | None = None) -> str:
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

    pivot = site_df.pivot(
        values="beta",
        index=["chrom", "pos"],
        on="sample",
        aggregate_function="mean",
    )
    for sample in samples:
        if sample not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(None).alias(sample))

    matrix = pivot.select(samples).to_numpy()
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, cmap="viridis", ax=ax)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Top DMC sites")
    ax.set_title(f"Methylation heatmap (top {n_top})")
    return _save_figure(md, fig, "methylation_heatmap", out_dir=out_dir)


def genomic_context_bar(md: MethylData, out_dir: str | None = None) -> str:
    dmc = md.dmc
    if dmc is None or "feature_type" not in dmc.columns:
        raise ValueError("No feature annotations found. Run ep.tl.annotate(md, gtf=...) first.")

    counts = dmc.group_by("feature_type").len().sort("len", descending=True)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts["feature_type"].to_list(), counts["len"].to_list())
    ax.set_xlabel("Feature type")
    ax.set_ylabel("Count")
    ax.set_title("Genomic context")
    ax.tick_params(axis="x", rotation=45)
    return _save_figure(md, fig, "genomic_context_bar", out_dir=out_dir)


def cpg_island_pie(md: MethylData, out_dir: str | None = None) -> str:
    dmc = md.dmc
    if dmc is None or "cpg_context" not in dmc.columns:
        raise ValueError("No CpG-island annotations found. Run ep.tl.annotate(md, cpg_islands=...) first.")

    counts = dmc.group_by("cpg_context").len().sort("len", descending=True)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(counts["len"].to_list(), labels=counts["cpg_context"].to_list(), autopct="%1.1f%%")
    ax.set_title("CpG island context")
    return _save_figure(md, fig, "cpg_island_pie", out_dir=out_dir)
