"""Interactive Plotly counterparts of the matplotlib plots used by the HTML
report.

Each function returns a ``plotly.graph_objects.Figure`` so the report
template can call ``fig.to_html(include_plotlyjs="cdn", full_html=False)``
and inline the result.

Plotly is an optional dependency. Functions import it lazily and raise a
friendly ImportError when the user lacks the report extras.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl

from .._style import PALETTE
from ..methyldata import MethylData


def _require_plotly():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for interactive report figures. "
            "Install with: pip install 'epykit[report]'"
        ) from exc


def volcano_plotly(md: MethylData, *, alpha: float = 0.05, min_abs_diff: float = 0.1):
    go = _require_plotly()
    dmc = md.dmc
    if dmc is None:
        return None
    p_col = "qvalue" if "qvalue" in dmc.columns else "pvalue"
    diff = dmc["meth_diff"].to_numpy()
    pval = dmc[p_col].to_numpy()
    y = -np.log10(np.maximum(pval, 1e-300))

    sig = (pval < alpha) & (np.abs(diff) >= min_abs_diff)
    hyper = sig & (diff > 0)
    hypo = sig & (diff < 0)
    ns = ~sig

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=diff[ns], y=y[ns], mode="markers",
        marker=dict(size=4, color=PALETTE["neutral"], opacity=0.4),
        name="ns", hoverinfo="skip",
    ))
    fig.add_trace(go.Scattergl(
        x=diff[hypo], y=y[hypo], mode="markers",
        marker=dict(size=5, color=PALETTE["hypo"], opacity=0.7),
        name=f"hypo ({int(hypo.sum())})",
    ))
    fig.add_trace(go.Scattergl(
        x=diff[hyper], y=y[hyper], mode="markers",
        marker=dict(size=5, color=PALETTE["hyper"], opacity=0.7),
        name=f"hyper ({int(hyper.sum())})",
    ))
    fig.add_hline(y=-np.log10(alpha), line=dict(color="grey", dash="dash", width=1))
    fig.add_vline(x=min_abs_diff, line=dict(color="grey", dash="dash", width=1))
    fig.add_vline(x=-min_abs_diff, line=dict(color="grey", dash="dash", width=1))
    fig.update_layout(
        title="DMC volcano",
        xaxis_title="Methylation difference (treatment - control)",
        yaxis_title=f"-log_1_0({p_col})",
        template="simple_white",
        height=420,
    )
    return fig


def ma_plot_plotly(md: MethylData, *, alpha: float = 0.05, min_abs_diff: float = 0.1):
    go = _require_plotly()
    dmc = md.dmc
    if dmc is None:
        return None
    p_col = "qvalue" if "qvalue" in dmc.columns else "pvalue"
    diff = dmc["meth_diff"].to_numpy()
    pval = dmc[p_col].to_numpy()
    mean_beta = (
        dmc["mean_beta_case"].to_numpy()
        + dmc["mean_beta_control"].to_numpy()
    ) / 2.0
    sig = (pval < alpha) & (np.abs(diff) >= min_abs_diff)
    hyper = sig & (diff > 0)
    hypo = sig & (diff < 0)
    ns = ~sig

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=mean_beta[ns], y=diff[ns], mode="markers",
        marker=dict(size=4, color=PALETTE["neutral"], opacity=0.4),
        name="ns", hoverinfo="skip",
    ))
    fig.add_trace(go.Scattergl(
        x=mean_beta[hypo], y=diff[hypo], mode="markers",
        marker=dict(size=5, color=PALETTE["hypo"], opacity=0.7),
        name=f"hypo ({int(hypo.sum())})",
    ))
    fig.add_trace(go.Scattergl(
        x=mean_beta[hyper], y=diff[hyper], mode="markers",
        marker=dict(size=5, color=PALETTE["hyper"], opacity=0.7),
        name=f"hyper ({int(hyper.sum())})",
    ))
    fig.add_hline(y=0, line=dict(color="black", width=1))
    fig.add_hline(y=min_abs_diff, line=dict(color="grey", dash="dash", width=1))
    fig.add_hline(y=-min_abs_diff, line=dict(color="grey", dash="dash", width=1))
    fig.update_layout(
        title="MA plot",
        xaxis_title="Mean methylation",
        yaxis_title="Methylation difference (treatment - control)",
        template="simple_white",
        height=420,
    )
    return fig


def manhattan_plotly(md: MethylData, *, alpha: float = 0.05):
    go = _require_plotly()
    dmc = md.dmc
    if dmc is None or "chrom" not in dmc.columns or "pos" not in dmc.columns:
        return None
    p_col = "qvalue" if "qvalue" in dmc.columns else "pvalue"
    dmc_sorted = dmc.sort(["chrom", "pos"])

    fig = go.Figure()
    chroms = dmc_sorted["chrom"].unique().to_list()
    order = (
        [f"chr{i}" for i in range(1, 23)]
        + [f"chr{c}" for c in ("X", "Y", "M")]
        + [c for c in chroms if c not in {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM"}]
    )
    cumulative = 0.0
    ticks_pos: list[float] = []
    ticks_label: list[str] = []
    colors = [PALETTE["hypo"], PALETTE["hyper"]]
    for idx, c in enumerate(order):
        if c not in chroms:
            continue
        sub = dmc_sorted.filter(pl.col("chrom") == c)
        if len(sub) == 0:
            continue
        pos = sub["pos"].to_numpy()
        pvals = sub[p_col].to_numpy()
        y = -np.log10(np.maximum(pvals, 1e-300))
        x = cumulative + pos
        fig.add_trace(go.Scattergl(
            x=x, y=y, mode="markers",
            marker=dict(size=3, color=colors[idx % 2], opacity=0.7),
            name=c, hoverinfo="skip", showlegend=False,
        ))
        mid = cumulative + (pos.max() - pos.min()) / 2.0
        ticks_pos.append(mid)
        ticks_label.append(c.replace("chr", ""))
        cumulative += float(pos.max()) + 1e7
    fig.add_hline(y=-np.log10(alpha), line=dict(color="red", dash="dash"))
    fig.update_layout(
        title="Manhattan plot",
        xaxis=dict(tickvals=ticks_pos, ticktext=ticks_label, title="Chromosome"),
        yaxis_title=f"-log_1_0({p_col})",
        template="simple_white",
        height=320,
    )
    return fig


def coverage_histogram_plotly(md: MethylData, *, bins: int = 100, max_points: int = 200_000):
    go = _require_plotly()
    pattern = f"{md.store}/sample=*/chrom=*/part-*.parquet"
    total = (
        pl.scan_parquet(pattern).select(pl.len()).collect()
    ).item()
    if total <= max_points:
        cov = (
            pl.scan_parquet(pattern).select("coverage").collect()["coverage"].to_numpy()
        )
    else:
        k = max(1, total // max_points)
        cov = (
            pl.scan_parquet(pattern)
            .select("coverage")
            .with_row_index("_row_num")
            .filter(pl.col("_row_num") % k == 0)
            .drop("_row_num")
            .collect()["coverage"]
            .to_numpy()
        )
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=cov, nbinsx=bins, marker_color=PALETTE["neutral"]))
    fig.update_layout(
        title="Coverage histogram",
        xaxis_title="Coverage",
        yaxis_title="CpG count",
        template="simple_white",
        height=320,
    )
    return fig


def pca_plotly(md: MethylData, *, n_sites: int = 10000):
    go = _require_plotly()
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return None

    samples = md.obs.get_column("sample_id").to_list()
    if "group" in md.obs.columns:
        groups = md.obs.get_column("group").to_list()
        group_col = "group"
    elif "treatment" in md.obs.columns:
        groups = md.obs.get_column("treatment").to_list()
        group_col = "treatment"
    else:
        groups = ["all"] * len(samples)
        group_col = "all"

    # Common sites across all samples
    common = None
    for s in samples:
        sites = (
            pl.scan_parquet(f"{md.store}/sample={s}/chrom=*/part-*.parquet")
            .select(["chrom", "pos"]).collect().unique()
        )
        common = sites if common is None else common.join(sites, on=["chrom", "pos"], how="inner")
    if common is None or len(common) == 0:
        return None
    if len(common) > n_sites:
        common = common.sample(n_sites, seed=42)

    parts = []
    for s in samples:
        sample_df = (
            pl.scan_parquet(f"{md.store}/sample={s}/chrom=*/part-*.parquet")
            .select(["chrom", "pos", "N_meth", "coverage", "sample"])
            .join(common.lazy(), on=["chrom", "pos"], how="inner")
            .collect()
        )
        parts.append(sample_df)
    all_data = pl.concat(parts).with_columns(
        pl.when(pl.col("coverage") > 0)
          .then(pl.col("N_meth") / pl.col("coverage"))
          .otherwise(None).alias("beta")
    )
    pivot = all_data.pivot(values="beta", index=["chrom", "pos"], on="sample", aggregate_function="mean")
    matrix = pivot.select(samples).to_numpy()
    mask = ~np.isnan(matrix).any(axis=1)
    matrix = matrix[mask]
    if matrix.shape[0] < 2:
        return None
    matrix = matrix.T
    pca = PCA(n_components=2)
    coords = pca.fit_transform(matrix)

    fig = go.Figure()
    unique = list(dict.fromkeys(groups))
    palette_cycle = [PALETTE["control"], PALETTE["treatment"], PALETTE["island"], PALETTE["shelf"], PALETTE["neutral"]]
    for i, g in enumerate(unique):
        mask_g = np.array([gg == g for gg in groups])
        fig.add_trace(go.Scatter(
            x=coords[mask_g, 0], y=coords[mask_g, 1],
            mode="markers+text",
            text=[s for s, m in zip(samples, mask_g) if m],
            textposition="top center",
            marker=dict(size=12, color=palette_cycle[i % len(palette_cycle)]),
            name=str(g),
        ))
    fig.update_layout(
        title="PCA",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
        template="simple_white",
        height=420,
        legend_title=group_col,
    )
    return fig


def feature_pie_plotly(md: MethylData):
    """Pie chart of feature_type distribution on the annotated DMC table."""
    go = _require_plotly()
    dmc = md.dmc
    if dmc is None or "feature_type" not in dmc.columns:
        return None
    counts = dmc.group_by("feature_type").len().sort("len", descending=True)
    fig = go.Figure(data=[go.Pie(
        labels=counts["feature_type"].to_list(),
        values=counts["len"].to_list(),
        hole=0.35,
    )])
    fig.update_layout(title="Genomic context (feature_type)", template="simple_white", height=360)
    return fig


def cpg_island_pie_plotly(md: MethylData):
    go = _require_plotly()
    dmc = md.dmc
    if dmc is None or "cpg_context" not in dmc.columns:
        return None
    counts = dmc.group_by("cpg_context").len().sort("len", descending=True)
    fig = go.Figure(data=[go.Pie(
        labels=counts["cpg_context"].to_list(),
        values=counts["len"].to_list(),
        hole=0.35,
    )])
    fig.update_layout(title="CpG-island context", template="simple_white", height=360)
    return fig


def tss_metaplot_plotly(md: MethylData, gtf_path: str, *, window_bp: int = 2000,
                        n_bins: int = 100, group_by: Optional[str] = "group",
                        max_genes: Optional[int] = None):
    """Plotly-rendered version of :func:`pl.tss_metaplot`."""
    go = _require_plotly()
    from .metaplot import _tss_table_from_gtf

    samples = md.obs.get_column("sample_id").to_list()
    if not samples:
        return None

    tss = _tss_table_from_gtf(gtf_path)
    if len(tss) == 0:
        return None
    if max_genes is not None and len(tss) > max_genes:
        tss = tss.head(max_genes)

    bin_size = (2 * window_bp) / n_bins
    sum_beta = np.zeros((len(samples), n_bins), dtype=np.float64)
    count = np.zeros((len(samples), n_bins), dtype=np.int64)
    sample_idx = {s: i for i, s in enumerate(samples)}

    chroms = sorted(set(tss["chrom"].to_list()))
    for chrom in chroms:
        tss_chrom = tss.filter(pl.col("chrom") == chrom)
        if len(tss_chrom) == 0:
            continue
        tss_positions = tss_chrom["tss"].to_numpy()
        strands = np.array(
            [1 if s != "-" else -1 for s in tss_chrom["strand"].to_list()],
            dtype=np.int8,
        )
        pattern = f"{md.store}/sample=*/chrom={chrom}/part-*.parquet"
        try:
            chrom_df = (
                pl.scan_parquet(pattern)
                .select(["pos", "sample", "N_meth", "coverage"])
                .filter(pl.col("coverage") > 0)
                .collect()
            )
        except Exception:
            continue
        if len(chrom_df) == 0:
            continue
        positions = chrom_df["pos"].to_numpy().astype(np.int64)
        samples_arr = chrom_df["sample"].to_list()
        betas = (
            chrom_df["N_meth"].to_numpy().astype(np.float64)
            / chrom_df["coverage"].to_numpy().astype(np.float64)
        )
        order = np.argsort(positions, kind="mergesort")
        positions = positions[order]
        betas = betas[order]
        samples_arr = [samples_arr[i] for i in order]
        samples_idx_arr = np.fromiter(
            (sample_idx.get(s, -1) for s in samples_arr),
            count=len(samples_arr), dtype=np.int32,
        )

        for tss_pos, strand in zip(tss_positions, strands):
            lo = tss_pos - window_bp
            hi = tss_pos + window_bp
            left = np.searchsorted(positions, lo, side="left")
            right = np.searchsorted(positions, hi, side="left")
            if right <= left:
                continue
            rel = (positions[left:right] - tss_pos) * strand
            bins = np.floor((rel + window_bp) / bin_size).astype(np.int64)
            np.clip(bins, 0, n_bins - 1, out=bins)
            sub_samples = samples_idx_arr[left:right]
            sub_betas = betas[left:right]
            mask = sub_samples >= 0
            if not mask.any():
                continue
            np.add.at(sum_beta, (sub_samples[mask], bins[mask]), sub_betas[mask])
            np.add.at(count, (sub_samples[mask], bins[mask]), 1)

    with np.errstate(invalid="ignore"):
        mean_beta = np.where(count > 0, sum_beta / count, np.nan)

    x = np.linspace(-window_bp, window_bp, n_bins, endpoint=False) + bin_size / 2.0

    fig = go.Figure()
    if group_by and group_by in md.obs.columns:
        groups = md.obs.get_column(group_by).to_list()
        unique = sorted(set(groups))
        palette_cycle = [PALETTE["control"], PALETTE["treatment"], PALETTE["island"], PALETTE["shelf"]]
        for i, samp in enumerate(samples):
            fig.add_trace(go.Scatter(
                x=x, y=mean_beta[i], mode="lines",
                line=dict(color=palette_cycle[unique.index(groups[i]) % len(palette_cycle)],
                          width=1), opacity=0.25,
                name=str(samp), showlegend=False,
            ))
        for j, g in enumerate(unique):
            mask = np.array([gg == g for gg in groups])
            if not mask.any():
                continue
            mean = np.nanmean(mean_beta[mask], axis=0)
            fig.add_trace(go.Scatter(
                x=x, y=mean, mode="lines",
                line=dict(color=palette_cycle[j % len(palette_cycle)], width=2.5),
                name=str(g),
            ))
    else:
        for i, samp in enumerate(samples):
            fig.add_trace(go.Scatter(
                x=x, y=mean_beta[i], mode="lines", name=str(samp),
            ))
    fig.add_vline(x=0, line=dict(color="black", dash="dash", width=1))
    fig.update_layout(
        title=f"TSS metaplot (+/-{window_bp} bp)",
        xaxis_title="Distance from TSS (bp)",
        yaxis_title="Mean beta",
        template="simple_white",
        height=380,
    )
    return fig


__all__ = [
    "volcano_plotly",
    "ma_plot_plotly",
    "manhattan_plotly",
    "coverage_histogram_plotly",
    "pca_plotly",
    "feature_pie_plotly",
    "cpg_island_pie_plotly",
    "tss_metaplot_plotly",
]
