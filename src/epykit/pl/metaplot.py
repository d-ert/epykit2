"""TSS / gene-body metaplots.

Reuses the cached GTF parser in :mod:`epykit.annotate` to enumerate gene
TSS coordinates, then bins per-CpG β values into ``n_bins`` slots from
``-window_bp`` to ``+window_bp`` around each TSS. One line per sample,
optionally grouped by ``md.obs[group_by]``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl

from .._style import PALETTE
from ._utils import _get_ax, _save_fig
from ..methyldata import MethylData
from ..annotate import _parse_gtf_streaming


def _tss_table_from_gtf(gtf_path: str) -> pl.DataFrame:
    """Build a per-gene TSS DataFrame from a (cached) parsed GTF.

    Columns: chrom (Utf8), tss (Int64), strand (Utf8), gene_id (Utf8),
    gene_name (Utf8). TSS = Start for + strand, End-1 for - strand
    (GTF is 0-based half-open inside epykit's parser).
    """
    genes_pd, _ = _parse_gtf_streaming(gtf_path)
    if genes_pd is None or len(genes_pd) == 0:
        return pl.DataFrame(schema={
            "chrom": pl.Utf8, "tss": pl.Int64, "strand": pl.Utf8,
            "gene_id": pl.Utf8, "gene_name": pl.Utf8,
        })

    genes = pl.from_pandas(
        genes_pd[["Chromosome", "Start", "End", "Strand", "gene_id", "gene_name"]]
    ).rename({"Chromosome": "chrom", "Strand": "strand"})
    return (
        genes.with_columns(
            pl.when(pl.col("strand") == "-")
              .then(pl.col("End") - 1)
              .otherwise(pl.col("Start"))
              .cast(pl.Int64).alias("tss")
        )
        .select(["chrom", "tss", "strand", "gene_id", "gene_name"])
    )


def tss_metaplot(
    md: MethylData,
    gtf_path: str,
    *,
    window_bp: int = 2000,
    n_bins: int = 100,
    group_by: Optional[str] = "group",
    max_genes: Optional[int] = None,
    ax=None,
    figsize=(7, 4),
    save: str | None = None,
):
    """Plot mean β around the TSS, averaged across genes.

    For each sample, β values are pooled across all gene TSS in a
    ``±window_bp`` window, binned into ``n_bins`` slots based on relative
    position (sign-flipped for - strand so 5'→3' is left→right), and
    averaged. Each sample is drawn as a faint line, with one bold line
    per group (if ``group_by`` matches a column on ``md.obs``).

    Parameters
    ----------
    md : MethylData
        Must have a methylstore at ``md.store`` (i.e. one of
        ``read_bismark`` / ``read_nfcore_methylseq`` / ``load`` was run).
    gtf_path : str
        GTF / GFF3 (gz-allowed) for TSS coordinates. Uses epykit's
        bounded LRU GTF cache, so repeated calls within one process
        skip the streaming parse.
    window_bp : int
        Half-window width in base pairs (default 2000 → ±2 kb).
    n_bins : int
        Number of bins to summarise β within the window. Default 100.
    group_by : str or None
        Optional ``md.obs`` column for per-group means. Pass ``None`` to
        draw only the per-sample lines.
    max_genes : int, optional
        Cap the number of TSS used. Useful for fast smoke tests; None
        uses every gene in the GTF.

    Returns
    -------
    (Figure, Axes)
    """
    samples = md.obs.get_column("sample_id").to_list()
    if not samples:
        raise ValueError("md.obs has no samples")

    tss = _tss_table_from_gtf(gtf_path)
    if len(tss) == 0:
        raise ValueError(f"No gene records found in GTF {gtf_path!r}")
    if max_genes is not None and len(tss) > max_genes:
        tss = tss.head(max_genes)

    # Bin layout: bin = floor((rel + window_bp) / bin_size); range [-window, +window).
    bin_size = (2 * window_bp) / n_bins

    # mean β per (sample, bin). Allocate before the per-chrom loop.
    sum_beta = np.zeros((len(samples), n_bins), dtype=np.float64)
    count = np.zeros((len(samples), n_bins), dtype=np.int64)
    sample_idx = {s: i for i, s in enumerate(samples)}

    chroms = sorted(set(tss["chrom"].to_list()))
    for chrom in chroms:
        tss_chrom = tss.filter(pl.col("chrom") == chrom)
        if len(tss_chrom) == 0:
            continue
        tss_positions = tss_chrom["tss"].to_numpy()
        strands = np.array([1 if s == "+" or s != "-" else -1 for s in tss_chrom["strand"].to_list()], dtype=np.int8)

        # Read CpGs for this chromosome across all samples
        pattern = f"{md.store}/sample=*/chrom={chrom}/part-*.parquet"
        try:
            chrom_df = (
                pl.scan_parquet(pattern)
                .select(["pos", "sample", "N_meth", "coverage"])
                .filter(pl.col("coverage") > 0)
                .collect()
            )
        except Exception:  # e.g. no files for that chromosome
            continue
        if len(chrom_df) == 0:
            continue

        positions = chrom_df["pos"].to_numpy().astype(np.int64)
        samples_arr = chrom_df["sample"].to_list()
        betas = (chrom_df["N_meth"].to_numpy().astype(np.float64) /
                 chrom_df["coverage"].to_numpy().astype(np.float64))

        # For each TSS, find CpGs in window via binary search on sorted positions
        # positions need to be sorted; scan_parquet doesn't guarantee that, so sort.
        order = np.argsort(positions, kind="mergesort")
        positions = positions[order]
        betas = betas[order]
        samples_arr = [samples_arr[i] for i in order]
        samples_idx_arr = np.fromiter((sample_idx.get(s, -1) for s in samples_arr),
                                       count=len(samples_arr), dtype=np.int32)

        for tss_pos, strand in zip(tss_positions, strands):
            lo = tss_pos - window_bp
            hi = tss_pos + window_bp
            left = np.searchsorted(positions, lo, side="left")
            right = np.searchsorted(positions, hi, side="left")
            if right <= left:
                continue
            rel = (positions[left:right] - tss_pos) * strand
            # Bin index
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

    fig, ax = _get_ax(ax, figsize)

    if group_by and group_by in md.obs.columns:
        groups = md.obs.get_column(group_by).to_list()
        unique_groups = sorted(set(groups))
        group_palette = {
            g: PALETTE.get("treatment" if i else "control", PALETTE["neutral"])
            for i, g in enumerate(unique_groups)
        }
        # per-sample faint lines
        for i, samp in enumerate(samples):
            ax.plot(x, mean_beta[i], color=group_palette.get(groups[i], PALETTE["neutral"]),
                    alpha=0.25, linewidth=1)
        # per-group bold mean
        for g in unique_groups:
            mask = np.array([gg == g for gg in groups])
            if not mask.any():
                continue
            grp_mean = np.nanmean(mean_beta[mask], axis=0)
            ax.plot(x, grp_mean, color=group_palette.get(g, PALETTE["neutral"]),
                    linewidth=2.2, label=str(g))
        ax.legend(title=group_by, frameon=False)
    else:
        for i, samp in enumerate(samples):
            ax.plot(x, mean_beta[i], alpha=0.7, linewidth=1.2, label=samp)
        if len(samples) <= 12:
            ax.legend(frameon=False, fontsize=8)

    ax.axvline(0, color="black", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("Distance from TSS (bp)")
    ax.set_ylabel("Mean β")
    ax.set_title(f"TSS metaplot (±{window_bp} bp, n_bins={n_bins})")

    if save:
        _save_fig(md, fig, save)
    return fig, ax


__all__ = ["tss_metaplot"]
