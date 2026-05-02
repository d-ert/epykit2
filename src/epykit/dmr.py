"""Differentially Methylated Region (DMR) calling.

Phase 2 of the epykit pipeline:
  - call_dmr_sliding_window: aggregates DMC sites into contiguous genomic
    windows using an overlapping sliding-window approach (MVP).
  - smooth_methylation_bsmooth: LOESS smoothing of per-sample beta values
    within each chromosome, used as a pre-processing step before DMR calling.

Design notes
------------
Overlapping windows (step_bp < window_bp) are used to avoid edge effects at
window boundaries.  Candidate windows that pass all filters are merged into
non-redundant DMR spans; site counts are then re-computed from the original
data over the merged span so that n_cpgs / n_significant are exact rather than
the max of contributing windows.

Direction consistency is assessed on the merged region: a DMR is called only
when the majority of sites agree on direction (hyper or hypo); ties are
discarded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

_DMR_EMPTY_SCHEMA = {
    "chrom":           pl.Utf8,
    "start":           pl.Int32,
    "end":             pl.Int32,
    "n_cpgs":          pl.Int32,
    "n_significant":   pl.Int32,
    "mean_meth_diff":  pl.Float32,
    "mean_pvalue":     pl.Float64,
    "dmr_type":        pl.Utf8,
}

_SMOOTH_EMPTY_SCHEMA = {
    "chrom":        pl.Utf8,
    "pos":          pl.Int32,
    "sample":       pl.Utf8,
    "beta_raw":     pl.Float32,
    "beta_smooth":  pl.Float32,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _merge_intervals(starts: list[int], ends: list[int]) -> list[tuple[int, int]]:
    """Merge overlapping (start, end) integer intervals.

    Parameters
    ----------
    starts, ends : list[int]
        Paired interval boundaries.

    Returns
    -------
    list of (start, end) tuples, sorted and non-overlapping.
    """
    if not starts:
        return []
    pairs = sorted(zip(starts, ends), key=lambda x: x[0])
    merged: list[tuple[int, int]] = [pairs[0]]
    for s, e in pairs[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _recompute_dmr_stats(
    chrom: str,
    start: int,
    end: int,
    positions: np.ndarray,
    meth_diffs: np.ndarray,
    pvals: np.ndarray,
    is_sig: np.ndarray,
    min_cpgs: int,
    min_sites_significant: int,
) -> dict | None:
    """Recompute accurate per-site statistics over a merged interval.

    Returns a record dict or None when the merged region fails any filter
    (insufficient CpGs, sites, or ambiguous direction).
    """
    mask  = (positions >= start) & (positions < end)
    n_cpgs = int(mask.sum())
    if n_cpgs < min_cpgs:
        return None

    n_sig = int(is_sig[mask].sum())
    if n_sig < min_sites_significant:
        return None

    window_diffs = meth_diffs[mask]
    window_pvals = pvals[mask]

    n_hyper = int((window_diffs > 0).sum())
    n_hypo  = int((window_diffs < 0).sum())
    if n_hyper == n_hypo:
        return None  # ambiguous direction

    return {
        "chrom":          chrom,
        "start":          start,
        "end":            end,
        "n_cpgs":         n_cpgs,
        "n_significant":  n_sig,
        "mean_meth_diff": float(np.float32(np.nanmean(window_diffs))),
        "mean_pvalue":    float(np.nanmean(window_pvals)),
        "dmr_type":       "hyper" if n_hyper > n_hypo else "hypo",
    }


# ---------------------------------------------------------------------------
# Public API — DMR calling
# ---------------------------------------------------------------------------

def call_dmr_sliding_window(
    dmc_results: pl.DataFrame,
    window_bp: int = 500,
    step_bp: int = 250,
    min_cpgs: int = 5,
    min_sites_significant: int = 3,
    alpha: float = 0.05,
    min_abs_meth_diff: float = 0.1,
) -> pl.DataFrame:
    """Call DMRs by aggregating DMC sites into overlapping sliding windows.

    Parameters
    ----------
    dmc_results : pl.DataFrame
        Output from ``process_chromosomes_dmc`` /
        ``apply_multiple_testing_correction``.
        Required columns: chrom, pos, meth_diff, pvalue.
        Optional: qvalue (used in preference to pvalue when present).
    window_bp : int
        Window width in base pairs (default 500 bp).
    step_bp : int
        Step size in base pairs (default 250 bp; must be ≤ window_bp).
    min_cpgs : int
        Minimum CpG count in a *merged* DMR (default 5).
    min_sites_significant : int
        Minimum significant CpG sites in a window for it to be a candidate
        (default 3).
    alpha : float
        Significance threshold for qvalue / pvalue (default 0.05).
    min_abs_meth_diff : float
        Minimum |meth_diff| for a site to count as significant (default 0.10).

    Returns
    -------
    pl.DataFrame
        Columns: chrom, start, end, n_cpgs, n_significant,
                 mean_meth_diff, mean_pvalue, dmr_type ("hyper" | "hypo").
    """
    required = {"chrom", "pos", "meth_diff", "pvalue"}
    missing  = required - set(dmc_results.columns)
    if missing:
        raise ValueError(f"DMC results missing required columns: {missing}")
    if step_bp > window_bp:
        raise ValueError(
            f"step_bp ({step_bp}) must be ≤ window_bp ({window_bp})"
        )

    # Prefer corrected q-values when available
    p_col = "qvalue" if "qvalue" in dmc_results.columns else "pvalue"
    logger.info(
        "call_dmr_sliding_window: window=%d bp, step=%d bp, "
        "min_cpgs=%d, min_sig=%d, alpha=%.3f, min_|Δβ|=%.2f, p_col=%s",
        window_bp, step_bp, min_cpgs, min_sites_significant,
        alpha, min_abs_meth_diff, p_col,
    )

    all_records: list[dict] = []

    for chrom in sorted(dmc_results["chrom"].unique().to_list()):
        chrom_df = (
            dmc_results
            .filter(pl.col("chrom") == chrom)
            .sort("pos")
        )
        if len(chrom_df) == 0:
            continue

        positions  = chrom_df["pos"].to_numpy()
        meth_diffs = chrom_df["meth_diff"].to_numpy(allow_copy=True).astype(np.float32)
        pvals      = chrom_df[p_col].to_numpy(allow_copy=True).astype(np.float64)

        is_sig = (
            (~np.isnan(pvals))
            & (pvals < alpha)
            & (~np.isnan(meth_diffs))
            & (np.abs(meth_diffs) >= min_abs_meth_diff)
        )

        pos_min = int(positions[0])
        pos_max = int(positions[-1])

        cand_starts: list[int] = []
        cand_ends:   list[int] = []

        win_start = pos_min
        while win_start <= pos_max:
            win_end = win_start + window_bp
            mask   = (positions >= win_start) & (positions < win_end)
            n_cpgs = int(mask.sum())
            n_sig  = int(is_sig[mask].sum())

            if n_cpgs >= min_cpgs and n_sig >= min_sites_significant:
                cand_starts.append(win_start)
                cand_ends.append(win_end)

            win_start += step_bp

        if not cand_starts:
            continue

        merged_spans = _merge_intervals(cand_starts, cand_ends)
        chrom_dmrs   = 0

        for start, end in merged_spans:
            rec = _recompute_dmr_stats(
                chrom, start, end,
                positions, meth_diffs, pvals, is_sig,
                min_cpgs, min_sites_significant,
            )
            if rec is not None:
                all_records.append(rec)
                chrom_dmrs += 1

        logger.info(
            "  %s: %d candidate span(s) → %d DMR(s)",
            chrom, len(merged_spans), chrom_dmrs,
        )

    if not all_records:
        logger.warning("No DMRs found with current filters")
        return pl.DataFrame(schema=_DMR_EMPTY_SCHEMA)

    return (
        pl.DataFrame(all_records)
        .with_columns([
            pl.col("start").cast(pl.Int32),
            pl.col("end").cast(pl.Int32),
            pl.col("n_cpgs").cast(pl.Int32),
            pl.col("n_significant").cast(pl.Int32),
            pl.col("mean_meth_diff").cast(pl.Float32),
        ])
        .sort(["chrom", "start"])
    )


# ---------------------------------------------------------------------------
# Public API — BSmooth-style smoothing
# ---------------------------------------------------------------------------

def smooth_methylation_bsmooth(
    methylstore_path: str,
    samples: list[str],
    bandwidth: int = 1000,
) -> pl.DataFrame:
    """Smooth per-sample beta values using local polynomial (LOESS) regression.

    Implements the core BSmooth pre-processing step: within each chromosome
    and each sample, raw beta values (N_meth / coverage) are smoothed along
    the genomic axis using LOESS.  The smoothed betas are intended to replace
    raw estimates before feeding into a DMR caller.

    Parameters
    ----------
    methylstore_path : str
        Path to the filtered partitioned Parquet methylstore.
    samples : list[str]
        Sample identifiers to smooth.
    bandwidth : int
        Approximate smoothing bandwidth in base pairs (default 1000 bp).
        Converted to a LOESS ``frac`` parameter:
            frac = bandwidth / (pos_max − pos_min + 1)
        and clamped to [0.01, 1.0].

    Returns
    -------
    pl.DataFrame
        Columns: chrom, pos, sample, beta_raw (Float32), beta_smooth (Float32).
        Sites with zero coverage have NaN in both beta columns.
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for BSmooth smoothing. "
            "Install with: pip install statsmodels"
        ) from exc

    store   = Path(methylstore_path)
    records: list[pl.DataFrame] = []

    for sample in samples:
        sample_dir = store / f"sample={sample}"
        if not sample_dir.exists():
            logger.warning("Sample '%s' not found in %s; skipping", sample, store)
            continue

        for chrom_dir in sorted(sample_dir.glob("chrom=*")):
            chrom = chrom_dir.name.removeprefix("chrom=")
            parts = list(chrom_dir.glob("part-*.parquet"))
            if not parts:
                continue

            df = pl.concat([
                pl.read_parquet(str(p), columns=["pos", "N_meth", "coverage"])
                for p in parts
            ]).sort("pos")

            pos  = df["pos"].to_numpy().astype(np.float64)
            meth = df["N_meth"].to_numpy().astype(np.float64)
            cov  = df["coverage"].to_numpy().astype(np.float64)

            with np.errstate(invalid="ignore", divide="ignore"):
                beta_raw = np.where(cov > 0, meth / cov, np.nan).astype(np.float32)

            beta_smooth = beta_raw.copy()
            valid       = ~np.isnan(beta_raw)

            if valid.sum() >= 4:
                pos_range = float(pos[valid][-1] - pos[valid][0])
                frac      = float(np.clip(bandwidth / max(pos_range, 1.0), 0.01, 1.0))
                smoothed  = lowess(
                    beta_raw[valid].astype(np.float64),
                    pos[valid],
                    frac=frac,
                    it=3,
                    return_sorted=False,
                )
                beta_smooth[valid] = np.clip(smoothed, 0.0, 1.0).astype(np.float32)
            else:
                logger.debug(
                    "  %s / %s: only %d valid sites; skipping smoothing",
                    sample, chrom, int(valid.sum()),
                )

            records.append(pl.DataFrame({
                "chrom":       pl.Series([chrom]  * len(df), dtype=pl.Utf8),
                "pos":         df["pos"],
                "sample":      pl.Series([sample] * len(df), dtype=pl.Utf8),
                "beta_raw":    pl.Series(beta_raw),
                "beta_smooth": pl.Series(beta_smooth),
            }))

    if not records:
        return pl.DataFrame(schema=_SMOOTH_EMPTY_SCHEMA)

    return pl.concat(records).sort(["chrom", "pos", "sample"])
