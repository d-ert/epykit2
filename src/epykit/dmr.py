"""Differentially Methylated Region (DMR) calling.

Two algorithms:

``call_dmr_tile_based(methylstore_path, samples_treatment, samples_control,
...)``
    methylKit-parity tile aggregation: sums (N_meth, coverage) across CpGs
    per sample within each fixed-size tile, then runs a full DMC test on
    the tile-level counts. Recommended path.

``call_dmr_sliding_window(dmc_results, ...)``
    Operates on a precomputed per-CpG DMC table; combines p-values per
    window via signed Stouffer's Z. Faster but lower-power; sign comes
    from each CpG's meth_diff so mixed-direction windows are downweighted.
    Direction (hyper / hypo / mixed) is set by the sign of the mean
    meth_diff rather than a raw site tally.

``smooth_methylation_gaussian`` is a coverage-weighted Gaussian-kernel
approximation to BSmooth — see its own docstring.
"""

from __future__ import annotations

import gc
import logging
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

_DMR_EMPTY_SCHEMA = {
    "chrom":            pl.Utf8,
    "start":            pl.Int32,
    "end":              pl.Int32,
    "n_cpgs":           pl.Int32,
    "n_significant":    pl.Int32,
    "mean_meth_diff":   pl.Float32,
    "combined_pvalue":  pl.Float64,
    "combined_qvalue":  pl.Float64,
    "dmr_type":         pl.Utf8,
}

_DMR_TILE_SCHEMA = {
    "chrom":            pl.Utf8,
    "start":            pl.Int32,
    "end":              pl.Int32,
    "n_cpgs":           pl.Int32,
    "n_case":           pl.Int32,
    "n_control":        pl.Int32,
    "mean_beta_case":   pl.Float32,
    "mean_beta_control": pl.Float32,
    "meth_diff":        pl.Float32,
    "log2_odds_ratio":  pl.Float64,
    "pvalue":           pl.Float64,
    "qvalue":           pl.Float64,
    "dmr_type":         pl.Utf8,
}

_SMOOTH_EMPTY_SCHEMA = {
    "chrom":        pl.Utf8,
    "pos":          pl.Int32,
    "sample":       pl.Utf8,
    "beta_raw":     pl.Float32,
    "beta_smooth":  pl.Float32,
}

# FIX-6: cap merged DMR size to prevent biologically implausible mega-DMRs.
# Mammalian DMRs are typically 200 bp – 5 kb; 10 kb is a generous ceiling.
_MAX_DMR_BP: int = 10_000

# BIO-10: a window's direction is called "mixed" when the fraction of
# valid sites agreeing with the sign of the mean is below this threshold.
_MIXED_DIRECTION_THRESHOLD: float = 0.6


# Internal helpers — p-value combination

def _stouffer_combine_signed(
    pvals: np.ndarray,
    meth_diffs: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Combine per-CpG two-sided p-values via signed Stouffer's Z.

    Each CpG contributes a signed Z-score:
        z_i = sign(meth_diff_i) · Φ⁻¹(1 - p_i / 2)
    so that hyper-methylated CpGs contribute positive Z and hypo-methylated
    CpGs contribute negative Z. The combined statistic is
        Z = Σ w_i · z_i  /  √(Σ w_i²)
    which is two-sided-tested. When all CpGs in a window agree in direction
    the |Z| grows as √k and the combined p-value gets correspondingly
    small; when directions are mixed, contributions cancel and the
    combined p-value stays large.

    BIO-9: this replaces the previous Brown's method implementation, which
    required a correlation matrix of the per-CpG test statistics. The old
    code estimated it from genomic distances between CpGs — a proxy for the
    correlation of methylation STATES, not of the test statistics — which
    systematically over-inflated the variance correction f and weakened
    combined p-values regardless of how strong the per-CpG signal was.
    Stouffer's Z is robust to mild positive correlation between tests
    (combined Z is conservative but not over-smoothed) and does not
    require correlation estimation.

    Parameters
    ----------
    pvals : np.ndarray
        Two-sided p-values for each CpG in the window.
    meth_diffs : np.ndarray
        Signed per-CpG effect sizes (mean_beta_case − mean_beta_ctrl).
        Used only for direction; magnitude is ignored.
    weights : np.ndarray, optional
        Per-CpG weights (e.g. coverage). Defaults to equal weights.

    Returns
    -------
    float
        Two-sided combined p-value.
    """
    pvals      = np.asarray(pvals,      dtype=np.float64)
    meth_diffs = np.asarray(meth_diffs, dtype=np.float64)

    valid = (
        ~np.isnan(pvals) & (pvals > 0.0) & (pvals <= 1.0)
        & ~np.isnan(meth_diffs)
    )
    if not np.any(valid):
        return float("nan")

    p_valid    = np.clip(pvals[valid], np.finfo(float).tiny, 1.0 - 1e-15)
    diff_valid = meth_diffs[valid]

    # Magnitude Z from two-sided p-value: |z| = Φ⁻¹(1 - p/2)
    z_mag = sp_stats.norm.isf(p_valid / 2.0)
    # Signed contribution: hyper (+) vs hypo (-). meth_diff == 0 → no
    # contribution (sign = 0), which is correct: zero-effect CpGs neither
    # add nor subtract evidence.
    z_signed = np.sign(diff_valid) * z_mag

    if weights is None:
        w = np.ones_like(z_signed)
    else:
        w = np.asarray(weights, dtype=np.float64)[valid]
        w = np.where(np.isfinite(w) & (w >= 0), w, 0.0)

    w_sq_sum = float(np.sum(w * w))
    if w_sq_sum <= 0.0:
        return float("nan")

    z_combined = float(np.sum(w * z_signed) / np.sqrt(w_sq_sum))
    # Two-sided normal tail
    return float(2.0 * sp_stats.norm.sf(abs(z_combined)))


def _merge_intervals(starts: list[int], ends: list[int]) -> list[tuple[int, int]]:
    """Merge overlapping (start, end) integer intervals."""
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


def _classify_direction(
    mean_diff: float,
    n_hyper: int,
    n_hypo: int,
) -> str:
    """Classify DMR direction from mean effect + per-site sign tally.

    BIO-10: previously the direction was the larger of n_hyper / n_hypo.
    A window with 6 hyper sites at +0.12 and 4 hypo sites at -0.40 was
    called "hyper" even though the mean effect was clearly negative.

    The new rule:
      - Mean direction governs (sign of mean_meth_diff). This matches
        what the tile-based path does when it pools reads.
      - If the per-site sign tally is too close (< 60 % majority), the
        window is labelled "mixed" so downstream filters can drop it.
    """
    total_signed = n_hyper + n_hypo
    if total_signed == 0:
        return "mixed"
    consensus = max(n_hyper, n_hypo) / total_signed
    if consensus < _MIXED_DIRECTION_THRESHOLD:
        return "mixed"
    if np.isnan(mean_diff):
        # Fall back to majority tally if mean is undefined for some reason.
        return "hyper" if n_hyper > n_hypo else "hypo"
    return "hyper" if mean_diff > 0 else "hypo"


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
    # Optional pre-computed prefix arrays for O(log n) slice
    cum_sig: np.ndarray | None = None,
) -> dict | None:
    """Recompute accurate per-site statistics over a merged interval.

    When ``cum_sig`` is supplied the significance count is computed in O(1)
    via prefix-sum lookup; otherwise falls back to a boolean mask scan.
    """
    # FIX-6: reject biologically implausible mega-DMRs that arise when
    # overlapping candidate windows collapse across many megabases.
    if (end - start) > _MAX_DMR_BP:
        return None

    # Use searchsorted for O(log n) slice instead of boolean mask
    lo = int(np.searchsorted(positions, start,  side="left"))
    hi = int(np.searchsorted(positions, end,    side="left"))

    n_cpgs = hi - lo
    if n_cpgs < min_cpgs:
        return None

    if cum_sig is not None:
        n_sig = int(cum_sig[hi] - cum_sig[lo])
    else:
        n_sig = int(is_sig[lo:hi].sum())

    if n_sig < min_sites_significant:
        return None

    window_diffs = meth_diffs[lo:hi]
    window_pvals = pvals[lo:hi]

    valid_diffs = window_diffs[~np.isnan(window_diffs)]
    if len(valid_diffs) == 0:
        return None

    n_hyper = int((valid_diffs > 0).sum())
    n_hypo  = int((valid_diffs < 0).sum())

    # BIO-9: signed Stouffer's Z. Sign comes from per-CpG meth_diff so the
    # combined statistic naturally cancels mixed-direction windows.
    combined_p = _stouffer_combine_signed(window_pvals, window_diffs)
    if np.isnan(combined_p):
        return None

    mean_diff = float(np.nanmean(window_diffs))
    dmr_type  = _classify_direction(mean_diff, n_hyper, n_hypo)

    return {
        "chrom":           chrom,
        "start":           start,
        "end":             end,
        "n_cpgs":          n_cpgs,
        "n_significant":   n_sig,
        "mean_meth_diff":  float(np.float32(mean_diff)),
        "combined_pvalue": float(combined_p),
        "dmr_type":        dmr_type,
    }


# Public API — sliding-window DMR calling (works from a DMC table)

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

    This method takes a precomputed DMC table and combines per-CpG p-values
    region-by-region with signed Stouffer's Z. It is fast and reuses an
    existing DMC call, but has lower power than the tile-based path
    (`call_dmr_tile_based`) because it cannot pool reads — windows whose
    individual CpGs aren't significant won't gather enough sig sites to
    pass the `min_sites_significant` gate.

    Parameters
    ----------
    dmc_results : pl.DataFrame
        Output from ``process_chromosomes_dmc`` /
        ``apply_multiple_testing_correction``.
        Required columns: chrom, pos, meth_diff, pvalue.
        Optional: qvalue (used in preference to pvalue when present).
    window_bp, step_bp : int
        Window width and step in base pairs.
    min_cpgs : int
        Minimum CpG count in a *merged* DMR.
    min_sites_significant : int
        Minimum significant CpG sites in a window for it to be a candidate.
    alpha : float
        Significance threshold for qvalue / pvalue.
    min_abs_meth_diff : float
        Minimum |meth_diff| for a site to count as significant.

    Returns
    -------
    pl.DataFrame
        Columns: chrom, start, end, n_cpgs, n_significant,
                 mean_meth_diff, combined_pvalue, combined_qvalue,
                 dmr_type ("hyper" | "hypo" | "mixed").
        ``combined_qvalue`` is BH-corrected genome-wide across DMR
        candidates that passed the per-window gate.
    """
    required = {"chrom", "pos", "meth_diff", "pvalue"}
    missing  = required - set(dmc_results.columns)
    if missing:
        raise ValueError(f"DMC results missing required columns: {missing}")
    if step_bp > window_bp:
        raise ValueError(
            f"step_bp ({step_bp}) must be ≤ window_bp ({window_bp})"
        )

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

        # ---------------------------------------------------------------
        # Build prefix-sum arrays for O(1) range count queries
        # ---------------------------------------------------------------
        cum_sig = np.empty(len(positions) + 1, dtype=np.int32)
        cum_sig[0] = 0
        np.cumsum(is_sig.astype(np.int32), out=cum_sig[1:])

        pos_min = int(positions[0])
        pos_max = int(positions[-1])

        win_starts_arr = np.arange(pos_min, pos_max + 1, step_bp, dtype=np.int64)
        win_ends_arr   = win_starts_arr + window_bp

        lefts  = np.searchsorted(positions, win_starts_arr, side="left")
        rights = np.searchsorted(positions, win_ends_arr,   side="left")

        n_cpgs_arr = (rights - lefts).astype(np.int32)
        n_sig_arr  = (cum_sig[rights] - cum_sig[lefts]).astype(np.int32)

        cand_mask = (n_cpgs_arr >= min_cpgs) & (n_sig_arr >= min_sites_significant)

        if not np.any(cand_mask):
            logger.info("  %s: no candidate windows", chrom)
            continue

        cand_starts = win_starts_arr[cand_mask].tolist()
        cand_ends   = win_ends_arr[cand_mask].tolist()

        merged_spans = _merge_intervals(cand_starts, cand_ends)
        chrom_dmrs   = 0

        for start, end in merged_spans:
            rec = _recompute_dmr_stats(
                chrom, start, end,
                positions, meth_diffs, pvals, is_sig,
                min_cpgs, min_sites_significant,
                cum_sig=cum_sig,
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

    dmr_df = (
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

    # BIO-11: BH-correct DMR-level combined p-values so downstream filters
    # are operating on q-values. Without this the sliding-window output
    # was effectively un-corrected at the region level.
    from .dmc import apply_multiple_testing_correction

    dmr_df = apply_multiple_testing_correction(
        dmr_df,
        method="fdr_bh",
        pvalue_col="combined_pvalue",
        qvalue_col="combined_qvalue",
    )
    return dmr_df


# Public API — tile-based DMR calling (methylKit-style)

def _aggregate_sample_to_tiles(
    src_part_file: Path,
    chrom: str,
    tile_size_bp: int,
) -> pl.DataFrame | None:
    """Aggregate one sample/chromosome's per-CpG counts to per-tile sums.

    Returns a DataFrame with columns: chrom, pos (= tile start), strand,
    N_meth, N_unmeth, coverage, n_cpgs.  Or None when the source is missing.
    """
    if not src_part_file.exists():
        return None

    df = pl.read_parquet(str(src_part_file))
    if len(df) == 0:
        return None

    # Tile assignment: pos // tile * tile gives left-inclusive boundary.
    tile_col = (pl.col("pos") // tile_size_bp) * tile_size_bp
    tiled = (
        df.with_columns(tile_col.cast(pl.Int32).alias("tile_start"))
        .group_by("tile_start")
        .agg([
            pl.sum("N_meth").alias("N_meth"),
            pl.sum("coverage").alias("coverage"),
            pl.len().alias("n_cpgs"),
            # Preserve a strand value: first non-"*" if available, else "*"
            (
                pl.when(pl.col("strand") != "*").then(pl.col("strand")).otherwise(None)
                .drop_nulls().first()
            ).alias("strand_real")
            if "strand" in df.columns
            else pl.lit("*").alias("strand_real"),
        ])
        .with_columns([
            pl.lit(chrom).alias("chrom"),
            pl.col("tile_start").alias("pos"),
            pl.col("strand_real").fill_null("*").alias("strand"),
            (pl.col("coverage") - pl.col("N_meth")).alias("N_unmeth"),
        ])
        .drop("tile_start", "strand_real")
        .sort("pos")
    )
    return tiled


def call_dmr_tile_based(
    methylstore_path: str,
    samples_treatment: list[str] | None = None,
    samples_control: list[str] | None = None,
    tile_size_bp: int = 1000,
    test: str = "logit_t",
    chromosomes: list[str] | None = None,
    min_cpgs_per_tile: int = 5,
    alpha: float = 0.05,
    min_abs_meth_diff: float = 0.1,
    unite: bool = True,
    min_samples_treatment: int | None = None,
    min_samples_control: int = 0,
    dispersion: str = "site",
    reference: str = "chi2",
    design_full: np.ndarray | None = None,
    design_reduced: np.ndarray | None = None,
    coef_idx: int | None = None,
    *,
    samples_case: list[str] | None = None,         # deprecated alias
    min_samples_case: int | None = None,           # deprecated alias
) -> pl.DataFrame:
    """Call DMRs by aggregating read counts within fixed-size tiles.

    BIO-5: This is the methylKit-parity DMR path. methylKit's tileMethylCounts
    sums N_meth and coverage across all CpGs in each tile per sample, then
    runs a single tile-level test (e.g. logistic regression with
    overdispersion). The sliding-window method in this module tests each
    CpG individually and combines p-values, which has dramatically lower
    power at typical WGBS coverage: a tile with 20 CpGs at +15 % effect
    might have zero individually-significant CpGs but still trivially
    pass when its 600 pooled reads are tested.

    Implementation
    --------------
    1. For each sample and chromosome, aggregate (N_meth, coverage) per tile
       and write a "tiled methylstore" to a temp directory. Tiles with
       fewer than ``min_cpgs_per_tile`` CpGs (in that sample) are dropped
       before writing.
    2. Run ``process_chromosomes_dmc`` on the tiled store with the requested
       test. Tiles are treated as "sites" with pos = tile_start.
    3. BH-correct the tile-level p-values.
    4. Filter on qvalue and |meth_diff|.
    5. Reshape into the DMR output schema.

    Parameters
    ----------
    methylstore_path : str
        Path to the filtered partitioned Parquet methylstore.
    samples_case, samples_control : list[str]
        Sample IDs.
    tile_size_bp : int
        Tile width in bp (default 1000, matching methylKit default).
        Adjacent tiles do not overlap.
    test : str
        Statistical test for tile-level counts. Defaults to ``"logit_t"``,
        which is the closest analogue to methylKit's logistic regression
        at n ≥ 3 replicates.
    chromosomes : list[str], optional
        Chromosomes to process. Auto-detected when None.
    min_cpgs_per_tile : int
        Skip tiles with fewer than this many CpGs (per sample) during the
        per-sample aggregation step. methylKit's default is 1; we use 5 to
        reduce noise at sparse coverage.
    alpha : float
        q-value threshold for significance.
    min_abs_meth_diff : float
        Minimum |meth_diff| for a tile to be called significant.
    unite : bool
        If True (default), only test tiles covered in every sample.
    min_samples_case, min_samples_control : int
        Per-tile minimum number of samples required to be present in each
        group (only relevant when unite=False).

    Returns
    -------
    pl.DataFrame
        Columns: chrom, start, end, n_cpgs, n_case, n_control,
                 mean_beta_case, mean_beta_control, meth_diff,
                 log2_odds_ratio, pvalue, qvalue, dmr_type.
    """
    from .dmc import (
        process_chromosomes_dmc,
        apply_multiple_testing_correction,
        _resolve_treatment_aliases,
    )

    samples_treatment, min_samples_treatment = _resolve_treatment_aliases(
        samples_treatment, samples_case, min_samples_treatment, min_samples_case
    )
    if samples_control is None:
        raise TypeError("Missing required argument: samples_control")
    samples_case = samples_treatment
    min_samples_case = min_samples_treatment

    store       = Path(methylstore_path)
    all_samples = samples_case + samples_control

    if chromosomes is None:
        chromosomes = sorted({
            d.name.removeprefix("chrom=")
            for s in store.glob("sample=*")
            for d in s.glob("chrom=*")
        })

    if not chromosomes:
        return pl.DataFrame(schema=_DMR_TILE_SCHEMA)

    logger.info(
        "call_dmr_tile_based: tile=%d bp, test=%s, n_case=%d, n_control=%d, "
        "min_cpgs/tile=%d, alpha=%.3f, min_|Δβ|=%.2f, unite=%s",
        tile_size_bp, test, len(samples_case), len(samples_control),
        min_cpgs_per_tile, alpha, min_abs_meth_diff, unite,
    )

    with tempfile.TemporaryDirectory(prefix="epykit_tile_") as tmpdir:
        tile_store = Path(tmpdir) / "tiled_store"

        # ----- Phase 1: aggregate per-sample, per-chromosome counts -----
        # Track per-tile CpG counts (max across samples) for output column.
        # Stored as {(chrom, tile_start): n_cpgs}.
        tile_n_cpgs: dict[tuple[str, int], int] = {}

        for sample in all_samples:
            for chrom in chromosomes:
                src = store / f"sample={sample}" / f"chrom={chrom}" / "part-0.parquet"
                tiled = _aggregate_sample_to_tiles(src, chrom, tile_size_bp)
                if tiled is None or len(tiled) == 0:
                    continue
                tiled = tiled.filter(pl.col("n_cpgs") >= min_cpgs_per_tile)
                if len(tiled) == 0:
                    continue

                # Record per-tile CpG count (use max across samples so the
                # output reflects the most CpG-dense observation of the
                # tile, comparable to methylKit's reporting).
                for tile_start, n_cpgs_val in zip(
                    tiled["pos"].to_list(), tiled["n_cpgs"].to_list()
                ):
                    key = (chrom, int(tile_start))
                    if n_cpgs_val > tile_n_cpgs.get(key, 0):
                        tile_n_cpgs[key] = int(n_cpgs_val)

                out_dir = tile_store / f"sample={sample}" / f"chrom={chrom}"
                out_dir.mkdir(parents=True, exist_ok=True)
                (
                    tiled
                    .select(["chrom", "pos", "strand", "N_meth", "N_unmeth", "coverage"])
                    .write_parquet(str(out_dir / "part-0.parquet"))
                )

        # ----- Phase 2: run DMC on the tiled store -----
        if not list(tile_store.glob("sample=*/chrom=*/part-0.parquet")):
            logger.warning("Tile aggregation produced no rows; returning empty DMR set")
            return pl.DataFrame(schema=_DMR_TILE_SCHEMA)

        tile_dmc = process_chromosomes_dmc(
            methylstore_path=str(tile_store),
            samples_treatment=samples_case,
            samples_control=samples_control,
            test=test,
            chromosomes=chromosomes,
            unite=unite,
            min_samples_treatment=min_samples_case,
            min_samples_control=min_samples_control,
            dispersion=dispersion,
            reference=reference,
            design_full=design_full,
            design_reduced=design_reduced,
            coef_idx=coef_idx,
        )

    if len(tile_dmc) == 0:
        return pl.DataFrame(schema=_DMR_TILE_SCHEMA)

    # ----- Phase 3: BH at tile level -----
    tile_dmc = apply_multiple_testing_correction(tile_dmc, method="fdr_bh")

    # ----- Phase 4: filter and reshape -----
    # Attach n_cpgs from the per-sample aggregation.
    n_cpgs_rows = [
        {"chrom": c, "pos": p, "n_cpgs": n}
        for (c, p), n in tile_n_cpgs.items()
    ]
    n_cpgs_df = pl.DataFrame(
        n_cpgs_rows,
        schema={"chrom": pl.Utf8, "pos": pl.Int32, "n_cpgs": pl.Int32},
    )

    dmr_df = (
        tile_dmc
        .join(n_cpgs_df, on=["chrom", "pos"], how="left")
        .with_columns(pl.col("n_cpgs").fill_null(0))
        .filter(
            (pl.col("qvalue") < alpha)
            & (pl.col("meth_diff").abs() >= min_abs_meth_diff)
            & (~pl.col("pvalue").is_nan())
        )
        .with_columns([
            pl.col("pos").alias("start"),
            (pl.col("pos") + tile_size_bp).cast(pl.Int32).alias("end"),
            pl.when(pl.col("meth_diff") > 0)
              .then(pl.lit("hyper"))
              .otherwise(pl.lit("hypo"))
              .alias("dmr_type"),
        ])
    )

    out_cols = [
        "chrom", "start", "end", "n_cpgs",
        "n_case", "n_control",
        "mean_beta_case", "mean_beta_control",
        "meth_diff", "log2_odds_ratio",
        "pvalue", "qvalue",
        "dmr_type",
    ]
    # GLM path adds adjusted log-odds effect size for the treatment coefficient.
    for extra in ("coef_treatment", "coef_se"):
        if extra in dmr_df.columns:
            out_cols.append(extra)
    dmr_df = dmr_df.select(out_cols).sort(["chrom", "start"])

    logger.info("Tile-based DMR: %s tiles → %s significant DMRs",
                f"{len(tile_dmc):,}", f"{len(dmr_df):,}")

    gc.collect()
    return dmr_df


# Permutation-based empirical FDR

def empirical_fdr_for_dmr(
    methylstore_path: str,
    samples_treatment: list[str],
    samples_control: list[str],
    observed_dmr: pl.DataFrame,
    *,
    n_perm: int = 100,
    seed: int = 42,
    n_jobs: int = 1,
    **dmr_kwargs,
) -> pl.DataFrame:
    """Empirical (permutation) FDR for tile-based DMRs.

    Re-runs ``call_dmr_tile_based`` ``n_perm`` times with treatment / control
    labels shuffled. For each observed DMR, the empirical p-value is
    estimated from the fraction of null DMRs (across all permutations) with
    raw p-value <= observed. The result is BH-adjusted to
    ``empirical_qvalue``.

    Parameters
    ----------
    methylstore_path, samples_treatment, samples_control
        Same arguments passed to :func:`call_dmr_tile_based`.
    observed_dmr
        The DMR DataFrame returned by the observed (unpermuted) run.
        Empirical columns are appended to a copy of this frame.
    n_perm
        Number of permutations.
    seed
        Seed for the per-permutation label shuffler.
    n_jobs
        joblib parallel worker count. -1 uses all cores. Falls back to
        serial execution when joblib is not installed.
    **dmr_kwargs
        Forwarded to ``call_dmr_tile_based`` for each permutation; should
        match the observed run's settings (tile_size_bp, test, alpha,
        min_abs_meth_diff, dispersion, reference, etc.).

    Returns
    -------
    pl.DataFrame
        ``observed_dmr`` with added columns ``empirical_pvalue`` and
        ``empirical_qvalue``. The full null pool (per-DMR raw pvalues from
        every permutation) is cached on ``observed_dmr.attrs`` only if the
        caller upstream wires it — this function just returns the
        annotated table.
    """
    if len(observed_dmr) == 0:
        return observed_dmr.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("empirical_pvalue"),
            pl.lit(None, dtype=pl.Float64).alias("empirical_qvalue"),
        ])

    n_treat = len(samples_treatment)
    pool = list(samples_treatment) + list(samples_control)
    rng = np.random.default_rng(seed)

    def _run_one_perm(perm_idx: int) -> np.ndarray:
        # Local RNG so parallel workers stay deterministic.
        local_rng = np.random.default_rng(seed + perm_idx + 1)
        shuffled = pool.copy()
        local_rng.shuffle(shuffled)
        perm_treat = shuffled[:n_treat]
        perm_ctrl = shuffled[n_treat:]
        # Force test='lr' or whatever observed used; do not run annotation.
        kwargs = dict(dmr_kwargs)
        kwargs.pop("samples_case", None)
        kwargs.pop("min_samples_case", None)
        try:
            null_df = call_dmr_tile_based(
                methylstore_path=methylstore_path,
                samples_treatment=perm_treat,
                samples_control=perm_ctrl,
                **kwargs,
            )
        except Exception as exc:
            logger.warning("permutation %d failed: %s", perm_idx, exc)
            return np.array([], dtype=np.float64)
        if "pvalue" not in null_df.columns or len(null_df) == 0:
            return np.array([], dtype=np.float64)
        return null_df.get_column("pvalue").drop_nulls().to_numpy()

    null_pvals_list: list[np.ndarray]
    if n_jobs == 1:
        null_pvals_list = [_run_one_perm(i) for i in range(n_perm)]
    else:
        try:
            from joblib import Parallel, delayed
            null_pvals_list = Parallel(n_jobs=n_jobs)(
                delayed(_run_one_perm)(i) for i in range(n_perm)
            )
        except ImportError:
            logger.warning("joblib not installed; falling back to serial execution.")
            null_pvals_list = [_run_one_perm(i) for i in range(n_perm)]

    if all(len(arr) == 0 for arr in null_pvals_list):
        logger.warning(
            "All %d permutations produced zero null DMRs. Empirical "
            "p-values default to 1 / (1 + n_perm).",
            n_perm,
        )
    null_pool = (
        np.concatenate(null_pvals_list) if any(len(a) for a in null_pvals_list)
        else np.array([1.0])
    )
    null_sorted = np.sort(null_pool)
    obs_p = observed_dmr.get_column("pvalue").to_numpy()
    # For each observed p, count null DMRs with raw pvalue <= obs_p.
    # Plus-one in num/den is the standard "add 1" correction so empirical
    # p never hits 0 with finite permutations.
    counts = np.searchsorted(null_sorted, obs_p, side="right")
    total_null = max(len(null_sorted), 1)
    emp_p = (counts + 1.0) / (total_null + 1.0)
    emp_p = np.clip(emp_p, 0.0, 1.0)

    # BH-adjust to empirical q-value
    from statsmodels.stats.multitest import multipletests
    finite = np.isfinite(emp_p)
    emp_q = np.full_like(emp_p, np.nan, dtype=np.float64)
    if finite.any():
        _, q_finite, _, _ = multipletests(emp_p[finite], method="fdr_bh")
        emp_q[finite] = q_finite

    return observed_dmr.with_columns([
        pl.Series("empirical_pvalue", emp_p),
        pl.Series("empirical_qvalue", emp_q),
    ])


# Public API — fast Gaussian smoothing (replaces statsmodels LOESS)

def smooth_methylation_gaussian(
    methylstore_path: str,
    samples: list[str],
    bandwidth: int = 1000,
    grid_resolution_bp: int | None = None,
    output_path: str | None = None,
) -> pl.DataFrame | None:
    """Smooth per-sample beta values with a fast Gaussian kernel.

    .. note::
       This is a Gaussian-kernel approximation, not the local-LOESS smoother
       used by Hansen et al.'s BSmooth. The function was previously named
       ``smooth_methylation_bsmooth``; that alias is kept for one
       deprecation cycle. A true LOESS-based BSmooth implementation is on
       the roadmap.

    Within each chromosome and sample, raw beta values are smoothed along
    the genomic axis. The implementation projects raw betas onto a regular
    grid, applies a coverage-weighted Gaussian filter
    (``scipy.ndimage.gaussian_filter1d``), then interpolates back to the
    original CpG positions. This is O(G) where G is the grid size,
    versus O(n²) for LOESS — typically 100-500× faster on WGBS-scale data.
    """
    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError as exc:
        raise ImportError(
            "scipy is required for BSmooth smoothing. "
            "Install with: pip install scipy"
        ) from exc

    store   = Path(methylstore_path)
    records: list[pl.DataFrame] = []

    # Determine grid resolution once (same for all samples/chroms)
    _grid_res = max(1, bandwidth // 20) if grid_resolution_bp is None else grid_resolution_bp
    _out_root = Path(output_path) if output_path else None
    if _out_root:
        _out_root.mkdir(parents=True, exist_ok=True)

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
            n_valid     = int(valid.sum())

            if n_valid >= 4:
                pos_valid   = pos[valid]
                beta_valid  = beta_raw[valid].astype(np.float64)

                # Build a regular grid spanning the valid positions.
                grid_start  = int(pos_valid[0])
                grid_end    = int(pos_valid[-1]) + _grid_res
                grid_pos    = np.arange(grid_start, grid_end, _grid_res,
                                        dtype=np.float64)

                # Coverage-weighted interpolation onto the regular grid
                cov_valid = cov[valid].astype(np.float64)
                grid_beta = np.interp(grid_pos, pos_valid, beta_valid)
                grid_weights = np.interp(grid_pos, pos_valid, cov_valid)
                grid_weights = np.maximum(grid_weights, 0.1)  # avoid exact zeros

                # Apply weighted Gaussian: smooth numerator and denominator separately
                sigma_grid = max(bandwidth / _grid_res, 0.5)
                grid_num = gaussian_filter1d(grid_beta * grid_weights, sigma=sigma_grid, mode="nearest")
                grid_den = gaussian_filter1d(grid_weights, sigma=sigma_grid, mode="nearest")
                smoothed_grid = grid_num / np.maximum(grid_den, 1e-9)
                np.clip(smoothed_grid, 0.0, 1.0, out=smoothed_grid)

                # Interpolate smoothed values back to original CpG positions
                smoothed_at_cpgs = np.interp(pos_valid, grid_pos, smoothed_grid)
                beta_smooth[valid] = smoothed_at_cpgs.astype(np.float32)
            else:
                logger.debug(
                    "  %s / %s: only %d valid sites; skipping smoothing",
                    sample, chrom, n_valid,
                )

            chunk = pl.DataFrame({
                "chrom":       pl.Series([chrom]  * len(df), dtype=pl.Utf8),
                "pos":         df["pos"],
                "sample":      pl.Series([sample] * len(df), dtype=pl.Utf8),
                "beta_raw":    pl.Series(beta_raw),
                "beta_smooth": pl.Series(beta_smooth),
            })

            if _out_root is not None:
                part_dir = _out_root / f"sample={sample}" / f"chrom={chrom}"
                part_dir.mkdir(parents=True, exist_ok=True)
                chunk.write_parquet(
                    str(part_dir / "part-0.parquet"), compression="zstd"
                )
            else:
                records.append(chunk)

    if _out_root is not None:
        return None

    if not records:
        return pl.DataFrame(schema=_SMOOTH_EMPTY_SCHEMA)

    return pl.concat(records).sort(["chrom", "pos", "sample"])


# Deprecated alias — remove in v0.3

def smooth_methylation_bsmooth(*args, **kwargs):
    """Deprecated alias for :func:`smooth_methylation_gaussian`.

    The implementation is Gaussian convolution on a regular grid, not the
    local LOESS used by Hansen et al.'s BSmooth. The old name was misleading.
    """
    import warnings
    warnings.warn(
        "smooth_methylation_bsmooth is deprecated; use "
        "smooth_methylation_gaussian instead. The implementation is "
        "Gaussian-kernel smoothing, not local LOESS, so the old name "
        "misrepresented the method.",
        DeprecationWarning,
        stacklevel=2,
    )
    return smooth_methylation_gaussian(*args, **kwargs)