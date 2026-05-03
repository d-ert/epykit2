"""Quality control and reporting for Parquet methylation stores.

Phase 5 of the epykit pipeline:
  - bisulfite_conversion_rate: estimates library conversion efficiency from
    CHH-context methylation (should be <0.5 % for a high-quality WGBS run).
  - global_methylation_report: per-sample, per-context global methylation
    levels with outlier detection.
  - coverage_uniformity: per-chromosome breadth-of-coverage statistics with
    automatic flagging of low-coverage samples.

All functions read from the partitioned Parquet methylstore layout produced
by epykit.convert and are intentionally lightweight: only the columns
required for each metric are loaded.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Thresholds used by coverage_uniformity for flagging
_MIN_GENOME_COVERAGE_FRACTION = 0.80   # 80 % of CpGs at ≥1×
_CONVERSION_WARNING_THRESHOLD  = 0.005  # 0.5 % CHH methylation


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bisulfite_conversion_rate(
    methylstore_path: str,
    sample: str,
    chh_context_store: str,
) -> float:
    """Estimate bisulfite conversion efficiency from CHH-context methylation.

    Under complete bisulfite conversion, non-CpG cytosines (CHH context) are
    converted to uracil and read as thymine.  Residual CHH methylation
    therefore reflects incomplete conversion.  Conversion efficiency is
    estimated as 1 − mean(CHH β).

    A value below 99.5 % (i.e. >0.5 % residual CHH methylation) should be
    flagged as a potential quality issue.

    Parameters
    ----------
    methylstore_path : str
        Path to the CpG Parquet methylstore (used only to verify that the
        sample exists; not directly read by this function).
    sample : str
        Sample identifier.
    chh_context_store : str
        Path to a *separate* Parquet methylstore generated from CHH-context
        Bismark output for the same sample.

    Returns
    -------
    float
        Estimated bisulfite conversion rate in [0, 1].
        Values close to 1.0 indicate high-quality conversion.

    Raises
    ------
    ValueError
        If no CHH data is found for the given sample.
    """
    chh_store = Path(chh_context_store)
    sample_dir = chh_store / f"sample={sample}"

    if not sample_dir.exists():
        raise ValueError(
            f"CHH store does not contain sample '{sample}': {sample_dir}"
        )

    parts = list(sample_dir.rglob("part-*.parquet"))
    if not parts:
        raise ValueError(
            f"No Parquet files found for sample '{sample}' in {chh_store}"
        )

    # Load only the count columns; ignore chrom partition
    lf = pl.scan_parquet(str(sample_dir / "**" / "part-*.parquet"))
    agg = lf.select([
        pl.sum("N_meth").alias("total_meth"),
        pl.sum("coverage").alias("total_cov"),
    ]).collect()

    total_meth = int(agg["total_meth"][0])
    total_cov  = int(agg["total_cov"][0])

    if total_cov == 0:
        raise ValueError(
            f"Zero CHH coverage for sample '{sample}'; cannot estimate rate."
        )

    mean_chh_methylation = total_meth / total_cov
    conversion_rate      = 1.0 - mean_chh_methylation

    if mean_chh_methylation > _CONVERSION_WARNING_THRESHOLD:
        logger.warning(
            "Sample '%s': CHH methylation = %.3f %% > %.1f %% threshold; "
            "consider checking library quality.",
            sample,
            mean_chh_methylation * 100,
            _CONVERSION_WARNING_THRESHOLD * 100,
        )
    else:
        logger.info(
            "Sample '%s': conversion rate = %.4f %%",
            sample, conversion_rate * 100,
        )

    return float(conversion_rate)


def global_methylation_report(
    methylstore_path: str,
    samples: list[str],
    contexts: list[str] | None = None,
) -> pl.DataFrame:
    """Compute per-sample, per-context global methylation levels.

    Reads methylation counts summed across the entire genome for each
    requested sample.  When the methylstore contains a ``context`` column
    (as written by epykit.convert), statistics are broken down by context
    (CpG / CHG / CHH); otherwise a single "CpG" row is reported.

    Outlier detection: samples whose global CpG methylation deviates by more
    than 3 MAD from the cohort median are flagged.

    Parameters
    ----------
    methylstore_path : str
        Path to the partitioned Parquet methylstore.
    samples : list[str]
        Sample identifiers to include.
    contexts : list[str], optional
        Contexts to report (default: all contexts found in the store).

    Returns
    -------
    pl.DataFrame
        Columns: sample (Utf8), context (Utf8), n_sites (Int64),
                 global_methylation (Float64), is_outlier (bool).
    """
    store = Path(methylstore_path)
    rows: list[dict] = []

    for sample in samples:
        sample_dir = store / f"sample={sample}"
        if not sample_dir.exists():
            logger.warning("Sample '%s' not found; skipping", sample)
            continue

        lf = pl.scan_parquet(str(sample_dir / "**" / "part-*.parquet"))

        # Determine if the context column is present
        schema = lf.collect_schema()
        has_context = "context" in schema

        if has_context:
            agg = (
                lf
                .group_by("context")
                .agg([
                    pl.len().alias("n_sites"),
                    pl.sum("N_meth").alias("total_meth"),
                    pl.sum("coverage").alias("total_cov"),
                ])
                .collect()
            )
        else:
            agg = (
                lf
                .select([
                    pl.len().alias("n_sites"),
                    pl.sum("N_meth").alias("total_meth"),
                    pl.sum("coverage").alias("total_cov"),
                ])
                .collect()
                .with_columns(pl.lit("CpG").alias("context"))
            )

        for row in agg.iter_rows(named=True):
            ctx = row.get("context", "CpG")
            if contexts and ctx not in contexts:
                continue
            total_cov  = row["total_cov"]
            total_meth = row["total_meth"]
            rows.append({
                "sample":             sample,
                "context":            ctx,
                "n_sites":            row["n_sites"],
                "global_methylation": (total_meth / total_cov)
                                       if total_cov > 0 else float("nan"),
            })

    if not rows:
        return pl.DataFrame({
            "sample":             pl.Series([], dtype=pl.Utf8),
            "context":            pl.Series([], dtype=pl.Utf8),
            "n_sites":            pl.Series([], dtype=pl.Int64),
            "global_methylation": pl.Series([], dtype=pl.Float64),
            "is_outlier":         pl.Series([], dtype=pl.Boolean),
        })

    result = pl.DataFrame(rows)

    # --- Outlier detection (MAD, per context) ---
    outlier_flags = np.zeros(len(result), dtype=bool)

    for ctx in result["context"].unique().to_list():
        ctx_mask = (result["context"] == ctx).to_numpy()
        meth_vals = result.filter(pl.col("context") == ctx)[
            "global_methylation"
        ].to_numpy()
        valid = ~np.isnan(meth_vals)

        if valid.sum() < 3:
            continue

        median = float(np.median(meth_vals[valid]))
        mad    = float(np.median(np.abs(meth_vals[valid] - median)))

        if mad == 0:
            continue

        z_scores = np.abs(meth_vals - median) / (1.4826 * mad)
        ctx_outliers = (z_scores > 3.0) & valid
        outlier_flags[ctx_mask] = ctx_outliers

        n_outliers = int(ctx_outliers.sum())
        if n_outliers:
            logger.warning(
                "global_methylation_report: %d outlier sample(s) detected "
                "in context %s (MAD threshold 3σ)",
                n_outliers, ctx,
            )

    return result.with_columns(
        pl.Series("is_outlier", outlier_flags, dtype=pl.Boolean)
    ).sort(["context", "sample"])


def coverage_uniformity(
    methylstore_path: str,
    sample: str,
    thresholds: list[int] | None = None,
) -> pl.DataFrame:
    """Compute per-chromosome coverage breadth statistics for one sample.

    Reports the fraction of CpG sites covered at each threshold depth and
    flags chromosomes (and the sample overall) when coverage breadth at 1×
    falls below 80 %.

    Parameters
    ----------
    methylstore_path : str
        Path to the partitioned Parquet methylstore.
    sample : str
        Sample identifier.
    thresholds : list[int], optional
        Coverage depth thresholds to report (default: [1, 5, 10]).

    Returns
    -------
    pl.DataFrame
        Columns: sample (Utf8), chrom (Utf8),
                 n_sites (Int64), mean_coverage (Float64),
                 frac_ge_1x (Float64), frac_ge_5x (Float64),
                 frac_ge_10x (Float64),   [one per threshold]
                 low_coverage_flag (bool).
        The final row has chrom="genome" and reports genome-wide aggregates.
    """
    if thresholds is None:
        thresholds = [1, 5, 10]

    store      = Path(methylstore_path)
    sample_dir = store / f"sample={sample}"

    if not sample_dir.exists():
        raise ValueError(
            f"Sample '{sample}' not found in methylstore: {sample_dir}"
        )

    chrom_rows: list[dict] = []

    for chrom_dir in sorted(sample_dir.glob("chrom=*")):
        chrom = chrom_dir.name.removeprefix("chrom=")
        parts = list(chrom_dir.glob("part-*.parquet"))
        if not parts:
            continue

        cov_series = pl.concat([
            pl.read_parquet(str(p), columns=["coverage"])["coverage"]
            for p in parts
        ])

        n       = len(cov_series)
        cov_arr = cov_series.to_numpy()
        row: dict = {
            "sample":        sample,
            "chrom":         chrom,
            "n_sites":       n,
            "mean_coverage": float(cov_arr.mean()) if n > 0 else float("nan"),
        }

        for t in thresholds:
            key      = f"frac_ge_{t}x"
            row[key] = float((cov_arr >= t).sum() / n) if n > 0 else float("nan")

        frac_1x = row.get("frac_ge_1x", float("nan"))
        row["low_coverage_flag"] = (
            (not np.isnan(frac_1x))
            and (frac_1x < _MIN_GENOME_COVERAGE_FRACTION)
        )

        chrom_rows.append(row)

        if row["low_coverage_flag"]:
            logger.warning(
                "Sample '%s', %s: only %.1f %% of sites covered at ≥1× "
                "(threshold: %.0f %%)",
                sample, chrom, frac_1x * 100,
                _MIN_GENOME_COVERAGE_FRACTION * 100,
            )

    if not chrom_rows:
        raise ValueError(f"No chromosome data found for sample '{sample}'")

    # --- Genome-wide aggregate row ---
    total_n       = sum(r["n_sites"] for r in chrom_rows)
    genome_row: dict = {
        "sample":        sample,
        "chrom":         "genome",
        "n_sites":       total_n,
        "mean_coverage": float(
            np.mean([r["mean_coverage"] for r in chrom_rows
                     if not np.isnan(r["mean_coverage"])])
        ) if total_n > 0 else float("nan"),
    }
    for t in thresholds:
        key = f"frac_ge_{t}x"
        values = [r[key] * r["n_sites"] for r in chrom_rows
                  if key in r and not np.isnan(r[key])]
        genome_row[key] = (sum(values) / total_n) if total_n > 0 else float("nan")

    frac_1x_genome = genome_row.get("frac_ge_1x", float("nan"))
    genome_row["low_coverage_flag"] = (
        (not np.isnan(frac_1x_genome))
        and (frac_1x_genome < _MIN_GENOME_COVERAGE_FRACTION)
    )
    chrom_rows.append(genome_row)

    # Build schema dynamically based on requested thresholds
    schema_extras: dict = {f"frac_ge_{t}x": pl.Float64 for t in thresholds}
    result = pl.DataFrame(chrom_rows).cast({
        "n_sites": pl.Int64,
        **schema_extras,
    })

    return result.sort(["chrom"])
