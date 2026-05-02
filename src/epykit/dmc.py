"""Differential methylation calling (DMC) on partitioned Parquet stores.

This module implements per-chromosome differential methylation analysis using:
  - Fisher exact test (MVP default)
  - Placeholder for beta-binomial and GLM tests (phase 2)
  - Chromosome-wise processing to maintain bounded memory
  - Genome-wide multiple testing correction
"""

from typing import Optional, Tuple, List, Callable
import numpy as np
import polars as pl
from scipy import stats as sp_stats
import logging

logger = logging.getLogger(__name__)


def fisher_exact_vectorized(
    meth_a: np.ndarray,
    unmeth_a: np.ndarray,
    meth_b: np.ndarray,
    unmeth_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Fisher exact test for methylation differences.

    Computes two-sided p-value and log2 odds ratio for each site.

    Parameters
    ----------
    meth_a, unmeth_a : np.ndarray (n_sites,)
        Methylated and unmethylated counts for group A
    meth_b, unmeth_b : np.ndarray (n_sites,)
        Methylated and unmethylated counts for group B

    Returns
    -------
    pvals : np.ndarray (n_sites,)
        Two-sided p-values from Fisher exact test
    log2_or : np.ndarray (n_sites,)
        Log2 odds ratios (NaN where counts are 0)
    """
    meth_a = np.asarray(meth_a, dtype=np.int64)
    unmeth_a = np.asarray(unmeth_a, dtype=np.int64)
    meth_b = np.asarray(meth_b, dtype=np.int64)
    unmeth_b = np.asarray(unmeth_b, dtype=np.int64)

    row1 = meth_a + unmeth_a
    row2 = meth_b + unmeth_b
    col1 = meth_a + meth_b
    total = row1 + row2

    pvals = np.full(len(meth_a), np.nan, dtype=np.float64)
    log2_or = np.full(len(meth_a), np.nan, dtype=np.float64)

    valid = (row1 > 0) & (row2 > 0)
    if np.any(valid):
        denominator = unmeth_a[valid] * meth_b[valid]
        numerator = meth_a[valid] * unmeth_b[valid]

        odds_ratio = np.full(denominator.shape, np.nan, dtype=np.float64)
        np.divide(numerator, denominator, out=odds_ratio, where=denominator > 0)
        odds_ratio = np.where(
            denominator > 0,
            odds_ratio,
            np.where(numerator > 0, np.inf, np.nan),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            log2_or[valid] = np.where(odds_ratio > 0, np.log2(odds_ratio), np.nan)

        # The older implementation used a hypergeometric tail approximation,
        # which is much cheaper than calling scipy.stats.fisher_exact once
        # per site.
        pvals_valid = sp_stats.hypergeom.sf(
            meth_a[valid] - 1,
            total[valid],
            col1[valid],
            row1[valid],
        )
        pvals[valid] = np.minimum(2.0 * pvals_valid, 1.0)

    return pvals, log2_or


def calculate_diff_meth_chromosome(
    chrom_df: pl.DataFrame,
    samples_case: list[str],
    samples_control: list[str],
    test: str = "fisher",
) -> pl.DataFrame:
    """Perform differential methylation analysis for one chromosome.

    Groups samples into case and control, collapses counts across replicates (sum),
    and performs statistical test for each CpG.

    Parameters
    ----------
    chrom_df : pl.DataFrame
        Data for one chromosome from all samples. Columns:
        chrom, pos, strand, N_meth, N_unmeth, coverage, sample
    samples_case, samples_control : list[str]
        Sample identifiers for each group
    test : str
        Statistical test to use: "fisher" (MVP), "glm", "beta_binomial"

    Returns
    -------
    pl.DataFrame
        Columns: chrom, pos, strand, n_case, n_control, pvalue, log2_odds_ratio, meth_diff
        Sorted by chrom, pos
    """
    if len(chrom_df) == 0:
        # Empty chromosome; return empty result with proper schema
        return pl.DataFrame(
            schema={
                "chrom": pl.Utf8,
                "pos": pl.Int32,
                "strand": pl.Utf8,
                "n_case": pl.Int32,
                "n_control": pl.Int32,
                "pvalue": pl.Float32,
                "log2_odds_ratio": pl.Float32,
                "meth_diff": pl.Float32,
            }
        )

    # Verify samples present
    present_samples = set(chrom_df["sample"].unique().to_list())
    expected_samples = set(samples_case + samples_control)
    if not expected_samples.issubset(present_samples):
        missing = expected_samples - present_samples
        logger.warning(f"Missing samples: {missing}. Proceeding with available data.")

    # Filter to only requested samples
    all_samples = samples_case + samples_control
    chrom_df = chrom_df.filter(pl.col("sample").is_in(all_samples))

    if len(chrom_df) == 0:
        # No data after filtering
        return pl.DataFrame(
            schema={
                "chrom": pl.Utf8,
                "pos": pl.Int32,
                "strand": pl.Utf8,
                "n_case": pl.Int32,
                "n_control": pl.Int32,
                "pvalue": pl.Float32,
                "log2_odds_ratio": pl.Float32,
                "meth_diff": pl.Float32,
            }
        )

    # Pivot: one row per CpG, columns like N_meth_s1, N_unmeth_s1, coverage_s1, ...
    wide_df = chrom_df.pivot(
        index=["chrom", "pos", "strand"],
        on="sample",
        values=["N_meth", "N_unmeth", "coverage"],
    )

    # Build column lists for case and control
    meth_cols_case = [
        f"N_meth_{s}" for s in samples_case if f"N_meth_{s}" in wide_df.columns
    ]
    meth_cols_ctrl = [
        f"N_meth_{s}" for s in samples_control if f"N_meth_{s}" in wide_df.columns
    ]
    cov_cols_case = [
        f"coverage_{s}" for s in samples_case if f"coverage_{s}" in wide_df.columns
    ]
    cov_cols_ctrl = [
        f"coverage_{s}" for s in samples_control if f"coverage_{s}" in wide_df.columns
    ]

    if not meth_cols_case or not meth_cols_ctrl:
        logger.warning(f"Incomplete sample coverage for chromosome")
        return pl.DataFrame(
            schema={
                "chrom": pl.Utf8,
                "pos": pl.Int32,
                "strand": pl.Utf8,
                "n_case": pl.Int32,
                "n_control": pl.Int32,
                "pvalue": pl.Float32,
                "log2_odds_ratio": pl.Float32,
                "meth_diff": pl.Float32,
            }
        )

    # Sum counts across replicates in each group
    meth_case = wide_df.select(meth_cols_case).sum_horizontal().to_numpy()
    meth_ctrl = wide_df.select(meth_cols_ctrl).sum_horizontal().to_numpy()
    cov_case = wide_df.select(cov_cols_case).sum_horizontal().to_numpy()
    cov_ctrl = wide_df.select(cov_cols_ctrl).sum_horizontal().to_numpy()
    unmeth_case = cov_case - meth_case
    unmeth_ctrl = cov_ctrl - meth_ctrl

    # Perform test
    if test == "fisher":
        pvals, log2_ors = fisher_exact_vectorized(
            meth_case, unmeth_case, meth_ctrl, unmeth_ctrl
        )
    else:
        raise NotImplementedError(f"Test '{test}' not yet implemented")

    # Compute methylation difference (beta difference)
    beta_case = meth_case / np.maximum(cov_case, 1)  # avoid division by zero
    beta_ctrl = meth_ctrl / np.maximum(cov_ctrl, 1)
    meth_diff = beta_case - beta_ctrl

    # Build results dataframe
    results = wide_df.select(["chrom", "pos", "strand"]).with_columns(
        [
            pl.Series("n_case", [len(samples_case)] * len(wide_df)),
            pl.Series("n_control", [len(samples_control)] * len(wide_df)),
            pl.Series("pvalue", pvals),
            pl.Series("log2_odds_ratio", log2_ors),
            pl.Series("meth_diff", meth_diff),
        ]
    ).sort(["chrom", "pos"])

    return results


def process_chromosomes_dmc(
    methylstore_path: str,
    samples_case: list[str],
    samples_control: list[str],
    test: str = "fisher",
    chromosomes: Optional[list[str]] = None,
    unite: bool = True,
) -> pl.DataFrame:
    """Process differential methylation for all chromosomes.

    Iterates through chromosomes, loads data, performs DMC, and concatenates results.

    Parameters
    ----------
    methylstore_path : str
        Path to partitioned Parquet methylstore
    samples_case, samples_control : list[str]
        Sample identifiers for case and control groups
    test : str
        Statistical test: "fisher" (default)
    chromosomes : list[str], optional
        Specific chromosomes to process. If None, processes all found in data.
    unite : bool
        If True, use only sites present in all samples (intersect). Default True.

    Returns
    -------
    pl.DataFrame
        Columns: chrom, pos, strand, n_case, n_control, pvalue, log2_odds_ratio, meth_diff
        All chromosomes concatenated; unsorted.
    """
    from pathlib import Path
    from .filter import intersect_sites, load_chromosome_data

    methylstore_path = Path(methylstore_path)
    all_samples = samples_case + samples_control

    # Auto-detect chromosomes if not provided
    if chromosomes is None:
        # Find unique chromosomes in store
        chrom_dirs = set()
        for sample_dir in methylstore_path.glob("sample=*"):
            for chrom_dir in sample_dir.glob("chrom=*"):
                chrom_name = chrom_dir.name.replace("chrom=", "")
                chrom_dirs.add(chrom_name)
        chromosomes = sorted(list(chrom_dirs))

    logger.info(f"Processing {len(chromosomes)} chromosomes with test={test}")

    # Compute intersection sites if needed
    site_intersect = None
    if unite:
        logger.info("Computing intersection of sites across samples...")
        site_intersect = intersect_sites(str(methylstore_path), all_samples)
        logger.info(f"  Intersection contains {len(site_intersect)} sites")

    # Process each chromosome
    result_dfs = []
    for i, chrom in enumerate(chromosomes):
        logger.info(f"[{i+1}/{len(chromosomes)}] Processing {chrom}...")

        try:
            # Load chromosome data
            chrom_df = load_chromosome_data(
                str(methylstore_path),
                chrom,
                all_samples,
                site_intersect=site_intersect,
            )

            if len(chrom_df) == 0:
                logger.warning(f"  No data for {chrom}; skipping")
                continue

            # Perform DMC
            chrom_results = calculate_diff_meth_chromosome(
                chrom_df, samples_case, samples_control, test=test
            )

            if len(chrom_results) > 0:
                result_dfs.append(chrom_results)
                logger.info(f"  {len(chrom_results)} sites tested")
            else:
                logger.warning(f"  No results for {chrom}")

        except Exception as e:
            logger.error(f"Error processing {chrom}: {e}")
            raise

    # Concatenate results
    if not result_dfs:
        logger.warning("No results generated; returning empty dataframe")
        return pl.DataFrame(
            schema={
                "chrom": pl.Utf8,
                "pos": pl.Int32,
                "strand": pl.Utf8,
                "n_case": pl.Int32,
                "n_control": pl.Int32,
                "pvalue": pl.Float32,
                "log2_odds_ratio": pl.Float32,
                "meth_diff": pl.Float32,
            }
        )

    combined = pl.concat(result_dfs)
    logger.info(f"Total DMC sites: {len(combined)}")

    return combined


def apply_multiple_testing_correction(
    dmc_results: pl.DataFrame,
    method: str = "fdr_bh",
) -> pl.DataFrame:
    """Apply genome-wide multiple testing correction to DMC results.

    Parameters
    ----------
    dmc_results : pl.DataFrame
        DMC results with 'pvalue' column
    method : str
        Correction method: "fdr_bh" (Benjamini-Hochberg, default), "bonferroni"

    Returns
    -------
    pl.DataFrame
        Input dataframe with added 'qvalue' column
    """
    from statsmodels.stats.multitest import multipletests

    pvals = dmc_results["pvalue"].to_numpy()

    # Handle NaN values
    nan_mask = np.isnan(pvals)
    pvals_clean = np.where(nan_mask, 1.0, pvals)

    # Apply correction
    reject, qvals, _, _ = multipletests(pvals_clean, method=method)

    # Restore NaN in qvalues
    qvals = np.where(nan_mask, np.nan, qvals)

    results = dmc_results.with_columns(
        [pl.Series("qvalue", qvals), pl.Series("reject", reject)]
    )

    return results
