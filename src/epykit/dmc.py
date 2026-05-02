"""Differential methylation calling (DMC) on partitioned Parquet stores.

Memory model
------------
The previous implementation loaded all samples for a chromosome into a single
long DataFrame and then pivoted it wide before running tests.  For chr1 with
5 M CpGs and 6 samples that pivot alone materialises ~360 M cells, reliably
causing OOM on typical WGBS workloads.

This rewrite keeps peak memory per chromosome to roughly:

    n_sites × (n_samples_case + n_samples_control) × 4 bytes   (beta matrix)
  + n_sites × 4 int32 arrays                                    (running sums)

which for the same example is ~180 MB instead of ~3 GB.

Key design decisions:
  1. Samples are loaded one Parquet file at a time, aligned to canonical
     positions via a left-join, and freed before the next sample is read.
  2. The pivot is eliminated entirely.  Running meth/coverage sums and per-
     replicate beta arrays are built up incrementally as numpy arrays.
  3. Each chromosome result is written to a temporary Parquet file immediately
     after the test and freed from memory (gc.collect() called explicitly).
     The final concat reads those small files rather than accumulating all
     chromosomes in a list in RAM.
  4. Site intersection is computed per-chromosome so only one chromosome's
     worth of positions is ever in memory at once.

Biological fixes (carried over):
  BIO-3: equal-weight per-replicate beta averaging (mean_beta_*)
  BIO-4: missing sites filled with 0 counts; excluded from beta mean via NaN
"""

from __future__ import annotations

import gc
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import polars as pl
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

_EMPTY_SCHEMA = {
    "chrom": pl.Utf8,
    "pos": pl.Int32,
    "strand": pl.Utf8,
    "n_case": pl.Int32,
    "n_control": pl.Int32,
    "mean_beta_case": pl.Float32,
    "mean_beta_control": pl.Float32,
    "pvalue": pl.Float64,
    "log2_odds_ratio": pl.Float64,
    "meth_diff": pl.Float32,
}


# ---------------------------------------------------------------------------
# Core statistical test
# ---------------------------------------------------------------------------

def fisher_exact_vectorized(
    meth_a: np.ndarray,
    unmeth_a: np.ndarray,
    meth_b: np.ndarray,
    unmeth_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised Fisher exact test via hypergeometric tail approximation.

    Parameters
    ----------
    meth_a, unmeth_a : np.ndarray (n_sites,)
        Summed methylated / unmethylated counts for group A (across replicates)
    meth_b, unmeth_b : np.ndarray (n_sites,)
        Summed methylated / unmethylated counts for group B

    Returns
    -------
    pvals : np.ndarray (n_sites,)  float64
    log2_or : np.ndarray (n_sites,)  float64
    """
    meth_a   = np.asarray(meth_a,   dtype=np.int64)
    unmeth_a = np.asarray(unmeth_a, dtype=np.int64)
    meth_b   = np.asarray(meth_b,   dtype=np.int64)
    unmeth_b = np.asarray(unmeth_b, dtype=np.int64)

    row1  = meth_a + unmeth_a
    row2  = meth_b + unmeth_b
    col1  = meth_a + meth_b
    total = row1 + row2

    n       = len(meth_a)
    pvals   = np.full(n, np.nan, dtype=np.float64)
    log2_or = np.full(n, np.nan, dtype=np.float64)

    valid = (row1 > 0) & (row2 > 0)
    if np.any(valid):
        denom = unmeth_a[valid] * meth_b[valid]
        numer = meth_a[valid]  * unmeth_b[valid]

        odds_ratio = np.full(denom.shape, np.nan, dtype=np.float64)
        np.divide(numer, denom, out=odds_ratio, where=denom > 0)
        odds_ratio = np.where(
            denom > 0,
            odds_ratio,
            np.where(numer > 0, np.inf, np.nan),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            log2_or[valid] = np.where(odds_ratio > 0, np.log2(odds_ratio), np.nan)

        pvals_valid = sp_stats.hypergeom.sf(
            meth_a[valid] - 1,
            total[valid],
            col1[valid],
            row1[valid],
        )
        pvals[valid] = np.minimum(2.0 * pvals_valid, 1.0)

    return pvals, log2_or


# ---------------------------------------------------------------------------
# Internal per-chromosome helpers
# ---------------------------------------------------------------------------

def _detect_chromosomes(methylstore_path: Path) -> list[str]:
    chroms: set[str] = set()
    for sample_dir in methylstore_path.glob("sample=*"):
        for chrom_dir in sample_dir.glob("chrom=*"):
            chroms.add(chrom_dir.name.removeprefix("chrom="))
    return sorted(chroms)


def _intersect_chrom(
    methylstore_path: Path,
    chrom: str,
    samples: list[str],
) -> pl.DataFrame:
    """Return (pos, strand) rows present in every sample for one chromosome.

    Reads only pos + strand columns per file.  Returns an empty DataFrame if
    any sample is missing the chromosome (strict intersection).
    """
    intersect: Optional[pl.DataFrame] = None

    for sample in samples:
        part_file = (
            methylstore_path
            / f"sample={sample}"
            / f"chrom={chrom}"
            / "part-0.parquet"
        )
        if not part_file.exists():
            logger.warning(
                f"  Sample '{sample}' missing {chrom}; "
                "chromosome excluded from intersection"
            )
            return pl.DataFrame({
                "pos":    pl.Series([], dtype=pl.Int32),
                "strand": pl.Series([], dtype=pl.Utf8),
            })

        sites = pl.read_parquet(str(part_file), columns=["pos", "strand"]).unique()

        if intersect is None:
            intersect = sites
        else:
            intersect = intersect.join(sites, on=["pos", "strand"], how="inner")

        if len(intersect) == 0:
            break

    if intersect is None:
        return pl.DataFrame({
            "pos":    pl.Series([], dtype=pl.Int32),
            "strand": pl.Series([], dtype=pl.Utf8),
        })

    return intersect.sort("pos")


def _union_chrom(
    methylstore_path: Path,
    chrom: str,
    samples: list[str],
) -> pl.DataFrame:
    """Return (pos, strand) rows seen in at least one sample for one chromosome."""
    site_dfs: list[pl.DataFrame] = []
    for sample in samples:
        part_file = (
            methylstore_path
            / f"sample={sample}"
            / f"chrom={chrom}"
            / "part-0.parquet"
        )
        if part_file.exists():
            site_dfs.append(
                pl.read_parquet(str(part_file), columns=["pos", "strand"])
            )
    if not site_dfs:
        return pl.DataFrame({
            "pos":    pl.Series([], dtype=pl.Int32),
            "strand": pl.Series([], dtype=pl.Utf8),
        })
    return pl.concat(site_dfs).unique().sort("pos")


def _load_sample_chrom(
    methylstore_path: Path,
    chrom: str,
    sample: str,
    canonical_pos: pl.DataFrame,  # single column: pos (Int32)
) -> tuple[np.ndarray, np.ndarray]:
    """Load N_meth and coverage for ONE sample / ONE chromosome.

    Left-joins to canonical_pos so arrays are always aligned to the same
    site order.  Missing sites are filled with 0.

    Returns
    -------
    meth : np.ndarray (n_sites,)  int32
    cov  : np.ndarray (n_sites,)  int32
    """
    part_file = (
        methylstore_path
        / f"sample={sample}"
        / f"chrom={chrom}"
        / "part-0.parquet"
    )
    n_sites = len(canonical_pos)

    if not part_file.exists():
        logger.warning(f"  Missing Parquet: {part_file}")
        return (
            np.zeros(n_sites, dtype=np.int32),
            np.zeros(n_sites, dtype=np.int32),
        )

    df = pl.read_parquet(str(part_file), columns=["pos", "N_meth", "coverage"])
    aligned = canonical_pos.join(df, on="pos", how="left").fill_null(0)

    return (
        aligned["N_meth"].to_numpy().astype(np.int32),
        aligned["coverage"].to_numpy().astype(np.int32),
    )


def _process_one_chromosome(
    methylstore_path: Path,
    chrom: str,
    canonical_df: pl.DataFrame,  # columns: pos, strand
    samples_case: list[str],
    samples_control: list[str],
    test: str,
) -> pl.DataFrame:
    """Run DMC for one chromosome loading one sample at a time.

    Never holds more than one sample's raw data in memory simultaneously.
    The pivot that caused OOM in the previous implementation is gone.

    Steps
    -----
    1. Case samples: load → add to running int64 sums → compute float32 beta
       column → free raw arrays.
    2. Same for control samples.
    3. Fisher test on pooled int64 sums.
    4. Stack beta columns into (n_sites, n_replicates) matrix → nanmean
       for equal-weight effect size (BIO-3).
    5. Return result DataFrame.
    """
    n_sites = len(canonical_df)
    if n_sites == 0:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    canonical_pos = canonical_df.select("pos")

    # Running sums (int64 to avoid overflow when aggregating many samples)
    meth_case_sum = np.zeros(n_sites, dtype=np.int64)
    cov_case_sum  = np.zeros(n_sites, dtype=np.int64)
    meth_ctrl_sum = np.zeros(n_sites, dtype=np.int64)
    cov_ctrl_sum  = np.zeros(n_sites, dtype=np.int64)

    # Per-replicate beta columns (float32 to halve memory vs float64)
    beta_case_cols: list[np.ndarray] = []
    beta_ctrl_cols: list[np.ndarray] = []

    # --- Case samples ---
    for sample in samples_case:
        meth, cov = _load_sample_chrom(methylstore_path, chrom, sample, canonical_pos)
        meth_case_sum += meth
        cov_case_sum  += cov
        with np.errstate(invalid="ignore", divide="ignore"):
            beta = np.where(cov > 0, meth.astype(np.float32) / cov, np.nan)
        beta_case_cols.append(beta.astype(np.float32))
        del meth, cov, beta

    # --- Control samples ---
    for sample in samples_control:
        meth, cov = _load_sample_chrom(methylstore_path, chrom, sample, canonical_pos)
        meth_ctrl_sum += meth
        cov_ctrl_sum  += cov
        with np.errstate(invalid="ignore", divide="ignore"):
            beta = np.where(cov > 0, meth.astype(np.float32) / cov, np.nan)
        beta_ctrl_cols.append(beta.astype(np.float32))
        del meth, cov, beta

    # --- Statistical test ---
    if test != "fisher":
        raise NotImplementedError(f"Test '{test}' not yet implemented")

    unmeth_case_sum = cov_case_sum - meth_case_sum
    unmeth_ctrl_sum = cov_ctrl_sum - meth_ctrl_sum

    pvals, log2_ors = fisher_exact_vectorized(
        meth_case_sum, unmeth_case_sum,
        meth_ctrl_sum, unmeth_ctrl_sum,
    )
    del meth_case_sum, unmeth_case_sum, meth_ctrl_sum, unmeth_ctrl_sum
    del cov_case_sum, cov_ctrl_sum

    # --- BIO-3: equal-weight beta averaging ---
    # Stack into (n_sites, n_replicates); nanmean excludes zero-coverage sites.
    beta_case_mat = np.stack(beta_case_cols, axis=1)
    beta_ctrl_mat = np.stack(beta_ctrl_cols, axis=1)
    del beta_case_cols, beta_ctrl_cols

    mean_beta_case = np.nanmean(beta_case_mat, axis=1).astype(np.float32)
    mean_beta_ctrl = np.nanmean(beta_ctrl_mat, axis=1).astype(np.float32)
    meth_diff      = (mean_beta_case - mean_beta_ctrl).astype(np.float32)
    del beta_case_mat, beta_ctrl_mat

    return pl.DataFrame({
        "chrom":             pl.Series([chrom] * n_sites, dtype=pl.Utf8),
        "pos":               canonical_df["pos"],
        "strand":            canonical_df["strand"],
        "n_case":            pl.Series(
                                 np.full(n_sites, len(samples_case),    dtype=np.int32)),
        "n_control":         pl.Series(
                                 np.full(n_sites, len(samples_control), dtype=np.int32)),
        "mean_beta_case":    pl.Series(mean_beta_case),
        "mean_beta_control": pl.Series(mean_beta_ctrl),
        "pvalue":            pl.Series(pvals),
        "log2_odds_ratio":   pl.Series(log2_ors),
        "meth_diff":         pl.Series(meth_diff),
    }).sort("pos")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_chromosomes_dmc(
    methylstore_path: str,
    samples_case: list[str],
    samples_control: list[str],
    test: str = "fisher",
    chromosomes: Optional[list[str]] = None,
    unite: bool = True,
) -> pl.DataFrame:
    """Process differential methylation for all chromosomes.

    Memory: one chromosome active at a time; within that, one sample at a
    time.  Chromosome results are written to a temporary directory on disk
    and freed (gc.collect()) immediately; the final DataFrame is assembled
    from those files only after all chromosomes are done.

    Parameters
    ----------
    methylstore_path : str
        Path to filtered partitioned Parquet methylstore.
    samples_case, samples_control : list[str]
        Sample identifiers for case and control groups.
    test : str
        Statistical test: "fisher" (default).
    chromosomes : list[str], optional
        Chromosomes to process. Auto-detected when None.
    unite : bool
        If True (default), test only CpG sites covered in every sample.

    Returns
    -------
    pl.DataFrame
        Columns: chrom, pos, strand, n_case, n_control,
                 mean_beta_case, mean_beta_control,
                 pvalue, log2_odds_ratio, meth_diff
    """
    store = Path(methylstore_path)
    all_samples = samples_case + samples_control

    if chromosomes is None:
        chromosomes = _detect_chromosomes(store)
        logger.info(f"Auto-detected {len(chromosomes)} chromosomes")

    logger.info(
        f"DMC: {len(samples_case)} case / {len(samples_control)} control, "
        f"test={test}, unite={unite}"
    )

    with tempfile.TemporaryDirectory(prefix="epykit_dmc_") as tmpdir:
        tmp = Path(tmpdir)
        written: list[Path] = []

        for i, chrom in enumerate(chromosomes):
            logger.info(f"[{i + 1}/{len(chromosomes)}] {chrom}")

            # --- Site selection ---
            canonical_df = (
                _intersect_chrom(store, chrom, all_samples)
                if unite
                else _union_chrom(store, chrom, all_samples)
            )

            if len(canonical_df) == 0:
                logger.warning(f"  No sites for {chrom}; skipping")
                continue

            logger.info(f"  {len(canonical_df):,} sites to test")

            # --- Sample-at-a-time DMC ---
            chrom_result = _process_one_chromosome(
                store, chrom, canonical_df,
                samples_case, samples_control, test,
            )
            del canonical_df

            if len(chrom_result) == 0:
                logger.warning(f"  No results for {chrom}")
                continue

            # --- Write to temp disk; free RAM immediately ---
            tmp_file = tmp / f"{chrom}.parquet"
            chrom_result.write_parquet(str(tmp_file))
            written.append(tmp_file)
            logger.info(f"  {len(chrom_result):,} sites → staged to disk")

            del chrom_result
            gc.collect()

        # --- Assemble final result from temp files ---
        if not written:
            logger.warning("No results generated")
            return pl.DataFrame(schema=_EMPTY_SCHEMA)

        logger.info(f"Assembling results from {len(written)} chromosome file(s)...")
        combined = pl.concat([pl.read_parquet(str(f)) for f in written])
        logger.info(f"Total DMC sites: {len(combined):,}")
        return combined


def calculate_diff_meth_chromosome(
    chrom_df: pl.DataFrame,
    samples_case: list[str],
    samples_control: list[str],
    test: str = "fisher",
) -> pl.DataFrame:
    """Legacy entry-point kept for unit-test compatibility.

    Writes a temporary mini-store and calls _process_one_chromosome.
    For production use, call process_chromosomes_dmc directly.
    """
    import warnings
    warnings.warn(
        "calculate_diff_meth_chromosome is deprecated; "
        "use process_chromosomes_dmc for production workloads.",
        DeprecationWarning,
        stacklevel=2,
    )

    if len(chrom_df) == 0:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    chroms = chrom_df["chrom"].unique().to_list()
    if len(chroms) != 1:
        raise ValueError(f"Expected exactly one chromosome; got {chroms}")
    chrom = chroms[0]

    with tempfile.TemporaryDirectory(prefix="epykit_legacy_") as tmpdir:
        store = Path(tmpdir) / "store"
        all_samples = samples_case + samples_control

        for sample in all_samples:
            sample_df = chrom_df.filter(pl.col("sample") == sample)
            if len(sample_df) == 0:
                continue
            part_dir = store / f"sample={sample}" / f"chrom={chrom}"
            part_dir.mkdir(parents=True, exist_ok=True)
            sample_df.write_parquet(str(part_dir / "part-0.parquet"))

        canonical_df = _intersect_chrom(store, chrom, all_samples)
        return _process_one_chromosome(
            store, chrom, canonical_df, samples_case, samples_control, test
        )


def apply_multiple_testing_correction(
    dmc_results: pl.DataFrame,
    method: str = "fdr_bh",
) -> pl.DataFrame:
    """Apply genome-wide multiple testing correction.

    Parameters
    ----------
    dmc_results : pl.DataFrame
        DMC results with a 'pvalue' column (float64, may contain NaN).
    method : str
        Any method accepted by statsmodels.stats.multitest.multipletests.
        Default: "fdr_bh" (Benjamini-Hochberg).

    Returns
    -------
    pl.DataFrame
        Input with added 'qvalue' (float64) and 'reject' (bool) columns.
    """
    from statsmodels.stats.multitest import multipletests

    pvals       = dmc_results["pvalue"].to_numpy()
    nan_mask    = np.isnan(pvals)
    pvals_clean = np.where(nan_mask, 1.0, pvals)

    reject, qvals, _, _ = multipletests(pvals_clean, method=method)

    qvals  = np.where(nan_mask, np.nan, qvals)
    reject = np.where(nan_mask, False,  reject)

    return dmc_results.with_columns([
        pl.Series("qvalue", qvals),
        pl.Series("reject", reject),
    ])