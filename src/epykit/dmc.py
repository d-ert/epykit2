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

Statistical tests
-----------------
  fisher  (default) — Fisher exact test via hypergeometric tail.
                      Recommended for 1–5 replicates / group.
                      Pools counts across replicates; ignores biological
                      variance.  Fast; fully vectorised.

  beta_binomial     — Welch t-test on per-replicate beta values with
                      Welch–Satterthwaite degrees of freedom (fast path,
                      default method="mom").  Properly accounts for
                      between-replicate variability unlike Fisher.
                      Recommended for ≥ 6 replicates / group.

Biological fixes (carried over):
  BIO-3: equal-weight per-replicate beta averaging (mean_beta_*)
  BIO-4: missing sites filled with 0 counts; excluded from beta mean via NaN
"""

from __future__ import annotations

import gc
import logging
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import polars as pl
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

_EMPTY_SCHEMA = {
    "chrom":             pl.Utf8,
    "pos":               pl.Int32,
    "strand":            pl.Utf8,
    "n_case":            pl.Int32,
    "n_control":         pl.Int32,
    "mean_beta_case":    pl.Float32,
    "mean_beta_control": pl.Float32,
    "pvalue":            pl.Float64,
    "log2_odds_ratio":   pl.Float64,
    "meth_diff":         pl.Float32,
}

# Guidance table surfaced in docstrings and logged at INFO level
_TEST_RECOMMENDATIONS = {
    range(1, 3):  "fisher",
    range(3, 6):  "fisher (report effect size; consider beta_binomial at ≥6)",
    range(6, 999):"beta_binomial",
}


# ---------------------------------------------------------------------------
# Core statistical tests
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


def beta_binomial_test(
    meth_counts: np.ndarray,
    total_counts: np.ndarray,
    group_labels: np.ndarray,
    method: str = "mom",
) -> Tuple[np.ndarray, np.ndarray]:
    """Beta-binomial–aware differential methylation test.

    Unlike ``fisher_exact_vectorized``, which pools counts across replicates
    and ignores between-replicate biological variability, this function
    computes a per-replicate beta value and tests for a difference in the
    group means while properly propagating biological variance.

    Two computational paths are available:

    ``method="mom"`` (default, fast)
        Welch t-test on per-replicate β values with Welch–Satterthwaite
        degrees of freedom.  Vectorised across all sites simultaneously.
        Accounts for unequal group sizes and unequal variances.  Recommended
        for production WGBS data (millions of sites).

    ``method="statsmodels"`` (slow, exact)
        Fits a ``BetaBinomialModel`` from ``statsmodels`` per site using MLE.
        Accurate but ~10 000× slower than the MOM path; use only for small
        validation datasets or targeted follow-up analysis.

    Recommended use by replicate count:

    | N replicates / group | Recommended test              |
    |----------------------|-------------------------------|
    | 1–2                  | fisher (only viable option)   |
    | 3–5                  | fisher + report effect size   |
    | 6+                   | beta_binomial (this function)  |

    Parameters
    ----------
    meth_counts : np.ndarray, shape (n_sites, n_replicates)
        Methylated read counts.  Columns ordered to match group_labels.
    total_counts : np.ndarray, shape (n_sites, n_replicates)
        Total coverage counts.  Same shape as meth_counts.
    group_labels : np.ndarray, shape (n_replicates,)
        0 = control, 1 = case.
    method : {"mom", "statsmodels"}
        Computational path (default "mom").

    Returns
    -------
    pvals : np.ndarray (n_sites,), float64
        Two-sided p-values.  NaN where variance is zero or coverage is absent.
    meth_diff : np.ndarray (n_sites,), float32
        mean_beta_case − mean_beta_control (equal-weight per replicate).
    """
    meth_counts  = np.asarray(meth_counts,  dtype=np.float64)
    total_counts = np.asarray(total_counts, dtype=np.float64)
    group_labels = np.asarray(group_labels, dtype=np.int32)

    if meth_counts.ndim == 1:
        meth_counts  = meth_counts[:, np.newaxis]
        total_counts = total_counts[:, np.newaxis]

    n_sites = meth_counts.shape[0]
    case_mask = group_labels == 1
    ctrl_mask = group_labels == 0
    n_case    = int(case_mask.sum())
    n_ctrl    = int(ctrl_mask.sum())

    if n_case < 2 or n_ctrl < 2:
        warnings.warn(
            f"beta_binomial_test requires ≥2 replicates per group; "
            f"got case={n_case}, control={n_ctrl}. "
            "Use fisher_exact_vectorized for 1-replicate comparisons.",
            UserWarning,
            stacklevel=2,
        )

    # Per-replicate beta (NaN where coverage = 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(total_counts > 0, meth_counts / total_counts, np.nan)

    beta_case = beta[:, case_mask]   # (n_sites, n_case)
    beta_ctrl = beta[:, ctrl_mask]   # (n_sites, n_ctrl)

    if method == "mom":
        return _beta_binom_mom(beta_case, beta_ctrl, n_case, n_ctrl, n_sites)

    elif method == "statsmodels":
        return _beta_binom_statsmodels(
            meth_counts, total_counts, case_mask, ctrl_mask, n_sites
        )

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose 'mom' or 'statsmodels'."
        )


def _beta_binom_mom(
    beta_case: np.ndarray,
    beta_ctrl: np.ndarray,
    n_case: int,
    n_ctrl: int,
    n_sites: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fast vectorised Welch t-test on per-replicate beta values.

    Properly accounts for unequal variances and unequal sample sizes via
    Welch–Satterthwaite degrees of freedom.  NaN propagates where all
    replicates in a group have zero coverage.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu_case = np.nanmean(beta_case, axis=1)
        mu_ctrl = np.nanmean(beta_ctrl, axis=1)

    # Bessel-corrected variance of the mean: s² / n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        var_mean_case = np.nanvar(beta_case, axis=1, ddof=1) / max(n_case, 1)
        var_mean_ctrl = np.nanvar(beta_ctrl, axis=1, ddof=1) / max(n_ctrl, 1)

    se = np.sqrt(var_mean_case + var_mean_ctrl)

    with np.errstate(invalid="ignore", divide="ignore"):
        t_stat = np.where(se > 0, (mu_case - mu_ctrl) / se, np.nan)

    # Welch–Satterthwaite degrees of freedom
    dof_num = (var_mean_case + var_mean_ctrl) ** 2
    dof_den = (
        np.where(n_case > 1, var_mean_case**2 / (n_case - 1), 0.0)
        + np.where(n_ctrl > 1, var_mean_ctrl**2 / (n_ctrl - 1), 0.0)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        dof = np.where(dof_den > 0, dof_num / dof_den, 1.0)
        dof = np.maximum(dof, 1.0)

    pvals     = 2.0 * sp_stats.t.sf(np.abs(t_stat), df=dof)
    meth_diff = (mu_case - mu_ctrl).astype(np.float32)

    # Propagate NaN from degenerate sites (all-zero coverage in a group)
    degenerate = np.isnan(mu_case) | np.isnan(mu_ctrl) | np.isnan(t_stat)
    pvals[degenerate]     = np.nan
    meth_diff[degenerate] = np.nan

    return pvals, meth_diff


def _beta_binom_statsmodels(
    meth_counts: np.ndarray,
    total_counts: np.ndarray,
    case_mask: np.ndarray,
    ctrl_mask: np.ndarray,
    n_sites: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-site statsmodels BetaBinomial MLE.

    WARNING: O(n_sites) sequential MLE fits.  Use only for small datasets
    (< 10 000 sites); for WGBS scale use method='mom'.
    """
    try:
        from statsmodels.discrete.discrete_model import NegativeBinomial  # noqa
        # BetaBinomialModel lives in statsmodels.discrete.count_model (sm ≥ 0.14)
        from statsmodels.discrete.count_model import BetaBinomialModel
    except ImportError as exc:
        raise ImportError(
            "statsmodels ≥ 0.14 is required for the 'statsmodels' path "
            "(BetaBinomialModel). Upgrade with: pip install --upgrade statsmodels"
        ) from exc

    import pandas as pd

    pvals     = np.full(n_sites, np.nan, dtype=np.float64)
    meth_diff = np.full(n_sites, np.nan, dtype=np.float32)

    groups = np.zeros(meth_counts.shape[1], dtype=np.int32)
    groups[case_mask] = 1

    for i in range(n_sites):
        mc  = meth_counts[i]
        tot = total_counts[i]
        if tot.sum() == 0:
            continue
        try:
            endog = np.column_stack([mc, tot - mc]).astype(np.float64)
            exog  = np.column_stack([
                np.ones(len(groups), dtype=np.float64),
                groups.astype(np.float64),
            ])
            model  = BetaBinomialModel(endog, exog)
            result = model.fit(disp=False, method="bfgs")
            # p-value for the group coefficient (index 1)
            pvals[i] = float(result.pvalues[1])
            # effect size: difference in predicted mean proportions
            mu_case = float(result.predict(
                exog=np.array([[1.0, 1.0]])
            )[0])
            mu_ctrl = float(result.predict(
                exog=np.array([[1.0, 0.0]])
            )[0])
            meth_diff[i] = np.float32(mu_case - mu_ctrl)
        except Exception as exc:
            logger.debug("statsmodels BetaBinomial failed at site %d: %s", i, exc)

    return pvals, meth_diff


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
    """Return (pos, strand) rows present in every sample for one chromosome."""
    intersect: Optional[pl.DataFrame] = None

    for sample in samples:
        part_file = (
            methylstore_path
            / f"sample={sample}"
            / f"chrom={chrom}"
            / "part-0.parquet"
        )
        if not part_file.exists():
            logger.debug(
                "  Sample '%s' missing %s; "
                "chromosome excluded from intersection",
                sample, chrom,
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
    """Return (pos, strand) rows seen in at least one sample."""
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
    canonical_pos: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Load N_meth and coverage for ONE sample / ONE chromosome.

    Left-joins to canonical_pos so arrays are aligned to the same site order.
    Missing sites are filled with 0.
    """
    part_file = (
        methylstore_path
        / f"sample={sample}"
        / f"chrom={chrom}"
        / "part-0.parquet"
    )
    n_sites = len(canonical_pos)

    if not part_file.exists():
        logger.debug("  Missing Parquet: %s", part_file)
        return (
            np.zeros(n_sites, dtype=np.int32),
            np.zeros(n_sites, dtype=np.int32),
        )

    df      = pl.read_parquet(str(part_file), columns=["pos", "N_meth", "coverage"])
    aligned = canonical_pos.join(df, on="pos", how="left").fill_null(0)

    return (
        aligned["N_meth"].to_numpy().astype(np.int32),
        aligned["coverage"].to_numpy().astype(np.int32),
    )


def _process_one_chromosome(
    methylstore_path: Path,
    chrom: str,
    canonical_df: pl.DataFrame,
    samples_case: list[str],
    samples_control: list[str],
    test: str,
) -> pl.DataFrame:
    """Run DMC for one chromosome, loading one sample at a time.

    For test="fisher": uses pooled count sums → fisher_exact_vectorized.
    For test="beta_binomial": passes per-replicate beta matrix to
    beta_binomial_test (method="mom").

    The pivot that caused OOM in the previous implementation is gone for the
    Fisher path; for beta_binomial, per-replicate beta arrays (float32) are
    stacked columnwise — memory is proportional to n_replicates.
    """
    n_sites = len(canonical_df)
    if n_sites == 0:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    canonical_pos = canonical_df.select("pos")

    # Running sums (int64 to avoid overflow)
    meth_case_sum = np.zeros(n_sites, dtype=np.int64)
    cov_case_sum  = np.zeros(n_sites, dtype=np.int64)
    meth_ctrl_sum = np.zeros(n_sites, dtype=np.int64)
    cov_ctrl_sum  = np.zeros(n_sites, dtype=np.int64)

    # Per-replicate arrays (used for both BIO-3 beta averaging and
    # beta_binomial test)
    beta_case_cols: list[np.ndarray] = []
    beta_ctrl_cols: list[np.ndarray] = []
    # For beta_binomial: also keep integer count arrays
    meth_case_reps: list[np.ndarray] = []
    cov_case_reps:  list[np.ndarray] = []
    meth_ctrl_reps: list[np.ndarray] = []
    cov_ctrl_reps:  list[np.ndarray] = []

    # --- Case samples ---
    for sample in samples_case:
        meth, cov = _load_sample_chrom(methylstore_path, chrom, sample, canonical_pos)
        meth_case_sum += meth
        cov_case_sum  += cov
        with np.errstate(invalid="ignore", divide="ignore"):
            beta = np.where(cov > 0, meth.astype(np.float32) / cov, np.nan)
        beta_case_cols.append(beta.astype(np.float32))
        if test == "beta_binomial":
            meth_case_reps.append(meth)
            cov_case_reps.append(cov)
        del meth, cov, beta

    # --- Control samples ---
    for sample in samples_control:
        meth, cov = _load_sample_chrom(methylstore_path, chrom, sample, canonical_pos)
        meth_ctrl_sum += meth
        cov_ctrl_sum  += cov
        with np.errstate(invalid="ignore", divide="ignore"):
            beta = np.where(cov > 0, meth.astype(np.float32) / cov, np.nan)
        beta_ctrl_cols.append(beta.astype(np.float32))
        if test == "beta_binomial":
            meth_ctrl_reps.append(meth)
            cov_ctrl_reps.append(cov)
        del meth, cov, beta

    # --- Statistical test ---
    if test == "fisher":
        unmeth_case_sum = cov_case_sum - meth_case_sum
        unmeth_ctrl_sum = cov_ctrl_sum - meth_ctrl_sum

        pvals, log2_ors = fisher_exact_vectorized(
            meth_case_sum, unmeth_case_sum,
            meth_ctrl_sum, unmeth_ctrl_sum,
        )

    elif test == "beta_binomial":
        # Stack count arrays into (n_sites, n_replicates) matrices
        meth_mat  = np.column_stack(meth_case_reps + meth_ctrl_reps).astype(np.float64)
        total_mat = np.column_stack(cov_case_reps  + cov_ctrl_reps).astype(np.float64)
        labels    = np.array(
            [1] * len(meth_case_reps) + [0] * len(meth_ctrl_reps),
            dtype=np.int32,
        )
        pvals, _ = beta_binomial_test(meth_mat, total_mat, labels, method="mom")
        # Use a dummy log2_or array (not meaningful for t-test)
        log2_ors = np.full(n_sites, np.nan, dtype=np.float64)

        del meth_mat, total_mat, labels
        del meth_case_reps, cov_case_reps, meth_ctrl_reps, cov_ctrl_reps

    else:
        raise NotImplementedError(
            f"Test '{test}' not implemented. Choose 'fisher' or 'beta_binomial'."
        )

    del meth_case_sum, meth_ctrl_sum, cov_case_sum, cov_ctrl_sum

    # --- BIO-3: equal-weight beta averaging (used regardless of test) ---
    beta_case_mat = np.stack(beta_case_cols, axis=1)
    beta_ctrl_mat = np.stack(beta_ctrl_cols, axis=1)
    del beta_case_cols, beta_ctrl_cols

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
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
    test : {"fisher", "beta_binomial"}
        Statistical test to apply.  "fisher" is recommended for < 6 replicates
        per group; "beta_binomial" for ≥ 6 replicates.
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
    store       = Path(methylstore_path)
    all_samples = samples_case + samples_control

    # Advisory: log recommended test based on replicate counts
    min_group = min(len(samples_case), len(samples_control))
    for rng, rec in _TEST_RECOMMENDATIONS.items():
        if min_group in rng:
            logger.info(
                "N replicates / group = %d; recommended test: %s", min_group, rec
            )
            break

    if chromosomes is None:
        chromosomes = _detect_chromosomes(store)
        logger.info("Auto-detected %d chromosomes", len(chromosomes))

    logger.info(
        "DMC: %d case / %d control, test=%s, unite=%s",
        len(samples_case), len(samples_control), test, unite,
    )

    with tempfile.TemporaryDirectory(prefix="epykit_dmc_") as tmpdir:
        tmp     = Path(tmpdir)
        written: list[Path] = []

        for i, chrom in enumerate(chromosomes):
            logger.info("[%d/%d] %s", i + 1, len(chromosomes), chrom)

            canonical_df = (
                _intersect_chrom(store, chrom, all_samples)
                if unite
                else _union_chrom(store, chrom, all_samples)
            )

            if len(canonical_df) == 0:
                logger.warning("  No sites for %s; skipping", chrom)
                continue

            logger.info("  %s sites to test", f"{len(canonical_df):,}")

            chrom_result = _process_one_chromosome(
                store, chrom, canonical_df,
                samples_case, samples_control, test,
            )
            del canonical_df

            if len(chrom_result) == 0:
                logger.warning("  No results for %s", chrom)
                continue

            tmp_file = tmp / f"{chrom}.parquet"
            chrom_result.write_parquet(str(tmp_file))
            written.append(tmp_file)
            logger.info("  %s sites → staged to disk", f"{len(chrom_result):,}")

            del chrom_result
            gc.collect()

        if not written:
            logger.warning("No results generated")
            return pl.DataFrame(schema=_EMPTY_SCHEMA)

        logger.info("Assembling results from %d chromosome file(s)...", len(written))
        combined = pl.concat([pl.read_parquet(str(f)) for f in written])
        logger.info("Total DMC sites: %s", f"{len(combined):,}")
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
        store       = Path(tmpdir) / "store"
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

    qvals  = np.where(nan_mask, np.nan,  qvals)
    reject = np.where(nan_mask, False,   reject)

    return dmc_results.with_columns([
        pl.Series("qvalue", qvals),
        pl.Series("reject", reject),
    ])
