"""Differential methylation calling (DMC) on partitioned Parquet stores.

Memory model
------------
The previous implementation kept per-replicate beta arrays and, for
beta_binomial, per-replicate meth/coverage matrices stacked into an
(n_sites × n_replicates) array before running the test.  For chr1 with
42 M CpGs (outer-join / unite=False) and 6 samples that was ~2 GB for
the float64 count matrix alone, causing OOM.

This version replaces all per-replicate accumulation with Welford's online
algorithm.  Peak memory per chromosome is now strictly O(n_sites):

    Fisher:         4 int64 running sums (meth/cov × case/ctrl)
    beta_binomial:  6 arrays per group — float64 mean, float64 M2,
                    int32 n_valid — derived from Welford updates

This scaling is independent of sample count, so the same code handles
20+ samples on 40 M+ sites without additional memory pressure.

Statistical tests
-----------------
  fisher  (default) — Fisher exact test via hypergeometric tail.
  beta_binomial     — Welch t-test on per-replicate beta values with
                      Welch–Satterthwaite DOF (MOM path, fully vectorised).
                      Now derived from Welford statistics rather than a
                      materialised beta matrix.

Biological fixes:
  BIO-3: equal-weight per-replicate beta averaging via Welford mean
         (mathematically identical to nanmean, but O(n_sites) memory).
  BIO-4: missing sites filled with 0 counts; excluded from beta mean
         via the n_valid counter in Welford accumulators.
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

_TEST_RECOMMENDATIONS = {
    range(1, 3):   "fisher",
    range(3, 6):   "fisher (report effect size; consider beta_binomial at ≥6)",
    range(6, 999): "beta_binomial",
}


# ---------------------------------------------------------------------------
# Core statistical tests (public, used by unit tests)
# ---------------------------------------------------------------------------

def fisher_exact_vectorized(
    meth_a: np.ndarray,
    unmeth_a: np.ndarray,
    meth_b: np.ndarray,
    unmeth_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised Fisher exact test via hypergeometric tail approximation."""
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

    Public entry point kept for unit-test compatibility.  Production code
    inside _process_one_chromosome now uses _beta_binom_mom_from_welford
    directly to avoid materialising the full beta matrix.

    method="mom"         — Welch t-test on per-replicate betas (fast, default)
    method="statsmodels" — per-site BetaBinomialModel MLE (slow, exact)
    """
    meth_counts  = np.asarray(meth_counts,  dtype=np.float64)
    total_counts = np.asarray(total_counts, dtype=np.float64)
    group_labels = np.asarray(group_labels, dtype=np.int32)

    if meth_counts.ndim == 1:
        meth_counts  = meth_counts[:, np.newaxis]
        total_counts = total_counts[:, np.newaxis]

    n_sites   = meth_counts.shape[0]
    case_mask = group_labels == 1
    ctrl_mask = group_labels == 0
    n_case    = int(case_mask.sum())
    n_ctrl    = int(ctrl_mask.sum())

    if n_case < 2 or n_ctrl < 2:
        warnings.warn(
            f"beta_binomial_test requires ≥2 replicates per group; "
            f"got case={n_case}, control={n_ctrl}.",
            UserWarning,
            stacklevel=2,
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(total_counts > 0, meth_counts / total_counts, np.nan)

    beta_case = beta[:, case_mask]
    beta_ctrl = beta[:, ctrl_mask]

    if method == "mom":
        return _beta_binom_mom(beta_case, beta_ctrl, n_case, n_ctrl, n_sites)
    elif method == "statsmodels":
        return _beta_binom_statsmodels(
            meth_counts, total_counts, case_mask, ctrl_mask, n_sites
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'mom' or 'statsmodels'.")


def _beta_binom_mom(
    beta_case: np.ndarray,
    beta_ctrl: np.ndarray,
    n_case: int,
    n_ctrl: int,
    n_sites: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised Welch t-test on per-replicate beta values (matrix form).

    Used by the public beta_binomial_test() entry point.  The production
    path inside _process_one_chromosome uses _beta_binom_mom_from_welford
    instead, which avoids building the matrix in the first place.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu_case = np.nanmean(beta_case, axis=1)
        mu_ctrl = np.nanmean(beta_ctrl, axis=1)
        var_mean_case = np.nanvar(beta_case, axis=1, ddof=1) / max(n_case, 1)
        var_mean_ctrl = np.nanvar(beta_ctrl, axis=1, ddof=1) / max(n_ctrl, 1)

    se = np.sqrt(var_mean_case + var_mean_ctrl)

    with np.errstate(invalid="ignore", divide="ignore"):
        t_stat = np.where(se > 0, (mu_case - mu_ctrl) / se, np.nan)

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

    degenerate = np.isnan(mu_case) | np.isnan(mu_ctrl) | np.isnan(t_stat)
    pvals[degenerate]     = np.nan
    meth_diff[degenerate] = np.nan

    return pvals, meth_diff


def _beta_binom_statsmodels(
    meth_counts, total_counts, case_mask, ctrl_mask, n_sites
):
    try:
        from statsmodels.discrete.count_model import BetaBinomialModel
    except ImportError as exc:
        raise ImportError(
            "statsmodels ≥ 0.14 is required for the 'statsmodels' path."
        ) from exc

    import pandas as pd

    pvals     = np.full(n_sites, np.nan, dtype=np.float64)
    meth_diff = np.full(n_sites, np.nan, dtype=np.float32)
    groups    = np.zeros(meth_counts.shape[1], dtype=np.int32)
    groups[case_mask] = 1

    for i in range(n_sites):
        mc  = meth_counts[i]
        tot = total_counts[i]
        if tot.sum() == 0:
            continue
        try:
            endog  = np.column_stack([mc, tot - mc]).astype(np.float64)
            exog   = np.column_stack([
                np.ones(len(groups), dtype=np.float64),
                groups.astype(np.float64),
            ])
            result = BetaBinomialModel(endog, exog).fit(disp=False, method="bfgs")
            pvals[i]     = float(result.pvalues[1])
            mu_case      = float(result.predict(exog=np.array([[1.0, 1.0]]))[0])
            mu_ctrl      = float(result.predict(exog=np.array([[1.0, 0.0]]))[0])
            meth_diff[i] = np.float32(mu_case - mu_ctrl)
        except Exception as exc:
            logger.debug("statsmodels BetaBinomial failed at site %d: %s", i, exc)

    return pvals, meth_diff


# ---------------------------------------------------------------------------
# CMH test — O(n_sites) memory, statistically correct for replicates
# ---------------------------------------------------------------------------

def _cmh_init(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Allocate CMH accumulators. Memory: ~32 bytes × n_sites total."""
    return (
        np.zeros(n, dtype=np.float64),  # Σ(a - E): obs minus expected
        np.zeros(n, dtype=np.float64),  # Σ V:      variance sum
        np.zeros(n, dtype=np.float64),  # Σ(ad/n):  MH OR numerator
        np.zeros(n, dtype=np.float64),  # Σ(bc/n):  MH OR denominator
    )


def _cmh_update(
    ome: np.ndarray,
    var_sum: np.ndarray,
    or_num: np.ndarray,
    or_den: np.ndarray,
    meth_case: np.ndarray,
    cov_case: np.ndarray,
    meth_ctrl: np.ndarray,
    cov_ctrl: np.ndarray,
) -> None:
    """In-place CMH accumulation from one case/control sample pair.

    Sites where either sample has zero coverage contribute V=0 and
    therefore do not influence the statistic — this correctly handles
    union-mode sites with partial coverage without any special casing.
    """
    a = meth_case.astype(np.float64)
    b = (cov_case - meth_case).astype(np.float64)  # unmeth case
    c = meth_ctrl.astype(np.float64)
    d = (cov_ctrl - meth_ctrl).astype(np.float64)  # unmeth ctrl
    n = a + b + c + d

    # Sites need n > 1 for a non-degenerate variance term
    valid = n > 1

    row1 = a + b  # case coverage
    row2 = c + d  # ctrl coverage
    col1 = a + c  # total methylated
    col2 = b + d  # total unmethylated

    # Use safe denominator to avoid divide-by-zero warnings
    n_safe = np.where(n > 0, n, 1.0)
    E = np.where(valid, row1 * col1 / n_safe, 0.0)
    # Safe denominators to avoid divide-by-zero warnings
    n_sq_safe = np.where(n > 1, n * n * (n - 1.0), 1.0)
    V = np.where(
        valid,
        row1 * row2 * col1 * col2 / n_sq_safe,
        0.0,
    )

    ome[valid] += (a - E)[valid]
    var_sum[valid] += V[valid]

    # Mantel-Haenszel common odds ratio terms (using n_safe from above)
    or_num += np.where(valid, a * d / n_safe, 0.0)
    or_den += np.where(valid, b * c / n_safe, 0.0)


def _cmh_finalize(
    ome: np.ndarray,
    var_sum: np.ndarray,
    or_num: np.ndarray,
    or_den: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute CMH p-value and MH log2 OR from accumulated sums."""
    # Use safe denominator to avoid divide-by-zero warnings
    var_safe = np.where(var_sum > 0, var_sum, 1.0)
    cmh_stat = np.where(var_sum > 0, ome ** 2 / var_safe, np.nan)
    pvals = np.where(
        ~np.isnan(cmh_stat),
        sp_stats.chi2.sf(cmh_stat, df=1),
        np.nan,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        mh_or = np.where(or_den > 0, or_num / or_den, np.nan)
        log2_mh_or = np.where(mh_or > 0, np.log2(mh_or), np.nan)

    return pvals, log2_mh_or


# ---------------------------------------------------------------------------
# Welford online statistics — O(n_sites) memory regardless of n_samples
# ---------------------------------------------------------------------------

def _welford_init(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Allocate Welford accumulators for n sites.

    Returns (mean, M2, n_valid).  Memory: ~20 bytes × n (float64 + int32).
    """
    return (
        np.zeros(n, dtype=np.float64),  # running mean
        np.zeros(n, dtype=np.float64),  # running sum of squared deviations
        np.zeros(n, dtype=np.int32),    # non-NaN replicate count per site
    )


def _welford_update(
    mean: np.ndarray,
    M2: np.ndarray,
    n_valid: np.ndarray,
    meth: np.ndarray,
    cov: np.ndarray,
) -> None:
    """In-place Welford update from one sample's integer meth/coverage arrays.

    Sites with zero coverage are treated as missing and skipped, so
    n_valid[i] counts only samples that actually covered site i.
    This handles BIO-4 (union sites with partial coverage) correctly.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(cov > 0, meth.astype(np.float64) / cov, np.nan)
    valid = ~np.isnan(beta)
    if not np.any(valid):
        return
    n_valid[valid] += 1
    delta          = beta[valid] - mean[valid]
    mean[valid]   += delta / n_valid[valid]
    delta2         = beta[valid] - mean[valid]
    M2[valid]     += delta * delta2


def _welford_var_mean(M2: np.ndarray, n_valid: np.ndarray) -> np.ndarray:
    """Bessel-corrected variance of the group mean: s²/n (per site).

    Sites with fewer than 2 valid replicates get NaN.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.where(n_valid > 1, M2 / (n_valid - 1), np.nan)
        return np.where(n_valid > 0, var / n_valid, np.nan)


def _beta_binom_mom_from_welford(
    mean_case: np.ndarray,
    M2_case: np.ndarray,
    n_valid_case: np.ndarray,
    mean_ctrl: np.ndarray,
    M2_ctrl: np.ndarray,
    n_valid_ctrl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch t-test derived from Welford accumulators.

    Mathematically equivalent to _beta_binom_mom but never builds the
    (n_sites × n_replicates) beta matrix.  Per-site valid counts are used
    for both variance estimation and Welch–Satterthwaite DOF, which correctly
    handles sites where some replicates have no coverage (union / outer-join
    mode).
    """
    n_sites = len(mean_case)
    vm_case = _welford_var_mean(M2_case, n_valid_case)
    vm_ctrl = _welford_var_mean(M2_ctrl, n_valid_ctrl)

    se = np.sqrt(vm_case + vm_ctrl)

    with np.errstate(invalid="ignore", divide="ignore"):
        t_stat = np.where(se > 0, (mean_case - mean_ctrl) / se, np.nan)

    # Welch–Satterthwaite degrees of freedom (per-site n_valid)
    dof_num = (vm_case + vm_ctrl) ** 2
    dof_den = (
        np.where(n_valid_case > 1, vm_case ** 2 / (n_valid_case - 1), 0.0)
        + np.where(n_valid_ctrl > 1, vm_ctrl ** 2 / (n_valid_ctrl - 1), 0.0)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        dof = np.where(dof_den > 0, dof_num / dof_den, 1.0)
        dof = np.maximum(dof, 1.0)

    pvals = 2.0 * sp_stats.t.sf(np.abs(t_stat), df=dof)

    degenerate = (
        np.isnan(mean_case) | np.isnan(mean_ctrl) | np.isnan(t_stat)
        | (n_valid_case == 0) | (n_valid_ctrl == 0)
    )
    pvals[degenerate] = np.nan

    # Compute log2 odds ratio from Welford means
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_ors = np.log2(
            (mean_case / np.maximum(1 - mean_case, 1e-9)) /
            np.maximum(mean_ctrl / np.maximum(1 - mean_ctrl, 1e-9), 1e-9)
        )
    log2_ors[degenerate] = np.nan
    
    return pvals, log2_ors


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

    FIX-5: The previous implementation joined on ["pos", "strand"].  Samples
    without a reference FASTA receive strand="*" while samples converted with
    a FASTA receive "+"/"-".  A mixed cohort produced an empty intersection
    with no warning.  We now join on "pos" only and resolve the strand column
    by taking the first non-"*" value seen across samples (falling back to "*"
    when all samples lack strand information).
    """
    intersect: Optional[pl.DataFrame] = None

    for sample in samples:
        part_file = (
            methylstore_path / f"sample={sample}" / f"chrom={chrom}" / "part-0.parquet"
        )
        if not part_file.exists():
            logger.debug(
                "  Sample '%s' missing %s; chromosome excluded from intersection",
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
            # Join on pos only to avoid strand-value mismatches between
            # samples converted with and without a reference FASTA.
            # Keep the strand column from the left frame; if it is "*",
            # prefer the right frame's value (may carry real strand info).
            intersect = (
                intersect
                .join(sites.rename({"strand": "_strand_r"}), on="pos", how="inner")
                .with_columns(
                    pl.when(pl.col("strand") == "*")
                    .then(pl.col("_strand_r"))
                    .otherwise(pl.col("strand"))
                    .alias("strand")
                )
                .drop("_strand_r")
            )

        if len(intersect) == 0:
            logger.warning(
                "  Intersection is empty after adding sample '%s' on %s. "
                "Check strand consistency across samples.", sample, chrom,
            )
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
            methylstore_path / f"sample={sample}" / f"chrom={chrom}" / "part-0.parquet"
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
    Missing sites are filled with 0 (BIO-4).
    """
    part_file = (
        methylstore_path / f"sample={sample}" / f"chrom={chrom}" / "part-0.parquet"
    )
    n_sites = len(canonical_pos)

    if not part_file.exists():
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

    Memory design
    -------------
    Peak memory is O(n_sites) regardless of sample count:

    Fisher:         4 int64 running sums (meth/cov × case/ctrl)
    beta_binomial:  6 arrays per group — float64 mean, float64 M2,
                    int32 n_valid — via Welford online algorithm.

    This replaces the previous approach that stacked per-replicate arrays
    into (n_sites × n_replicates) matrices, which caused OOM at 42 M sites.
    The Welford mean also provides the BIO-3 equal-weight mean beta directly,
    so no separate beta accumulation is needed for either test.
    """
    n_sites = len(canonical_df)
    if n_sites == 0:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    canonical_pos = canonical_df.select("pos")

    # Integer running sums — used by Fisher test and for sanity logging.
    meth_case_sum = np.zeros(n_sites, dtype=np.int64)
    cov_case_sum  = np.zeros(n_sites, dtype=np.int64)
    meth_ctrl_sum = np.zeros(n_sites, dtype=np.int64)
    cov_ctrl_sum  = np.zeros(n_sites, dtype=np.int64)

    # Welford accumulators: O(n_sites) memory, independent of n_samples.
    # mean_* provides BIO-3 equal-weight mean beta (equivalent to nanmean).
    # M2_* and n_valid_* provide per-site variance for the MOM t-test.
    mean_case, M2_case, n_valid_case = _welford_init(n_sites)
    mean_ctrl, M2_ctrl, n_valid_ctrl = _welford_init(n_sites)

    # --- Statistical test ---
    if test in ("fisher", "cmh"):
        # FIX-CMH: The previous implementation formed one stratum per
        # (case_i, ctrl_j) pair, so each observation was counted
        # n_other_group times.  The CMH chi-squared statistic was inflated
        # by ~n_case × n_control, producing grossly anti-conservative p-values.
        #
        # Correct approach: pool all control reads into a single count vector,
        # then contribute one stratum per case sample (case_i vs pooled ctrl).
        # With n_case=1, n_ctrl=1 this degenerates to Fisher exact.
        # With replicates it gives a properly weighted CMH result.
        ome, var_sum, or_num, or_den = _cmh_init(n_sites)

        # Pass 1: accumulate pooled control sums and Welford mean beta
        meth_ctrl_pool = np.zeros(n_sites, dtype=np.int64)
        cov_ctrl_pool  = np.zeros(n_sites, dtype=np.int64)
        for ctrl in samples_control:
            meth, cov = _load_sample_chrom(methylstore_path, chrom, ctrl, canonical_pos)
            meth_ctrl_pool += meth
            cov_ctrl_pool  += cov
            _welford_update(mean_ctrl, M2_ctrl, n_valid_ctrl, meth, cov)
            del meth, cov

        # Pass 2: one CMH stratum per case sample vs the pooled control
        for case in samples_case:
            meth_i, cov_i = _load_sample_chrom(methylstore_path, chrom, case, canonical_pos)
            _welford_update(mean_case, M2_case, n_valid_case, meth_i, cov_i)
            _cmh_update(ome, var_sum, or_num, or_den,
                        meth_i, cov_i, meth_ctrl_pool, cov_ctrl_pool)
            del meth_i, cov_i

        del meth_ctrl_pool, cov_ctrl_pool
        pvals, log2_ors = _cmh_finalize(ome, var_sum, or_num, or_den)
        del ome, var_sum, or_num, or_den

    elif test == "beta_binomial":
        # Load all samples for Welford accumulators
        for sample in samples_case:
            meth, cov = _load_sample_chrom(methylstore_path, chrom, sample, canonical_pos)
            _welford_update(mean_case, M2_case, n_valid_case, meth, cov)
            del meth, cov

        for sample in samples_control:
            meth, cov = _load_sample_chrom(methylstore_path, chrom, sample, canonical_pos)
            _welford_update(mean_ctrl, M2_ctrl, n_valid_ctrl, meth, cov)
            del meth, cov

        # Welford path: no (n_sites × n_replicates) matrix ever built.
        pvals, log2_ors = _beta_binom_mom_from_welford(
            mean_case, M2_case, n_valid_case,
            mean_ctrl, M2_ctrl, n_valid_ctrl,
        )

    else:
        raise NotImplementedError(
            f"Test '{test}' not implemented. Choose 'fisher' or 'beta_binomial'."
        )

    del meth_case_sum, meth_ctrl_sum, cov_case_sum, cov_ctrl_sum

    # --- BIO-3: equal-weight per-replicate mean beta ---
    # Welford mean IS the equal-weight nanmean — no extra storage needed.
    mean_beta_case = mean_case.astype(np.float32)
    mean_beta_ctrl = mean_ctrl.astype(np.float32)
    mean_beta_case[n_valid_case == 0] = np.nan
    mean_beta_ctrl[n_valid_ctrl == 0] = np.nan
    meth_diff = (mean_beta_case - mean_beta_ctrl).astype(np.float32)
    del mean_case, M2_case, n_valid_case, mean_ctrl, M2_ctrl, n_valid_ctrl

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

    Parameters
    ----------
    methylstore_path : str
        Path to filtered partitioned Parquet methylstore.
    samples_case, samples_control : list[str]
        Sample identifiers for case and control groups.
    test : {"fisher", "beta_binomial"}
        Statistical test.  "fisher" recommended for < 6 replicates / group;
        "beta_binomial" for ≥ 6.
    chromosomes : list[str], optional
        Chromosomes to process. Auto-detected when None.
    unite : bool
        If True (default), test only CpG sites covered in every sample
        (intersection / inner join).
        If False, test all sites covered in at least one sample
        (union / outer join).  The Welford accumulator correctly handles
        the resulting missing-data pattern.

    Returns
    -------
    pl.DataFrame
        Columns: chrom, pos, strand, n_case, n_control,
                 mean_beta_case, mean_beta_control,
                 pvalue, log2_odds_ratio, meth_diff
    """
    store       = Path(methylstore_path)
    all_samples = samples_case + samples_control

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
    """Legacy entry-point kept for unit-test compatibility."""
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
    """Apply genome-wide multiple testing correction (Benjamini-Hochberg default)."""
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