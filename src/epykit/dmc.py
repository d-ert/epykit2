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

    Fisher (pooled): 4 int64 running sums (meth/cov × case/ctrl)
    CMH (stratified): 4 float64 accumulators + case data cached at int32
    beta_binomial:   6 arrays per group — float64 mean, float64 M2,
                     int32 n_valid — derived from Welford updates

This scaling is independent of sample count (for non-CMH paths), so the
same code handles 20+ samples on 40 M+ sites without additional memory
pressure.

Statistical tests
-----------------
  score        — (RECOMMENDED DEFAULT) Quasi-binomial score test for
                 differential methylation, with a chromosome-level
                 overdispersion correction estimated from Pearson residuals
                 under the full model. The score statistic is computed on
                 per-group count sums (M_case, N_case, M_ctrl, N_ctrl) under
                 H0: π_case = π_ctrl, and the variance is inflated by the
                 global dispersion ϕ̂ to account for between-replicate
                 variability. This matches methylKit's
                 calculateDiffMeth(overdispersion='MN', test='Chisq') and is
                 the closest in-tree analogue to the DSS workflow without
                 empirical-Bayes per-site dispersion shrinkage. Streaming /
                 single-pass: no per-sample caching is needed.
  fisher       — Fisher exact on reads POOLED across replicates. Fast and
                 widely understood, but ignores between-replicate variance
                 and is anti-conservative at high WGBS coverage. Emits a
                 warning when called. Use for parity with single-rep tools
                 only.
  cmh          — Cochran-Mantel-Haenszel with one 2×2 stratum per
                 (case_i, ctrl_j) pair. Preserves between-sample variance
                 via the per-stratum variance term.
  logit_t      — Welch t-test on logit-transformed per-replicate beta values
                 derived from Welford accumulators. Variance-stabilising;
                 fallback when count-model assumptions are doubtful (e.g.
                 very low coverage).
  beta_binomial — Welch t-test on per-replicate beta values (untransformed).
                 Despite the name, this is NOT a beta-binomial GLM. Use
                 ``score`` for a real count-based test with overdispersion.

Biological fixes:
  BIO-3: equal-weight per-replicate beta averaging via Welford mean
         (mathematically identical to nanmean, but O(n_sites) memory).
  BIO-4: missing sites filled with 0 counts; excluded from beta mean
         via the n_valid counter in Welford accumulators.
  BIO-5 (this revision): the previous "cmh" / "fisher" path pooled all
         control reads into a single super-sample and ran one stratum
         per case sample against that pool. This is equivalent to Fisher
         exact on pooled reads (with inflated effective N), not a
         properly stratified CMH. The two paths are now distinct:
         "fisher" calls fisher_exact_vectorized on per-group read sums,
         and "cmh" implements true per-pair stratification.
  BIO-6 (this revision): log2_odds_ratio now uses a symmetric clamp on
         both group means in [epsilon, 1-epsilon], so the OR is bounded
         even when one group has β=1 or β=0 exactly. Previously the
         numerator mean/(1-mean) could diverge to ±inf.
  BIO-7 (this revision): optional per-site min_samples guard drops sites
         with fewer than `min_samples_case` / `min_samples_control` valid
         replicates after the test. Avoids singleton-observation tests
         in union (outer-join) mode.
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
    range(1, 3):   "fisher (single-rep only; effect size dominates)",
    range(3, 999): "lr (quasi-binomial likelihood-ratio with MN overdispersion)",
}

# Shared epsilon for boundary clipping in logit / log-OR computations.
_BETA_EPSILON: float = 1e-6


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


def _logit_transform(beta: np.ndarray) -> np.ndarray:
    """Transform beta values to logit scale.

    logit(β) = log(β / (1 - β))

    Handles boundary cases (β=0, β=1) by clipping to [ε, 1-ε].
    """
    beta_clipped = np.clip(beta, _BETA_EPSILON, 1 - _BETA_EPSILON)
    with np.errstate(divide="ignore", invalid="ignore"):
        logit_beta = np.log(beta_clipped / (1 - beta_clipped))
    return logit_beta


def _logit_variance_jacobian(beta: np.ndarray) -> np.ndarray:
    """Compute Jacobian for delta-method variance transformation.

    If Y = logit(X), then Var(Y) ≈ Var(X) × [dY/dX]²
    where dY/dX = 1 / [X(1-X)]
    """
    beta_clipped = np.clip(beta, _BETA_EPSILON, 1 - _BETA_EPSILON)
    with np.errstate(divide="ignore", invalid="ignore"):
        jacobian = 1.0 / (beta_clipped * (1 - beta_clipped))
    return jacobian


def _safe_log2_odds_ratio(
    mean_case: np.ndarray,
    mean_ctrl: np.ndarray,
) -> np.ndarray:
    """Symmetric log2 odds ratio with bounded clipping in both groups.

    BIO-6 fix: the previous formulation clipped only `(1 - mean_case)` and
    `(1 - mean_ctrl)` in the denominators of the inner ratios, so the
    numerators `mean_case` and `mean_ctrl` could remain at their raw values
    of 1.0, producing ratios of ``1 / ε`` that propagated to ``log2 ≈ 30``
    and, when the outer ratio compounded, ``±inf``.

    Symmetric clipping in [ε, 1-ε] for both means caps the OR at
    ``log2((1-ε)/ε)² ≈ 39.8`` regardless of which group is at the boundary,
    so finite-but-large values still signal extreme effects and ``inf``
    no longer pollutes the output column.
    """
    case_clip = np.clip(mean_case, _BETA_EPSILON, 1 - _BETA_EPSILON)
    ctrl_clip = np.clip(mean_ctrl, _BETA_EPSILON, 1 - _BETA_EPSILON)

    odds_case = case_clip / (1 - case_clip)
    odds_ctrl = ctrl_clip / (1 - ctrl_clip)

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(odds_case / odds_ctrl)


# ---------------------------------------------------------------------------
# Quasi-binomial score test with chromosome-level overdispersion correction
# (a.k.a. "Version 1" of the count-model DMC family).
#
# Model for site i with replicates j and group g(j) ∈ {case, ctrl}:
#
#     m_ij | n_ij  ~  Binomial(n_ij, π_g(j)_i)             (binomial mean)
#     Var(m_ij)   =  φ · n_ij · π_g(j)_i · (1 − π_g(j)_i)   (quasi-binomial)
#
# Test H0: π_case_i = π_ctrl_i, against a two-sided alternative.  Let
#     S0_g = Σ_j n_ij,   S1_g = Σ_j m_ij,   S2_g = Σ_j m_ij² / n_ij
# (group-wise running sums).  Then the group MLEs are π̂_g = S1_g / S0_g and
# the pooled MLE under H0 is π̂_pool = (S1_case + S1_ctrl) / (S0_case + S0_ctrl).
# The score for the group contrast and its null variance are
#     U      = S1_case − S0_case · π̂_pool
#     Var(U) = φ · (S0_case · S0_ctrl / (S0_case + S0_ctrl)) · π̂_pool · (1 − π̂_pool)
# so the test statistic is U² / Var(U), χ²₁ under H0.
#
# The dispersion φ is estimated from the full-model Pearson statistic:
#     X²_g(i) = Σ_j (m_ij − n_ij π̂_g_i)² / (n_ij π̂_g_i (1 − π̂_g_i))
#             = (S2_g − S1_g²/S0_g) / (π̂_g_i (1 − π̂_g_i))
# (the closed-form expansion lets us avoid materialising the n_sites × n_reps
# matrix). φ̂ = sum_i_g X²_g(i) / (n_obs − 2·n_sites_fit), clamped at 1.
#
# This is what methylKit does in calculateDiffMeth(overdispersion='MN',
# test='Chisq'). It is replicate-aware (variance scales with the *number of
# replicates* via φ̂, not with the number of pooled reads), it is a real
# count-based test (does not throw away the information that 5/10 carries
# less weight than 500/1000), and it is single-pass / O(n_sites) memory.
# ---------------------------------------------------------------------------

def _score_init(
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Allocate quasi-binomial score accumulators (per group).

    Returns four arrays of length ``n``:

    sum_n   : float64 — Σ_j n_ij        (group coverage sum per site)
    sum_m   : float64 — Σ_j m_ij        (group meth-count sum per site)
    sum_m2n : float64 — Σ_j m_ij²/n_ij  (used for closed-form Pearson)
    n_valid : int32   — # replicates with coverage > 0 at this site

    Memory per group: 28 bytes × n_sites (~280 MB at 10 M sites).
    """
    return (
        np.zeros(n, dtype=np.float64),
        np.zeros(n, dtype=np.float64),
        np.zeros(n, dtype=np.float64),
        np.zeros(n, dtype=np.int32),
    )


def _score_update(
    sum_n:   np.ndarray,
    sum_m:   np.ndarray,
    sum_m2n: np.ndarray,
    n_valid: np.ndarray,
    meth:    np.ndarray,
    cov:     np.ndarray,
) -> None:
    """Fold one sample's (meth, coverage) arrays into the accumulators."""
    cov_f  = cov.astype(np.float64,  copy=False)
    meth_f = meth.astype(np.float64, copy=False)
    valid  = cov > 0

    sum_n += cov_f
    sum_m += meth_f
    # m² / n; zero contribution where the sample has no coverage.
    with np.errstate(invalid="ignore", divide="ignore"):
        sum_m2n += np.where(valid, meth_f * meth_f / np.maximum(cov_f, 1.0), 0.0)
    n_valid += valid.astype(np.int32)


def _score_finalize(
    sn_case:  np.ndarray, sm_case:  np.ndarray, sm2n_case: np.ndarray, nv_case: np.ndarray,
    sn_ctrl:  np.ndarray, sm_ctrl:  np.ndarray, sm2n_ctrl: np.ndarray, nv_ctrl: np.ndarray,
    chrom_name:     str   = "?",
    min_dispersion: float = 1.0,
    min_disp_sites: int   = 100,
    dispersion:     str   = "site",
    shrink_pseudo_df: float = 4.0,
    statistic:      str   = "lr",
    reference:      str   = "methylkit",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute per-site score p-values with McCullagh-Nelder overdispersion.

    Parameters
    ----------
    sn_*, sm_*, sm2n_*, nv_* : np.ndarray
        Per-group accumulators returned by ``_score_init`` / ``_score_update``.
    chrom_name : str
        Used only for logging.
    min_dispersion : float
        Floor on φ̂.  Underdispersion (φ̂ < 1) usually reflects model
        misspecification rather than truly less-than-binomial variability;
        clamping at 1 is the conservative choice and matches methylKit's
        default.  Set < 1 to allow underdispersion.
    min_disp_sites : int
        If fewer than this many sites are usable for the chromosome-level
        dispersion estimate (e.g. tiny alt contigs), fall back to
        φ̂ = ``min_dispersion`` instead of computing an unstable estimate.
        Used by the ``"chrom"`` and ``"shrink"`` modes; the ``"site"`` mode
        is unaffected because each site provides its own φ̂_i.
    dispersion : {"site", "chrom", "shrink"}
        Strategy for the McCullagh-Nelder dispersion correction:

        ``"site"`` (default, methylKit ``overdispersion="MN"`` parity)
            Each site/tile gets its own φ̂_i computed from its 4-df Pearson
            residual sum:  φ̂_i = (X²_case_i + X²_ctrl_i) / max(nv_i − 2, 1),
            clamped at ``min_dispersion``. This is what R's
            ``glm(family=quasibinomial)`` does on a per-site fit and is the
            assumption baked into methylKit's reference pipeline. The
            estimator is noisy with the typical 4 df, but it correctly
            tracks region-specific dispersion (CpG-islands vs gene bodies
            have different between-replicate variance).
        ``"chrom"`` (previous default)
            Single chromosome-pooled φ̂ from all qualifying sites. Most
            powerful when between-replicate variance really is constant
            along the chromosome, but anti-conservative when it is not.
            Equivalent to fitting the quasi-binomial GLM as a single model
            with one shared dispersion. Use when you want strictly more
            power and you accept the modelling assumption.
        ``"shrink"`` (methylKit ``overdispersion="shrinkMN"`` parity)
            James-Stein-style shrinkage: φ̂_shrunk_i is a weighted average of
            φ̂_site_i (with weight = site df) and the chromosome-pooled
            φ̂_chrom (with weight = ``shrink_pseudo_df``, default 4).
            Trades a small bias for a large variance reduction on the
            per-site estimate; reproduces the published behaviour of
            methylKit's shrunk dispersion and DSS's empirical Bayes prior.
    shrink_pseudo_df : float
        Pseudo-df weight on φ̂_chrom in the ``"shrink"`` mode (default 4 ≈
        the typical real per-site df). Ignored otherwise.
    statistic : {"lr", "score"}
        Functional form of the test statistic. Both use the same per-group
        sufficient statistics (S0_g = Σ_j n_ij and S1_g = Σ_j m_ij) and the
        same dispersion correction, so the difference manifests only at
        small effective sample sizes (n=6 is small):

        ``"lr"`` (default, methylKit ``test='Chisq'`` parity)
            Quasi-binomial likelihood-ratio chi-square. Closed-form in
            S0_g and S1_g, so no per-tile GLM fit is required:
              LRT = 2 · Σ_g [ S1_g·log(p̂_g/p̂_pool)
                            + (S0_g − S1_g)·log((1 − p̂_g)/(1 − p̂_pool)) ]
            divided by the dispersion φ̂_i. Closer to nominal coverage near
            the boundaries (π̂ near 0 or 1) — exactly where DMR tiles tend
            to live. This is what methylKit reports.

        ``"score"`` (previous default; slightly more powerful)
            Pearson score statistic U²/V_pool with quasi-binomial inflation.
            Asymptotically equivalent to the LR test but mildly
            anti-conservative at the boundaries. Kept as an option for
            users who want the small extra power.
    reference : {"methylkit", "chi2", "F"}
        Reference distribution used to convert the test statistic to a
        p-value.

        ``"methylkit"`` (default, exact methylKit-parity)
            Per-site adaptive: F(1, df_residual_i) where the per-site
            dispersion φ̂_i > 1 (overdispersion detected), χ²(1) elsewhere.
            This is what methylKit's ``calculateDiffMeth`` does with its
            default ``test`` argument (the first option of
            ``c("F","Chisq",...)`` resolves to ``"F"``, which logReg then
            switches to ``"Chisq"`` only where ``phi <= 1``). See line
            273 of methylKit's ``R/diffMeth.R``. This is the only choice
            that gives parity at BOTH the per-CpG (DMC) and per-tile
            (DMR) levels.
        ``"chi2"``
            Always reference to χ²(1). Matches methylKit only at sites
            where φ̂ was clamped to 1. Over-liberal at tiles with real
            overdispersion (typical DMR setting).
        ``"F"``
            Always reference to F(1, df_residual_i). Matches methylKit
            only at sites where φ̂ > 1. Wildly conservative at sites with
            clamped φ̂ = 1 — will reject zero genome-wide CpGs on typical
            n=3+3 WGBS.

    Returns
    -------
    pvals : float64 array, NaN at degenerate sites
    log2_or : float64 array, NaN at degenerate sites
    pi_case, pi_ctrl : float64 arrays of coverage-weighted group methylation
        (= group MLE proportion under the full model). NaN where the
        corresponding group has zero coverage at the site.
    phi_hat : float
        Chromosome-pooled dispersion estimate, returned for logging /
        downstream introspection regardless of which mode was used.
    """
    if dispersion not in {"site", "chrom", "shrink"}:
        raise ValueError(
            f"dispersion must be 'site', 'chrom', or 'shrink'; got {dispersion!r}"
        )
    if statistic not in {"lr", "score"}:
        raise ValueError(
            f"statistic must be 'lr' or 'score'; got {statistic!r}"
        )
    if reference not in {"methylkit", "F", "chi2"}:
        raise ValueError(
            f"reference must be 'methylkit', 'F', or 'chi2'; got {reference!r}"
        )
    eps = _BETA_EPSILON

    # --- Group MLE proportions under the full (unrestricted) model ---
    with np.errstate(invalid="ignore", divide="ignore"):
        pi_case = np.where(sn_case > 0, sm_case / sn_case, np.nan)
        pi_ctrl = np.where(sn_ctrl > 0, sm_ctrl / sn_ctrl, np.nan)

    # --- Pearson chi-sq contribution from each site & group, full model ---
    # Numerator (closed form): N_g · S2_g − S1_g² , scaled by 1/S0_g²/π̂(1-π̂).
    # Compact form:  contrib = (S2 − S1²/S0) / (π̂ · (1 − π̂))
    # where π̂(1-π̂) = S1·(S0 − S1) / S0².  Sites with S1 ∈ {0, S0} have
    # variance 0 and contribute nothing to the dispersion estimate.
    with np.errstate(invalid="ignore", divide="ignore"):
        # numerator term Σⱼ(m − nπ̂)²/n   =  S2 − S1²/S0
        num_case = sm2n_case - np.where(sn_case > 0, sm_case ** 2 / sn_case, 0.0)
        num_ctrl = sm2n_ctrl - np.where(sn_ctrl > 0, sm_ctrl ** 2 / sn_ctrl, 0.0)

        den_case = np.where(
            sn_case > 0,
            sm_case * (sn_case - sm_case) / (sn_case ** 2),
            0.0,
        )
        den_ctrl = np.where(
            sn_ctrl > 0,
            sm_ctrl * (sn_ctrl - sm_ctrl) / (sn_ctrl ** 2),
            0.0,
        )

        chi_case = np.where(den_case > 0, num_case / den_case, 0.0)
        chi_ctrl = np.where(den_ctrl > 0, num_ctrl / den_ctrl, 0.0)

    sites_both    = (sn_case > 0) & (sn_ctrl > 0) & (nv_case > 0) & (nv_ctrl > 0)
    sites_dispers = sites_both & (den_case > 0) & (den_ctrl > 0)

    # --- Chromosome-pooled φ̂ (always computed; used directly in "chrom"
    #     mode, used as the shrinkage anchor in "shrink" mode, logged
    #     otherwise) -----------------------------------------------------
    n_disp = int(sites_dispers.sum())
    if n_disp < min_disp_sites:
        if dispersion == "chrom":
            logger.warning(
                "%s: only %d sites usable for dispersion estimation; "
                "falling back to φ̂ = %.2f (no overdispersion correction).",
                chrom_name, n_disp, min_dispersion,
            )
        phi_hat = float(min_dispersion)
        phi_raw = float(min_dispersion)
    else:
        n_obs = int(nv_case[sites_dispers].sum() + nv_ctrl[sites_dispers].sum())
        df    = max(n_obs - 2 * n_disp, 1)

        pearson_sum = float(
            chi_case[sites_dispers].sum() + chi_ctrl[sites_dispers].sum()
        )
        phi_raw = pearson_sum / df
        phi_hat = float(max(min_dispersion, phi_raw))

        logger.info(
            "%s: chrom-pooled φ̂ = %.3f (raw %.3f, %s sites, %s obs, df=%s); "
            "applying dispersion='%s'",
            chrom_name, phi_hat, phi_raw,
            f"{n_disp:,}", f"{n_obs:,}", f"{df:,}", dispersion,
        )

    # --- Per-site Pearson dispersion φ̂_i (only used when needed) ---------
    if dispersion in ("site", "shrink"):
        # df_i = (replicates_case + replicates_ctrl) − 2 fitted proportions.
        # At a typical methylKit-style site with n=3 per group this is 4.
        df_i = (nv_case + nv_ctrl).astype(np.float64) - 2.0
        df_i_safe = np.where(df_i > 0, df_i, 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            phi_site = (chi_case + chi_ctrl) / df_i_safe

        # Sites with zero dispersion contribution (perfect fit OR degenerate
        # variance term) cannot inform φ̂_i. Apply the floor; the
        # ``min_dispersion`` clamp also handles the underdispersion case.
        phi_site = np.where(sites_dispers & (df_i > 0), phi_site, min_dispersion)
        phi_site = np.maximum(phi_site, min_dispersion)

        if dispersion == "site":
            phi_eff = phi_site
        else:  # "shrink": James-Stein-style weighted average toward chrom mean
            # φ̂_shrunk_i = (df_i · φ̂_site_i + w · φ̂_chrom) / (df_i + w)
            w   = float(shrink_pseudo_df)
            num = df_i_safe * phi_site + w * phi_hat
            den = df_i_safe + w
            phi_eff = np.maximum(num / den, min_dispersion)
            # Where the per-site estimator was unusable, fall back to the
            # chromosome value rather than the floor.
            phi_eff = np.where(sites_dispers & (df_i > 0), phi_eff, phi_hat)
    else:  # "chrom"
        phi_eff = np.full_like(sn_case, phi_hat, dtype=np.float64)

    # --- Test for H0: π_case = π_ctrl --------------------------------------
    # Both the score and LR statistics use the same per-group sufficient
    # statistics (sn_*, sm_*) and the same dispersion φ̂_eff. They differ
    # only in functional form; both are referenced to χ²₁ asymptotically.
    sn_total = sn_case + sn_ctrl
    sm_total = sm_case + sm_ctrl
    with np.errstate(invalid="ignore", divide="ignore"):
        pi_pool      = np.where(sn_total > 0, sm_total / sn_total, np.nan)
        pi_pool_safe = np.clip(pi_pool, eps, 1.0 - eps)

        # Variance of the null-MLE score U.  Used by both branches: the
        # ``score`` test divides U² by it; the ``lr`` test only needs it
        # as a degenerate-site guard (variance == 0 → no information at
        # that site, so the LR is also undefined).
        var_U_bin = (
            (sn_case * sn_ctrl / np.maximum(sn_total, 1.0))
            * pi_pool_safe
            * (1.0 - pi_pool_safe)
        )

        if statistic == "score":
            U = sm_case - sn_case * pi_pool
            stat_raw = np.where(var_U_bin > 0, U * U / var_U_bin, np.nan)
        else:  # "lr": closed-form quasi-binomial log-likelihood ratio
            # LR = 2 · Σ_g [ S1_g · log(p̂_g/p̂_pool)
            #             + (S0_g − S1_g) · log((1 − p̂_g)/(1 − p̂_pool)) ]
            # Each term is x · log(y/z) where y, z ∈ [ε, 1-ε] after clipping,
            # so log is bounded; the multiplicative 0 at x=0 cleanly zeroes
            # the contribution (no special-casing needed).
            pc_safe = np.clip(pi_case, eps, 1.0 - eps)
            pk_safe = np.clip(pi_ctrl, eps, 1.0 - eps)

            u_case = sm_case
            u_ctrl = sm_ctrl
            v_case = sn_case - sm_case
            v_ctrl = sn_ctrl - sm_ctrl

            lr_terms = (
                u_case * np.log(pc_safe / pi_pool_safe)
                + v_case * np.log((1.0 - pc_safe) / (1.0 - pi_pool_safe))
                + u_ctrl * np.log(pk_safe / pi_pool_safe)
                + v_ctrl * np.log((1.0 - pk_safe) / (1.0 - pi_pool_safe))
            )
            stat_raw = 2.0 * lr_terms

        # Apply quasi-binomial dispersion inflation per site/tile.
        chi2_stat = np.where(phi_eff > 0, stat_raw / phi_eff, np.nan)
        chi2_stat = np.where(var_U_bin > 0, chi2_stat, np.nan)

    # --- Reference distribution → p-value ---------------------------------
    # methylKit's logReg (R/diffMeth.R line 273):
    #     test = ifelse(test=="F" & phi>1, "F", "Chisq")
    # i.e. with the default test="F", logReg uses F(1, df_residual) at sites
    # where overdispersion was detected (phi > 1) and falls back to χ²(1)
    # where φ̂ was clamped to 1.
    df_resid = np.maximum(
        (nv_case + nv_ctrl).astype(np.float64) - 2.0,
        1.0,
    )
    if reference == "methylkit":
        # Per-site adaptive: F where phi_eff > 1 (i.e. real overdispersion
        # signal made it past the floor), chi² where phi_eff is clamped to 1.
        p_F    = sp_stats.f.sf(chi2_stat, dfn=1, dfd=df_resid)
        p_chi2 = sp_stats.chi2.sf(chi2_stat, df=1)
        pvals  = np.where(phi_eff > 1.0, p_F, p_chi2)
    elif reference == "F":
        pvals = sp_stats.f.sf(chi2_stat, dfn=1, dfd=df_resid)
    else:  # "chi2"
        pvals = sp_stats.chi2.sf(chi2_stat, df=1)

    degenerate = (
        ~sites_both
        | np.isnan(chi2_stat)
        | (var_U_bin <= 0)
    )
    pvals = np.where(degenerate, np.nan, pvals)

    log2_or = _safe_log2_odds_ratio(pi_case, pi_ctrl)
    log2_or[degenerate] = np.nan

    return pvals, log2_or, pi_case, pi_ctrl, phi_hat


def _beta_binom_mom_from_welford_logit(
    mean_case: np.ndarray,
    M2_case: np.ndarray,
    n_valid_case: np.ndarray,
    mean_ctrl: np.ndarray,
    M2_ctrl: np.ndarray,
    n_valid_ctrl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch t-test on logit-transformed beta values (from Welford accumulators).

    Beta values are highly skewed near boundaries [0, 1]. Logit transformation
    stabilizes variance and improves normality for robust t-testing.

    Parameters match _beta_binom_mom_from_welford; output is p-values and
    log2 odds ratio on original (non-logit) scale.
    """
    # Transform means to logit scale
    logit_mean_case = _logit_transform(mean_case)
    logit_mean_ctrl = _logit_transform(mean_ctrl)

    # Compute variance on logit scale via delta method
    # Var(logit(β)) = Var(β) × jacobian²
    jac_case = _logit_variance_jacobian(mean_case)
    jac_ctrl = _logit_variance_jacobian(mean_ctrl)

    var_case = M2_case / np.maximum(n_valid_case - 1, 1)
    var_ctrl = M2_ctrl / np.maximum(n_valid_ctrl - 1, 1)

    var_logit_case = var_case * (jac_case ** 2)
    var_logit_ctrl = var_ctrl * (jac_ctrl ** 2)

    # Normalize by sample size (variance of the mean)
    var_mean_logit_case = var_logit_case / np.maximum(n_valid_case, 1)
    var_mean_logit_ctrl = var_logit_ctrl / np.maximum(n_valid_ctrl, 1)

    se = np.sqrt(var_mean_logit_case + var_mean_logit_ctrl)

    # Welch t-test on logit scale
    with np.errstate(invalid="ignore", divide="ignore"):
        t_stat = np.where(se > 0, (logit_mean_case - logit_mean_ctrl) / se, np.nan)

    # Welch–Satterthwaite degrees of freedom
    dof_num = (var_mean_logit_case + var_mean_logit_ctrl) ** 2
    dof_den = (
        np.where(n_valid_case > 1, var_mean_logit_case ** 2 / (n_valid_case - 1), 0.0)
        + np.where(n_valid_ctrl > 1, var_mean_logit_ctrl ** 2 / (n_valid_ctrl - 1), 0.0)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        dof = np.where(dof_den > 0, dof_num / dof_den, 1.0)
        dof = np.maximum(dof, 1.0)

    pvals = 2.0 * sp_stats.t.sf(np.abs(t_stat), df=dof)

    # Mark degenerate cases
    degenerate = (
        np.isnan(mean_case) | np.isnan(mean_ctrl) | np.isnan(t_stat)
        | (n_valid_case == 0) | (n_valid_ctrl == 0)
    )
    pvals[degenerate] = np.nan

    # Compute log2 odds ratio on original scale using symmetric clamp (BIO-6)
    log2_ors = _safe_log2_odds_ratio(mean_case, mean_ctrl)
    log2_ors[degenerate] = np.nan

    return pvals, log2_ors


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

    # BIO-6: symmetric clamp on both group means so log2 OR cannot blow up
    # to ±inf when one group is at the boundary 0 or 1.
    log2_ors = _safe_log2_odds_ratio(mean_case, mean_ctrl)
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

    BIO-8 (this revision): the per-sample `sites` frame is now deduplicated on
    `pos` (keeping the first row) before the join.  Without this guard, a
    sample whose .cov file recorded both strands of one CpG dinucleotide
    (e.g. + at N and - at N+1 that were not merged by _merge_cpg_pairs)
    would produce one row per strand at the same pos, and the inner join
    on `pos` would multiply rows downstream — silently breaking the
    one-row-per-site contract that _load_sample_chrom relies on.
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

        # BIO-8: dedupe on pos to prevent duplicate-row blow-up from unmerged
        # +/- strand pairs of a single CpG dinucleotide.
        sites = (
            pl.read_parquet(str(part_file), columns=["pos", "strand"])
            .unique(subset=["pos"], keep="first")
        )

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

    # BIO-8: belt-and-braces dedupe in case a left-frame pos had multiple
    # right-frame strand variants after a join.
    return intersect.unique(subset=["pos"], keep="first").sort("pos")


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
    return (
        pl.concat(site_dfs)
        .unique(subset=["pos"], keep="first")  # BIO-8: dedupe on pos only
        .sort("pos")
    )


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

    df = (
        pl.read_parquet(str(part_file), columns=["pos", "N_meth", "coverage"])
        # BIO-8: collapse any duplicate-pos rows by summing reads so the
        # left-join below yields exactly one row per canonical position.
        .group_by("pos")
        .agg([
            pl.sum("N_meth").alias("N_meth"),
            pl.sum("coverage").alias("coverage"),
        ])
    )
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
    min_samples_case: int = 0,
    min_samples_control: int = 0,
    dispersion: str = "site",
    reference: str = "methylkit",
) -> pl.DataFrame:
    """Run DMC for one chromosome, loading one sample at a time.

    Memory design
    -------------
    Peak memory is O(n_sites) regardless of sample count for the
    Fisher and Welford paths:

        fisher / logit_t / beta_binomial:
            4 int64 running sums (Fisher) OR
            6 arrays per group (Welford: float64 mean, float64 M2, int32 n_valid)

    The CMH path caches the case-sample int32 (meth, coverage) arrays in
    memory because each case sample contributes to one stratum per control,
    so the case data is reused. Memory overhead: ~8 bytes × n_sites × n_case,
    which is bounded for typical experiments (n_case ≤ 10, ~300 MB on chr1).

    Statistical paths
    -----------------
    fisher
        Fisher exact on reads pooled across replicates. Emits a warning;
        anti-conservative because between-replicate variance is ignored.
        Provided for parity with single-rep tools and aggregate reporting.

    cmh
        Cochran-Mantel-Haenszel test with one 2×2 stratum per
        (case_i, ctrl_j) pair. Preserves between-replicate variability
        because each replicate contributes its own coverage marginal.

    logit_t / beta_binomial
        Welch t-test on per-replicate beta values (logit-transformed for
        `logit_t`). Welford accumulators give per-site variance without
        materialising the count matrix.
    """
    n_sites = len(canonical_df)
    if n_sites == 0:
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    canonical_pos = canonical_df.select("pos")

    # Welford accumulators provide mean_beta_case / mean_beta_ctrl and
    # n_valid_* for every code path. They are always populated even when
    # the chosen test (e.g. fisher) does not use the variance.
    mean_case, M2_case, n_valid_case = _welford_init(n_sites)
    mean_ctrl, M2_ctrl, n_valid_ctrl = _welford_init(n_sites)

    # --- Statistical test ---
    if test == "fisher":
        # BIO-5: Fisher exact on per-group POOLED read counts.
        #
        # The previous "fisher"/"cmh" path pooled control reads then ran
        # one CMH stratum per case sample against the pool, producing a
        # test that was structurally identical to Fisher on pooled reads
        # but with the variance term deflated by the pooling (effective N
        # inflated by Σ coverage per control). The corrected code below
        # makes the pooling explicit and routes it through the well-tested
        # fisher_exact_vectorized() helper.
        #
        # NOTE: this test ignores between-replicate variability. Use
        # logit_t or beta_binomial for replicate-aware testing at n ≥ 3.
        warnings.warn(
            "test='fisher' pools reads across replicates; "
            "between-sample variance is ignored and p-values may be "
            "anti-conservative at WGBS coverage. "
            "Prefer test='logit_t' or test='cmh' for n ≥ 3 replicates.",
            UserWarning,
            stacklevel=2,
        )
        meth_case_sum = np.zeros(n_sites, dtype=np.int64)
        cov_case_sum  = np.zeros(n_sites, dtype=np.int64)
        meth_ctrl_sum = np.zeros(n_sites, dtype=np.int64)
        cov_ctrl_sum  = np.zeros(n_sites, dtype=np.int64)

        for sample in samples_case:
            meth, cov = _load_sample_chrom(methylstore_path, chrom, sample, canonical_pos)
            meth_case_sum += meth.astype(np.int64)
            cov_case_sum  += cov.astype(np.int64)
            _welford_update(mean_case, M2_case, n_valid_case, meth, cov)
            del meth, cov

        for sample in samples_control:
            meth, cov = _load_sample_chrom(methylstore_path, chrom, sample, canonical_pos)
            meth_ctrl_sum += meth.astype(np.int64)
            cov_ctrl_sum  += cov.astype(np.int64)
            _welford_update(mean_ctrl, M2_ctrl, n_valid_ctrl, meth, cov)
            del meth, cov

        unmeth_case_sum = cov_case_sum - meth_case_sum
        unmeth_ctrl_sum = cov_ctrl_sum - meth_ctrl_sum
        pvals, log2_ors = fisher_exact_vectorized(
            meth_case_sum, unmeth_case_sum, meth_ctrl_sum, unmeth_ctrl_sum
        )
        del meth_case_sum, cov_case_sum, unmeth_case_sum
        del meth_ctrl_sum, cov_ctrl_sum, unmeth_ctrl_sum

    elif test == "cmh":
        # BIO-5: properly stratified CMH — one 2×2 stratum per
        # (case_i, ctrl_j) pair, so each replicate's coverage marginal
        # enters its own variance term V. With n_case = n_ctrl = 1 this
        # degenerates to a single 2×2 table and matches Fisher; with
        # replicates the variance term grows correctly with n_case × n_ctrl,
        # avoiding the inflated chi² of the old pooled-control approach.
        #
        # Memory: we cache the case samples in int32 (n_case × n_sites × 8 B)
        # so each is contributed against every control sample without
        # re-reading parquet n_ctrl times.
        ome, var_sum, or_num, or_den = _cmh_init(n_sites)

        case_data: list[tuple[np.ndarray, np.ndarray]] = []
        for sample in samples_case:
            meth, cov = _load_sample_chrom(methylstore_path, chrom, sample, canonical_pos)
            case_data.append((meth, cov))
            _welford_update(mean_case, M2_case, n_valid_case, meth, cov)

        for ctrl in samples_control:
            meth_c, cov_c = _load_sample_chrom(methylstore_path, chrom, ctrl, canonical_pos)
            _welford_update(mean_ctrl, M2_ctrl, n_valid_ctrl, meth_c, cov_c)
            for meth_case_i, cov_case_i in case_data:
                _cmh_update(
                    ome, var_sum, or_num, or_den,
                    meth_case_i, cov_case_i, meth_c, cov_c,
                )
            del meth_c, cov_c

        del case_data
        pvals, log2_ors = _cmh_finalize(ome, var_sum, or_num, or_den)
        del ome, var_sum, or_num, or_den

    elif test in ("score", "lr"):
        # Quasi-binomial count-model test with McCullagh-Nelder overdispersion.
        # Both the score and likelihood-ratio statistics share the same
        # streaming accumulators (sn, sm, sm²/n, nv per group) and the same
        # dispersion machinery; ``_score_finalize`` picks the functional form
        # based on the ``statistic=`` argument it receives.
        sn_case, sm_case, sm2n_case, nv_case = _score_init(n_sites)
        sn_ctrl, sm_ctrl, sm2n_ctrl, nv_ctrl = _score_init(n_sites)

        for sample in samples_case:
            meth, cov = _load_sample_chrom(methylstore_path, chrom, sample, canonical_pos)
            _score_update(sn_case, sm_case, sm2n_case, nv_case, meth, cov)
            # Welford accumulators are also updated so that downstream code
            # which reads ``n_valid_case`` for the BIO-7 guard sees the
            # same per-site sample count.  ``mean_case`` from Welford is
            # overwritten below with the coverage-weighted score-test
            # equivalent, so its post-update value is irrelevant.
            _welford_update(mean_case, M2_case, n_valid_case, meth, cov)
            del meth, cov

        for sample in samples_control:
            meth, cov = _load_sample_chrom(methylstore_path, chrom, sample, canonical_pos)
            _score_update(sn_ctrl, sm_ctrl, sm2n_ctrl, nv_ctrl, meth, cov)
            _welford_update(mean_ctrl, M2_ctrl, n_valid_ctrl, meth, cov)
            del meth, cov

        pvals, log2_ors, pi_case, pi_ctrl, _phi_hat = _score_finalize(
            sn_case, sm_case, sm2n_case, nv_case,
            sn_ctrl, sm_ctrl, sm2n_ctrl, nv_ctrl,
            chrom_name=chrom,
            dispersion=dispersion,
            statistic=test,
            reference=reference,
        )

        # Coverage-weighted (= pooled MLE) group methylation for output.
        # Overwrite Welford's unweighted means with the score-test
        # equivalents so the unified output block at the bottom of this
        # function reports the values consistent with the test's math.
        mean_case[:] = np.where(np.isnan(pi_case), 0.0, pi_case)
        mean_ctrl[:] = np.where(np.isnan(pi_ctrl), 0.0, pi_ctrl)
        # nv_case / nv_ctrl from the score path agree with n_valid_case /
        # n_valid_ctrl from Welford by construction, so we don't overwrite.

        del sn_case, sm_case, sm2n_case, nv_case
        del sn_ctrl, sm_ctrl, sm2n_ctrl, nv_ctrl

    elif test in ("beta_binomial", "logit_t"):
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
        if test == "logit_t":
            pvals, log2_ors = _beta_binom_mom_from_welford_logit(
                mean_case, M2_case, n_valid_case,
                mean_ctrl, M2_ctrl, n_valid_ctrl,
            )
        else:
            pvals, log2_ors = _beta_binom_mom_from_welford(
                mean_case, M2_case, n_valid_case,
                mean_ctrl, M2_ctrl, n_valid_ctrl,
            )

    else:
        raise NotImplementedError(
            f"Test '{test}' not implemented. "
            "Choose 'lr', 'score', 'fisher', 'cmh', 'logit_t', or 'beta_binomial'."
        )

    # --- BIO-3: equal-weight per-replicate mean beta ---
    # Welford mean IS the equal-weight nanmean — no extra storage needed.
    mean_beta_case = mean_case.astype(np.float32)
    mean_beta_ctrl = mean_ctrl.astype(np.float32)
    mean_beta_case[n_valid_case == 0] = np.nan
    mean_beta_ctrl[n_valid_ctrl == 0] = np.nan
    meth_diff = (mean_beta_case - mean_beta_ctrl).astype(np.float32)

    # BIO-7: per-site min-samples guard. Sites where fewer than
    # `min_samples_*` replicates contributed valid (coverage > 0) data
    # have their p-value masked to NaN. apply_multiple_testing_correction
    # passes NaNs through, so these sites are effectively excluded from
    # genome-wide FDR control without disturbing site-position alignment.
    if min_samples_case > 0 or min_samples_control > 0:
        keep_mask = (
            (n_valid_case >= max(min_samples_case, 0))
            & (n_valid_ctrl >= max(min_samples_control, 0))
        )
        n_dropped = int((~keep_mask).sum())
        if n_dropped > 0:
            logger.info(
                "  %s: masking %s/%s sites with n_valid_case < %d or "
                "n_valid_ctrl < %d",
                chrom, f"{n_dropped:,}", f"{n_sites:,}",
                min_samples_case, min_samples_control,
            )
            pvals = np.where(keep_mask, pvals, np.nan)
            log2_ors = np.where(keep_mask, log2_ors, np.nan)
            meth_diff = np.where(keep_mask, meth_diff, np.float32(np.nan))
            mean_beta_case = np.where(keep_mask, mean_beta_case, np.float32(np.nan))
            mean_beta_ctrl = np.where(keep_mask, mean_beta_ctrl, np.float32(np.nan))

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


def _validate_sample_size_and_warn(n_case: int, n_ctrl: int, test: str) -> None:
    """Validate sample sizes and issue appropriate warnings."""
    min_n = min(n_case, n_ctrl)
    max_n = max(n_case, n_ctrl)

    if min_n == 0:
        raise ValueError(
            "Cannot perform DMC with zero samples in a group. "
            f"n_case={n_case}, n_control={n_ctrl}"
        )

    if min_n == 1:
        logger.warning(
            "⚠️  CRITICAL: Only 1 replicate per group detected!\n"
            "   Statistical results are UNRELIABLE without biological replicates.\n"
            "   Effect sizes may be reported, but p-values should NOT be trusted.\n"
            "   Recommendation: Collect at least 3 biological replicates per group."
        )
    elif min_n == 2:
        logger.warning(
            "⚠️  WARNING: Only 2 replicates per group.\n"
            "   Statistical power is very low. Many true positives will be missed.\n"
            "   Recommendation: Use n≥3 for reliable differential methylation calling."
        )
    elif min_n < 6 and test == "beta_binomial":
        logger.warning(
            "⚠️  Beta-binomial test with n<6 may have poor variance estimates.\n"
            "   Consider using test='score' (recommended) or test='logit_t'."
        )

    if test == "fisher" and min_n >= 2:
        logger.info(
            "Tip: 'fisher' pools reads across replicates and ignores between-"
            "replicate variance. At n>=2 the 'lr' test is statistically "
            "preferable (quasi-binomial likelihood-ratio with MN dispersion)."
        )

    if max_n / min_n > 2:
        logger.warning(
            f"⚠️  Unbalanced design detected: n_case={n_case}, n_control={n_ctrl}\n"
            "   Large imbalance may reduce statistical power."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_chromosomes_dmc(
    methylstore_path: str,
    samples_case: list[str],
    samples_control: list[str],
    test: str = "lr",
    chromosomes: Optional[list[str]] = None,
    unite: bool = True,
    min_samples_case: int = 0,
    min_samples_control: int = 0,
    dispersion: str = "site",
    reference: str = "methylkit",
) -> pl.DataFrame:
    """Process differential methylation for all chromosomes.

    Parameters
    ----------
    methylstore_path : str
        Path to filtered partitioned Parquet methylstore.
    samples_case, samples_control : list[str]
        Sample identifiers for case and control groups.
    test : {"lr", "score", "fisher", "cmh", "logit_t", "beta_binomial"}
        Statistical test.
            "lr"       (default) — Quasi-binomial likelihood-ratio chi-square
                                   on per-group read counts with per-site
                                   McCullagh-Nelder dispersion. Matches
                                   methylKit's calculateDiffMeth
                                   (overdispersion='MN', test='Chisq').
                                   Recommended at n >= 2.
            "score"              — Pearson score statistic on the same
                                   accumulators. Marginally more powerful
                                   than "lr" but mildly anti-conservative
                                   when π̂ is near 0 or 1.
            "logit_t"            — Welch t on logit(beta), variance via
                                   Welford. Fallback when count-model
                                   assumptions are doubtful (e.g. very low
                                   coverage). Not a GLM.
            "beta_binomial"      — Welch t on raw betas. Despite the name,
                                   not a beta-binomial GLM; superseded by
                                   "lr". Kept for backward compatibility.
            "cmh"                — Cochran-Mantel-Haenszel with one stratum
                                   per (case_i, ctrl_j) pair.
            "fisher"             — Fisher exact on reads pooled across
                                   replicates (anti-conservative; warns).
    chromosomes : list[str], optional
        Chromosomes to process. Auto-detected when None.
    unite : bool
        If True (default), test only CpG sites covered in every sample
        (intersection / inner join).
        If False, test all sites covered in at least one sample
        (union / outer join).
    min_samples_case, min_samples_control : int
        BIO-7: per-site minimum number of replicates with non-zero coverage
        required in each group. Sites failing the threshold have their
        p-value masked to NaN before FDR correction. Use this with
        ``unite=False`` (union mode) to drop tests that effectively run on
        a singleton observation in one group.
    dispersion : {"site", "chrom", "shrink"}
        McCullagh-Nelder dispersion strategy used by ``test="lr"`` and
        ``test="score"``. Default ``"site"`` matches methylKit
        ``overdispersion="MN"``. See :func:`_score_finalize` for details.
        Ignored for other tests.
    reference : {"methylkit", "chi2", "F"}
        Reference distribution for the quasi-binomial test statistic.
        Default ``"methylkit"`` switches per-site between F(1, df) where
        φ̂ > 1 and χ²(1) where φ̂ was clamped to 1 — exactly what
        methylKit's ``logReg`` does (see ``R/diffMeth.R`` line 273).
        ``"chi2"`` and ``"F"`` force a single reference distribution
        regardless of dispersion. See :func:`_score_finalize` for details.
        Ignored for other tests.

    Returns
    -------
    pl.DataFrame
        Columns: chrom, pos, strand, n_case, n_control,
                 mean_beta_case, mean_beta_control,
                 pvalue, log2_odds_ratio, meth_diff

        For the ``score`` test, ``mean_beta_*`` are coverage-weighted (the
        group MLE proportion M/N). For all other tests they are the
        unweighted per-replicate mean (Welford). The two differ when
        per-replicate coverage is uneven.
    """
    store       = Path(methylstore_path)
    all_samples = samples_case + samples_control

    min_group = min(len(samples_case), len(samples_control))
    _validate_sample_size_and_warn(len(samples_case), len(samples_control), test)

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
        "DMC: %d case / %d control, test=%s, unite=%s, "
        "min_samples_case=%d, min_samples_control=%d",
        len(samples_case), len(samples_control), test, unite,
        min_samples_case, min_samples_control,
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
                min_samples_case=min_samples_case,
                min_samples_control=min_samples_control,
                dispersion=dispersion,
                reference=reference,
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
    test: str = "logit_t",
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
    pvalue_col: str = "pvalue",
    qvalue_col: str = "qvalue",
) -> pl.DataFrame:
    """Apply multiple testing correction (Benjamini-Hochberg default).

    Generalised to accept arbitrary p-value / q-value column names so the
    same routine can correct DMC and DMR tables. ``reject`` is written as
    ``<qvalue_col>_reject`` when the column name differs from the default
    so the two outputs don't collide.
    """
    from statsmodels.stats.multitest import multipletests

    pvals       = dmc_results[pvalue_col].to_numpy()
    nan_mask    = np.isnan(pvals)
    pvals_clean = np.where(nan_mask, 1.0, pvals)

    reject, qvals, _, _ = multipletests(pvals_clean, method=method)

    qvals  = np.where(nan_mask, np.nan,  qvals)
    reject = np.where(nan_mask, False,   reject)

    reject_col = "reject" if qvalue_col == "qvalue" else f"{qvalue_col}_reject"
    return dmc_results.with_columns([
        pl.Series(qvalue_col, qvals),
        pl.Series(reject_col, reject),
    ])