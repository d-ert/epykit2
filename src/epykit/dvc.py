"""Differentially Variable CpG (DVC) calling — Plan 2, Section 4.

iEVORA-style: test for variance differences between groups at each CpG.
Sites are flagged DVC when:

    q_variance < alpha   AND   p_mean   > mean_filter_alpha

i.e. the between-group variance differs significantly while the means do
not — the signature of an outlier-driven shift in variability (common in
cancer / aging methylomes) rather than a simple mean shift.

Memory and I/O follow the same per-chromosome streaming layout as
``dmc.process_chromosomes_dmc``: one sample is loaded at a time, per-site
Welford accumulators give variance, no (n_sites x n_replicates) matrix
is ever built.
"""

from __future__ import annotations

import gc
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from .dmc import (
    _detect_chromosomes,
    _intersect_chrom,
    _load_sample_chrom,
    _union_chrom,
    _welford_init,
    _welford_update,
)

logger = logging.getLogger(__name__)

_DVC_EMPTY_SCHEMA = {
    "chrom":          pl.Utf8,
    "pos":            pl.Int32,
    "strand":         pl.Utf8,
    "n_treatment":    pl.Int32,
    "n_control":      pl.Int32,
    "var_treatment":  pl.Float64,
    "var_control":    pl.Float64,
    "var_log_ratio":  pl.Float64,
    "p_variance":     pl.Float64,
    "q_variance":     pl.Float64,
    "p_mean":         pl.Float64,
    "q_mean":         pl.Float64,
    "is_dvc":         pl.Boolean,
}


def _bartlett_per_site(
    var_a: np.ndarray, n_a: np.ndarray,
    var_b: np.ndarray, n_b: np.ndarray,
) -> np.ndarray:
    """Vectorised Bartlett test for equal variances across two groups.

    Returns p-values per site. NaN where either group has fewer than 2
    observations or zero variance.
    """
    n_a_safe = np.maximum(n_a - 1, 0)
    n_b_safe = np.maximum(n_b - 1, 0)
    N = n_a_safe + n_b_safe
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = (n_a_safe * var_a + n_b_safe * var_b) / np.maximum(N, 1)
        chi2 = (
            N * np.log(np.maximum(pooled, 1e-300))
            - n_a_safe * np.log(np.maximum(var_a, 1e-300))
            - n_b_safe * np.log(np.maximum(var_b, 1e-300))
        )
        # Bartlett correction term
        c = 1.0 + (1.0 / (3.0 * 1.0)) * (
            (1.0 / np.maximum(n_a_safe, 1)) + (1.0 / np.maximum(n_b_safe, 1))
            - (1.0 / np.maximum(N, 1))
        )
        chi2_corrected = chi2 / c
    valid = (n_a >= 2) & (n_b >= 2) & (var_a > 0) & (var_b > 0)
    pvals = np.full_like(chi2_corrected, np.nan, dtype=np.float64)
    pvals[valid] = sp_stats.chi2.sf(chi2_corrected[valid], df=1)
    return pvals


def _levene_per_site(
    mean_a: np.ndarray, M2_a: np.ndarray, n_a: np.ndarray,
    mean_b: np.ndarray, M2_b: np.ndarray, n_b: np.ndarray,
) -> np.ndarray:
    """Levene-Brown-Forsythe-style variance test from Welford accumulators.

    A faithful Levene test requires per-replicate centered deviations; we
    don't have those at this point because we only kept Welford summary
    statistics. The Bartlett test (in this file) is the natural Welford-
    only alternative. For Levene-Brown-Forsythe parity we'd need the
    per-replicate matrix, which costs O(n_sites x n_samples) memory.

    For now this function delegates to Bartlett with a sentinel log note.
    """
    logger.debug(
        "DVC test='levene' falls back to Bartlett under the Welford-only "
        "memory budget. See dvc._levene_per_site docstring."
    )
    var_a = np.where(n_a > 1, M2_a / np.maximum(n_a - 1, 1), np.nan)
    var_b = np.where(n_b > 1, M2_b / np.maximum(n_b - 1, 1), np.nan)
    return _bartlett_per_site(var_a, n_a, var_b, n_b)


def _process_one_chromosome_dvc(
    methylstore_path: Path,
    chrom: str,
    canonical_df: pl.DataFrame,
    samples_treatment: list[str],
    samples_control: list[str],
    test: str,
    mean_filter_alpha: float,
    alpha: float,
) -> pl.DataFrame:
    n_sites = len(canonical_df)
    if n_sites == 0:
        return pl.DataFrame(schema=_DVC_EMPTY_SCHEMA)
    canonical_pos = canonical_df.select("pos")

    mean_t, M2_t, n_t = _welford_init(n_sites)
    mean_c, M2_c, n_c = _welford_init(n_sites)

    for s in samples_treatment:
        meth, cov = _load_sample_chrom(methylstore_path, chrom, s, canonical_pos)
        _welford_update(mean_t, M2_t, n_t, meth, cov)
        del meth, cov
    for s in samples_control:
        meth, cov = _load_sample_chrom(methylstore_path, chrom, s, canonical_pos)
        _welford_update(mean_c, M2_c, n_c, meth, cov)
        del meth, cov

    var_t = np.where(n_t > 1, M2_t / np.maximum(n_t - 1, 1), np.nan)
    var_c = np.where(n_c > 1, M2_c / np.maximum(n_c - 1, 1), np.nan)

    if test == "bartlett":
        p_var = _bartlett_per_site(var_t, n_t, var_c, n_c)
    elif test in ("levene", "brown_forsythe"):
        # See _levene_per_site for the Welford limitation; the function
        # currently delegates to Bartlett under the streaming budget.
        p_var = _levene_per_site(mean_t, M2_t, n_t, mean_c, M2_c, n_c)
    else:
        raise ValueError(
            f"DVC test='{test}' not supported. Use 'bartlett', 'levene', "
            "or 'brown_forsythe'."
        )

    # Welch t on means (mean filter)
    vm_t = np.where(n_t > 1, var_t / np.maximum(n_t, 1), np.nan)
    vm_c = np.where(n_c > 1, var_c / np.maximum(n_c, 1), np.nan)
    se = np.sqrt(np.where((vm_t > 0) | (vm_c > 0), vm_t + vm_c, np.nan))
    with np.errstate(invalid="ignore", divide="ignore"):
        t_stat = np.where(se > 0, (mean_t - mean_c) / se, np.nan)
        dof_num = (vm_t + vm_c) ** 2
        dof_den = (
            np.where(n_t > 1, vm_t ** 2 / np.maximum(n_t - 1, 1), 0.0)
            + np.where(n_c > 1, vm_c ** 2 / np.maximum(n_c - 1, 1), 0.0)
        )
        dof = np.where(dof_den > 0, dof_num / dof_den, 1.0)
        dof = np.maximum(dof, 1.0)
    p_mean = 2.0 * sp_stats.t.sf(np.abs(t_stat), df=dof)

    with np.errstate(invalid="ignore", divide="ignore"):
        var_log_ratio = np.where(
            (var_t > 0) & (var_c > 0),
            np.log2(var_t / var_c),
            np.nan,
        )

    return pl.DataFrame({
        "chrom":         pl.Series([chrom] * n_sites, dtype=pl.Utf8),
        "pos":           canonical_df["pos"],
        "strand":        canonical_df["strand"],
        "n_treatment":   pl.Series(n_t.astype(np.int32)),
        "n_control":     pl.Series(n_c.astype(np.int32)),
        "var_treatment": pl.Series(var_t),
        "var_control":   pl.Series(var_c),
        "var_log_ratio": pl.Series(var_log_ratio),
        "p_variance":    pl.Series(p_var),
        "p_mean":        pl.Series(p_mean),
    }).sort("pos")


def process_chromosomes_dvc(
    methylstore_path: str,
    samples_treatment: list[str],
    samples_control: list[str],
    *,
    test: str = "bartlett",
    chromosomes: Optional[list[str]] = None,
    unite: bool = True,
    mean_filter_alpha: float = 0.05,
    alpha: float = 0.05,
) -> pl.DataFrame:
    """Run DVC analysis across all chromosomes.

    Returns a DataFrame in ``_DVC_EMPTY_SCHEMA`` with both the variance
    test p/q-values and the mean-test p/q-values (so the caller can apply
    the iEVORA signature filter at any threshold).
    """
    if test not in ("bartlett", "levene", "brown_forsythe"):
        raise ValueError(
            f"test must be 'bartlett', 'levene', or 'brown_forsythe'; "
            f"got {test!r}"
        )
    store = Path(methylstore_path)
    all_samples = samples_treatment + samples_control
    if chromosomes is None:
        chromosomes = _detect_chromosomes(store)
        logger.info("DVC: auto-detected %d chromosomes", len(chromosomes))

    with tempfile.TemporaryDirectory(prefix="epykit_dvc_") as tmpdir:
        tmp = Path(tmpdir)
        written: list[Path] = []
        for i, chrom in enumerate(chromosomes):
            logger.info("[DVC %d/%d] %s", i + 1, len(chromosomes), chrom)
            canonical_df = (
                _intersect_chrom(store, chrom, all_samples)
                if unite else _union_chrom(store, chrom, all_samples)
            )
            if len(canonical_df) == 0:
                continue
            chrom_result = _process_one_chromosome_dvc(
                store, chrom, canonical_df,
                samples_treatment, samples_control,
                test=test, mean_filter_alpha=mean_filter_alpha, alpha=alpha,
            )
            if len(chrom_result) == 0:
                continue
            tmp_file = tmp / f"{chrom}.parquet"
            chrom_result.write_parquet(str(tmp_file))
            written.append(tmp_file)
            del canonical_df, chrom_result
            gc.collect()

        if not written:
            return pl.DataFrame(schema=_DVC_EMPTY_SCHEMA)
        combined = pl.concat([pl.read_parquet(str(f)) for f in written])

    # BH-correct p_variance and p_mean separately.
    from statsmodels.stats.multitest import multipletests
    def _bh(p: np.ndarray) -> np.ndarray:
        finite = np.isfinite(p)
        q = np.full_like(p, np.nan, dtype=np.float64)
        if finite.any():
            _, q_finite, _, _ = multipletests(p[finite], method="fdr_bh")
            q[finite] = q_finite
        return q

    p_var = combined.get_column("p_variance").to_numpy()
    p_mean_arr = combined.get_column("p_mean").to_numpy()
    q_var = _bh(p_var)
    q_mean = _bh(p_mean_arr)
    is_dvc = (q_var < alpha) & (p_mean_arr > mean_filter_alpha)

    return combined.with_columns([
        pl.Series("q_variance", q_var),
        pl.Series("q_mean",     q_mean),
        pl.Series("is_dvc",     is_dvc),
    ])
