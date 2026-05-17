"""meth_diff CI columns + welch_t / bb_lr / beta_binomial rename."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

import epykit as ep


@pytest.mark.parametrize("test", ["lr", "score", "logit_t", "welch_t", "bb_lr"])
def test_dmc_emits_meth_diff_ci_columns(synth_md_filtered, test):
    """Every DMC test path emits meth_diff_ci_lo / meth_diff_ci_hi."""
    md = synth_md_filtered
    ep.tl.dmc(md, test=test)
    df = md.get_dmc(test=test)
    assert df is not None
    assert "meth_diff_ci_lo" in df.columns
    assert "meth_diff_ci_hi" in df.columns
    # Sanity: for sites with non-NaN meth_diff, CI brackets it (allow ties).
    finite = df.filter(
        pl.col("meth_diff").is_not_null()
        & pl.col("meth_diff_ci_lo").is_not_null()
        & pl.col("meth_diff_ci_hi").is_not_null()
    )
    if len(finite) > 100:
        m = finite.get_column("meth_diff").to_numpy()
        lo = finite.get_column("meth_diff_ci_lo").to_numpy()
        hi = finite.get_column("meth_diff_ci_hi").to_numpy()
        assert np.all(lo <= m + 1e-6)
        assert np.all(m - 1e-6 <= hi)
        # Bracketing should be informative for at least most rows.
        frac_inside = float(((lo <= m) & (m <= hi)).mean())
        assert frac_inside >= 0.95


def test_beta_binomial_deprecation_warning(synth_md_filtered):
    """test='beta_binomial' fires a DeprecationWarning and routes
    to welch_t.
    """
    md = synth_md_filtered
    # Reset the module-level guard so the warning fires deterministically.
    from epykit import dmc as _dmc_mod
    _dmc_mod._WELCH_T_RENAME_WARNED = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ep.tl.dmc(md, test="beta_binomial")
    dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert any("beta_binomial" in str(x.message) for x in dep), (
        "Expected DeprecationWarning mentioning 'beta_binomial', got "
        f"{[str(x.message) for x in w]}"
    )
    # The result lands at key 'dmc_welch_t'
    assert "dmc_welch_t" in md.varm


def test_bb_lr_is_distinct_from_lr(synth_md_filtered):
    """bb_lr (true quasi-binomial LRT) produces a separate
    output table from lr and surfaces coef_treatment / coef_se.
    """
    md = synth_md_filtered
    ep.tl.dmc(md, test="bb_lr")
    df = md.get_dmc(test="bb_lr")
    assert df is not None
    assert "coef_treatment" in df.columns
    assert "coef_se" in df.columns
    # Most coef_treatment values should be finite somewhere.
    coef = df.get_column("coef_treatment").drop_nulls().to_numpy()
    assert coef.size > 0
    assert np.isfinite(coef).any()
