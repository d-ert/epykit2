"""permutation-based empirical FDR for DMR."""

from __future__ import annotations

import polars as pl

import epykit as ep


def test_dmr_empirical_fdr_columns(synth_md_filtered):
    """tl.dmr(empirical_fdr=True) appends empirical_pvalue / qvalue columns."""
    md = synth_md_filtered
    ep.tl.dmr(
        md,
        method="tile",
        empirical_fdr=True,
        n_perm=5,  # tiny on the synthetic fixture; just smoke + schema test
        perm_seed=42,
        chromosomes=["chr1"],
    )
    dmr = md.uns["dmr"]
    assert isinstance(dmr, pl.DataFrame)
    if len(dmr) > 0:
        assert "empirical_pvalue" in dmr.columns
        assert "empirical_qvalue" in dmr.columns
        # Empirical pvalues are bounded in [0, 1].
        emp = dmr.get_column("empirical_pvalue").drop_nulls().to_numpy()
        assert (emp >= 0).all() and (emp <= 1).all()
    # Params record the permutation settings.
    params = md.uns["dmr_params"]
    assert params.get("empirical_fdr") is True
    assert params.get("n_perm") == 5
