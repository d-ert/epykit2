"""Differential variability calling (tl.dvc / iEVORA-style)."""

from __future__ import annotations

import polars as pl
import pytest

import epykit as ep


def test_dvc_writes_expected_schema(synth_md_filtered):
    md = synth_md_filtered
    ep.tl.dvc(md, test="bartlett")
    assert "dvc" in md.varm
    df = md.varm["dvc"]
    for col in (
        "chrom", "pos", "n_treatment", "n_control",
        "var_treatment", "var_control", "var_log_ratio",
        "p_variance", "q_variance", "p_mean", "q_mean", "is_dvc",
    ):
        assert col in df.columns, f"missing column {col}"
    # is_dvc must be boolean.
    assert df.schema["is_dvc"] == pl.Boolean


@pytest.mark.parametrize("bad_test", ["levene", "brown_forsythe", "f_test"])
def test_dvc_rejects_unsupported_tests(synth_md_filtered, bad_test):
    """Only 'bartlett' is supported; others should raise a clear ValueError."""
    md = synth_md_filtered
    with pytest.raises(ValueError, match="bartlett"):
        ep.tl.dvc(md, test=bad_test)
