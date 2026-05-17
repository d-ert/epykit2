"""Plan 2 §4: differential variability calling (tl.dvc / iEVORA-style)."""

from __future__ import annotations

import polars as pl

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


def test_dvc_levene_and_brown_forsythe_aliases(synth_md_filtered):
    """Both 'levene' and 'brown_forsythe' should run without error."""
    md = synth_md_filtered
    ep.tl.dvc(md, test="levene")
    assert "dvc" in md.varm
    ep.tl.dvc(md, test="brown_forsythe")
    assert "dvc" in md.varm
