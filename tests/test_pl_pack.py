"""Plan 2 §6: visualization pack smoke tests."""

from __future__ import annotations

import polars as pl
import pytest

import epykit as ep
import matplotlib

matplotlib.use("Agg", force=True)


def test_pl_umap_returns_axes_or_skips(synth_md_filtered):
    pytest.importorskip("umap")
    md = synth_md_filtered
    fig, ax = ep.pl.umap(md, n_neighbors=4, min_dist=0.3)
    assert ax is not None


def test_pl_sample_correlation_renders(synth_md_filtered):
    md = synth_md_filtered
    fig, ax = ep.pl.sample_correlation(md, method="spearman", cluster=False)
    assert ax is not None
    # Should have populated md.uns
    assert "qc_sample_correlation" in md.uns


def test_pl_qc_dashboard_renders(synth_md_filtered):
    md = synth_md_filtered
    ep.tl.qc(md, run_sample_correlation=True)
    fig, axes = ep.pl.qc_dashboard(md)
    assert len(axes) >= 5


def test_pl_dmr_boxplot_needs_dmr(synth_md_filtered):
    md = synth_md_filtered
    with pytest.raises(ValueError):
        ep.pl.dmr_boxplot(md, top_n=3)
    ep.tl.dmr(md, method="tile", chromosomes=["chr1"])
    if len(md.uns.get("dmr", pl.DataFrame())) > 0:
        fig, axes = ep.pl.dmr_boxplot(md, top_n=3)
        assert len(axes) >= 1
