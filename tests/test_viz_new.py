"""Smoke tests for new viz features: karyogram, dmr_overlap, gene_body_metaplot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import epykit as ep


# karyogram


def test_karyogram_basic(synth_md_filtered):
    """karyogram() runs on a small DMC table and produces a figure with
    one row per chromosome."""
    ep.tl.dmc(synth_md_filtered, test="lr")
    fig, ax = ep.pl.karyogram(
        synth_md_filtered, value="meth_diff", bin_size_bp=50_000,
    )
    n_chroms = (
        synth_md_filtered.dmc.get_column("chrom").unique().to_list()
    )
    assert len(ax.get_yticklabels()) == len(n_chroms)
    assert ax.get_ylabel() == "Chromosome"


def test_karyogram_log10_qvalue_autocomputes(synth_md_filtered):
    """Asking for -log10_qvalue when it's absent derives it from qvalue."""
    ep.tl.dmc(synth_md_filtered, test="lr")
    fig, ax = ep.pl.karyogram(
        synth_md_filtered, value="-log10_qvalue", bin_size_bp=100_000,
        cmap="viridis",
    )
    # imshow renders the colormap; just confirm the title carries the metric.
    assert "log10_qvalue" in ax.get_title()


def test_karyogram_only_significant(synth_md_filtered):
    ep.tl.dmc(synth_md_filtered, test="lr")
    fig, ax = ep.pl.karyogram(
        synth_md_filtered, only_significant=True, alpha=0.05,
        bin_size_bp=100_000,
    )
    # Should render without erroring (some DMCs in the fixture are
    # significant — see test_accuracy).
    assert len(ax.get_yticklabels()) > 0


# dmr_overlap


def _toy_dmr_table(rows: list[tuple[str, int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema={"chrom": pl.Utf8, "start": pl.Int64, "end": pl.Int64},
        orient="row",
    )


def test_dmr_overlap_venn_2_sets():
    a = _toy_dmr_table([
        ("chr1", 100, 200),
        ("chr1", 300, 400),
        ("chr1", 500, 600),
    ])
    b = _toy_dmr_table([
        ("chr1", 100, 200),   # shared with a
        ("chr1", 300, 400),   # shared with a
        ("chr1", 700, 800),   # unique to b
    ])
    fig, ax = ep.pl.dmr_overlap({"A": a, "B": b})
    # Just confirm a figure was produced and 5 text annotations exist
    # (set labels + 3 counts).
    texts = ax.texts
    assert len(texts) >= 5


def test_dmr_overlap_upset_3_sets():
    a = _toy_dmr_table([("chr1", i*100, i*100+50) for i in range(10)])
    b = _toy_dmr_table([("chr1", i*100, i*100+50) for i in range(5, 12)])
    c = _toy_dmr_table([("chr1", i*100, i*100+50) for i in range(3, 8)])
    fig, axes = ep.pl.dmr_overlap({"early": a, "mid": b, "late": c})
    # Returned axes for UpSet is a 3-tuple (bar, matrix, totals).
    assert isinstance(axes, tuple) and len(axes) == 3
    ax_bar = axes[0]
    # The bar axis should have at least one bar.
    assert len(ax_bar.patches) >= 1


def test_dmr_overlap_too_few_sets_errors():
    a = _toy_dmr_table([("chr1", 0, 10)])
    with pytest.raises(ValueError, match="at least 2"):
        ep.pl.dmr_overlap({"only_one": a})


def test_dmr_overlap_too_many_sets_errors():
    sets = {chr(ord("A") + i): _toy_dmr_table([("chr1", 0, 10)]) for i in range(7)}
    with pytest.raises(ValueError, match="up to 6"):
        ep.pl.dmr_overlap(sets)


# gene_body_metaplot (smoke test only — needs a tiny GTF)


def _write_tiny_gtf(path: Path) -> None:
    """A minimal 4-gene GTF that covers the same chromosomes as the
    synth fixture (chr1..chr5)."""
    rows = []
    for i, chrom in enumerate(("chr1", "chr2", "chr3", "chr4")):
        start = 5_000 + i * 1_000
        end = start + 50_000  # >min_gene_bp = 500
        strand = "+" if i % 2 == 0 else "-"
        rows.append(
            f'{chrom}\thavana\tgene\t{start}\t{end}\t.\t{strand}\t.\t'
            f'gene_id "g{i}"; gene_name "Gene{i}";'
        )
    path.write_text("\n".join(rows) + "\n")


def test_gene_body_metaplot_runs(synth_md_filtered, tmp_path):
    gtf = tmp_path / "tiny.gtf"
    _write_tiny_gtf(gtf)
    fig, ax = ep.pl.gene_body_metaplot(
        synth_md_filtered, str(gtf),
        flank_bp=2000, n_bins_flank=10, n_bins_body=20,
        min_gene_bp=500, group_by="group",
    )
    # Three labeled regions: -flank, TSS, gene body, TES, +flank.
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert "TSS" in labels and "TES" in labels
    assert ax.get_ylabel() == "Mean β"
