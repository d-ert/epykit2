"""Tests for ``ep.pl.tss_metaplot``."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import polars as pl
import pytest

matplotlib.use("Agg", force=True)


def _write_synthetic_gtf(path: Path, synth_md) -> Path:
    """Write a small GTF whose TSS coordinates land inside the synthetic
    chromosomes so the metaplot has real CpGs to bin.
    """
    chrom_positions = (
        pl.scan_parquet(f"{synth_md.store}/sample=*/chrom=*/part-*.parquet")
        .select(["chrom", "pos"])
        .unique()
        .group_by("chrom")
        .agg([pl.col("pos").sort().alias("positions")])
        .collect()
    )
    lines = ['##gtf-version 2']
    gene_idx = 0
    for row in chrom_positions.iter_rows(named=True):
        chrom = row["chrom"]
        positions = row["positions"]
        # Pick 2 gene starts per chromosome at roughly 1/3 and 2/3 of the
        # observed range, on the + strand.
        if len(positions) < 6:
            continue
        for frac, strand in ((0.33, "+"), (0.66, "-")):
            tss = int(positions[int(len(positions) * frac)])
            start = tss + 1  # GTF is 1-based
            end = tss + 1000
            gene_idx += 1
            attrs = (
                f'gene_id "synth_gene_{gene_idx}"; '
                f'gene_name "synth_gene_{gene_idx}";'
            )
            lines.append(
                "\t".join([
                    chrom, "synth", "gene", str(start), str(end), ".", strand, ".", attrs
                ])
            )
            # one exon to keep parser happy
            lines.append(
                "\t".join([
                    chrom, "synth", "exon", str(start), str(end), ".", strand, ".", attrs
                ])
            )
    path.write_text("\n".join(lines) + "\n")
    return path


def test_tss_metaplot_smoke(synth_md_filtered, tmp_path):
    import epykit as ep

    gtf = _write_synthetic_gtf(tmp_path / "synth.gtf", synth_md_filtered)
    fig, ax = ep.pl.tss_metaplot(
        synth_md_filtered, str(gtf),
        window_bp=2000, n_bins=20, group_by="group", max_genes=200,
    )
    # Lines drawn (one per sample, possibly + per-group means)
    assert len(ax.lines) >= synth_md_filtered.n_samples
    assert ax.get_xlabel().startswith("Distance from TSS")
    plt.close(fig)


def test_tss_metaplot_no_group(synth_md_filtered, tmp_path):
    import epykit as ep

    gtf = _write_synthetic_gtf(tmp_path / "synth2.gtf", synth_md_filtered)
    fig, ax = ep.pl.tss_metaplot(
        synth_md_filtered, str(gtf),
        window_bp=1500, n_bins=15, group_by=None, max_genes=200,
    )
    assert len(ax.lines) >= synth_md_filtered.n_samples
    plt.close(fig)
