"""Tests for ``ep.pp.aggregate_regions`` (methylKit ``regionCounts`` analogue)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest


def _write_bed(path: Path, rows: list[tuple[str, int, int, str]]) -> Path:
    path.write_text(
        "\n".join(f"{c}\t{s}\t{e}\t{name}" for c, s, e, name in rows) + "\n"
    )
    return path


def test_aggregate_regions_round_trip(synth_md_filtered, tmp_path):
    """Aggregating to a few wide regions yields a row per (region, sample)."""
    import epykit as ep

    md = synth_md_filtered
    # Build a BED covering each chromosome with three large bins
    bed_rows: list[tuple[str, int, int, str]] = []
    # Pull min/max positions per chromosome from the filtered store
    chrom_bounds = (
        pl.scan_parquet(f"{md.store}/sample=*/chrom=*/part-*.parquet")
        .group_by("chrom")
        .agg([pl.min("pos").alias("lo"), pl.max("pos").alias("hi")])
        .collect()
    )
    for r in chrom_bounds.iter_rows(named=True):
        lo, hi = int(r["lo"]), int(r["hi"])
        if hi - lo < 30:
            continue
        third = (hi - lo) // 3
        bed_rows.extend([
            (r["chrom"], lo, lo + third, f"{r['chrom']}_a"),
            (r["chrom"], lo + third, lo + 2 * third, f"{r['chrom']}_b"),
            (r["chrom"], lo + 2 * third, hi + 1, f"{r['chrom']}_c"),
        ])
    bed = _write_bed(tmp_path / "regions.bed", bed_rows)

    ep.pp.aggregate_regions(md, str(bed), min_cpgs_per_region=1)

    # md.store now points at a fresh per-region partitioned methylstore
    new_store = Path(md.store)
    assert new_store.exists()
    sample_dirs = list(new_store.glob("sample=*"))
    assert len(sample_dirs) == md.n_samples

    df = pl.read_parquet(
        f"{md.store}/sample=*/chrom=*/part-*.parquet"
    )
    # Schema sanity
    for col in (
        "chrom", "pos", "strand", "context",
        "N_meth", "N_unmeth", "coverage", "sample",
        "region_id", "start", "end", "n_cpgs",
    ):
        assert col in df.columns, f"missing column {col}"
    assert df["coverage"].eq(df["N_meth"] + df["N_unmeth"]).all()
    # uns recorded
    assert md.uns["regions"]["n_regions"] == len(bed_rows)
    assert any(h["step"] == "regions" for h in md.uns["_store_history"])


def test_aggregate_regions_then_dmc(synth_md_filtered, tmp_path):
    """Downstream `tl.dmc` runs on the region-aggregated store without errors."""
    import epykit as ep

    md = synth_md_filtered
    chrom_bounds = (
        pl.scan_parquet(f"{md.store}/sample=*/chrom=*/part-*.parquet")
        .group_by("chrom")
        .agg([pl.min("pos").alias("lo"), pl.max("pos").alias("hi")])
        .collect()
    )
    bed_rows: list[tuple[str, int, int, str]] = []
    for r in chrom_bounds.iter_rows(named=True):
        lo, hi = int(r["lo"]), int(r["hi"])
        # 5 fat bins per chrom
        step = max(1, (hi - lo) // 5)
        for i in range(5):
            bed_rows.append(
                (r["chrom"], lo + i * step, lo + (i + 1) * step, f"{r['chrom']}_b{i}")
            )
    bed = _write_bed(tmp_path / "regions.bed", bed_rows)
    ep.pp.aggregate_regions(md, str(bed), min_cpgs_per_region=1)
    # Drop the unite marker since we aggregated; rerun unite so DMC works.
    md.uns.pop("unite", None)
    ep.pp.unite(md, type="intersect")
    ep.tl.dmc(md, test="lr")
    dmc = md.dmc
    assert dmc is not None and len(dmc) > 0
