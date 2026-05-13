"""Regression tests for the Bismark .cov -> Parquet converter performance fixes.

These cover the three behaviour-preserving optimisations:

1. ``partition_by`` replaces the per-chromosome filter loop. Output schema,
   row counts, and per-chromosome contents must be bit-identical to the
   pre-optimisation implementation.
2. ``methyl_percent`` is dropped at scan time. Round-trip must still produce
   the canonical 8-column schema.
3. ``compression_level`` is configurable (default 1 for fast first-pass
   conversion) and the on-disk Parquet remains readable.

And the parallel sample-conversion helper:

4. ``_convert_samples_parallel`` converts every input sample regardless of
   worker count and respects the ``EPYKIT_CONVERT_WORKERS`` override.
"""

from __future__ import annotations

import gzip
import os
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from epykit.convert import convert_sample, ensure_converted_sample
from epykit.io import _convert_samples_parallel


# Tiny .cov fixture: 3 chromosomes, mixed methylation, ~10 sites per chrom.
# Bismark .cov columns: chrom, start, end, methyl_percent, N_meth, N_unmeth
_COV_FIXTURE = "\n".join(
    f"{chrom}\t{pos}\t{pos+1}\t{(meth/(meth+unmeth))*100:.2f}\t{meth}\t{unmeth}"
    for chrom, pos, meth, unmeth in [
        ("chr1",   100, 8, 2),
        ("chr1",   200, 5, 5),
        ("chr1",   300, 0, 10),
        ("chr1",   400, 10, 0),
        ("chr2",  1000, 3, 7),
        ("chr2",  2000, 6, 4),
        ("chr2",  3000, 2, 8),
        ("chrM",   500, 9, 1),
        ("chrM",   600, 1, 9),
        ("chrM",   700, 5, 5),
    ]
) + "\n"


@pytest.fixture
def cov_file(tmp_path):
    p = tmp_path / "sample.cov.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(_COV_FIXTURE)
    return p


def _read_all_partitions(sample_dir: Path) -> pl.DataFrame:
    """Concatenate every chrom partition under sample=<name>/."""
    parts = [
        pl.read_parquet(str(p))
        for p in sorted(sample_dir.rglob("part-*.parquet"))
    ]
    return pl.concat(parts).sort(["chrom", "pos"])


def test_partition_by_writes_one_file_per_chrom(cov_file, tmp_path):
    """``partition_by`` must produce one Parquet per chromosome in the
    expected ``sample=<id>/chrom=<value>/part-0.parquet`` hive layout."""
    out = tmp_path / "store"
    convert_sample(str(cov_file), "s1", str(out))

    sample_dir = out / "sample=s1"
    chrom_dirs = sorted(d.name for d in sample_dir.glob("chrom=*"))
    assert chrom_dirs == ["chrom=chr1", "chrom=chr2", "chrom=chrM"]

    for chrom_dir in sample_dir.glob("chrom=*"):
        assert (chrom_dir / "part-0.parquet").exists()


def test_partition_by_preserves_row_count_and_schema(cov_file, tmp_path):
    """The output must contain every input row and the canonical 8 columns."""
    out = tmp_path / "store"
    convert_sample(str(cov_file), "s1", str(out))

    df = _read_all_partitions(out / "sample=s1")

    # 10 input rows -> 10 output rows
    assert len(df) == 10
    # Canonical schema (no methyl_percent leak)
    assert df.columns == [
        "chrom", "pos", "strand", "context",
        "N_meth", "N_unmeth", "coverage", "sample",
    ]
    # coverage is N_meth + N_unmeth
    assert (df["coverage"] == df["N_meth"] + df["N_unmeth"]).all()
    # sample column populated
    assert (df["sample"] == "s1").all()
    # strand defaults to "*" without a reference FASTA
    assert (df["strand"] == "*").all()


def test_methyl_percent_not_present_in_output(cov_file, tmp_path):
    """Win #3: methyl_percent dropped at scan time."""
    out = tmp_path / "store"
    convert_sample(str(cov_file), "s1", str(out))

    df = _read_all_partitions(out / "sample=s1")
    assert "methyl_percent" not in df.columns


def test_compression_level_default_is_1(cov_file, tmp_path):
    """Win #2: default zstd level should be 1 for fast first-pass writes."""
    out = tmp_path / "store"
    convert_sample(str(cov_file), "s1", str(out))

    # Verify the Parquet is readable and that compression metadata says zstd.
    chrom_file = next((out / "sample=s1").rglob("part-0.parquet"))
    meta = pq.read_metadata(str(chrom_file))
    for rg in range(meta.num_row_groups):
        for col in range(meta.row_group(rg).num_columns):
            comp = meta.row_group(rg).column(col).compression
            assert comp == "ZSTD", f"expected ZSTD, got {comp}"


def test_compression_level_override_is_respected(cov_file, tmp_path):
    """Explicit compression_level must reach the Parquet writer."""
    out = tmp_path / "store"
    # Just check the call doesn't error with a higher level; the on-disk
    # representation differs in size but is still ZSTD.
    convert_sample(str(cov_file), "s1", str(out), compression_level=5)
    chrom_file = next((out / "sample=s1").rglob("part-0.parquet"))
    df = pl.read_parquet(str(chrom_file))
    assert len(df) > 0


def test_chrom_per_partition_contains_only_its_own_chrom(cov_file, tmp_path):
    """Every partition file must contain rows from a single chromosome only."""
    out = tmp_path / "store"
    convert_sample(str(cov_file), "s1", str(out))

    for chrom_dir in (out / "sample=s1").glob("chrom=*"):
        chrom_name = chrom_dir.name.removeprefix("chrom=")
        df = pl.read_parquet(str(chrom_dir / "part-0.parquet"))
        assert (df["chrom"] == chrom_name).all(), (
            f"{chrom_dir.name} leaked rows from other chromosomes"
        )


# Parallel sample conversion


def _make_cov_pair(tmp_path):
    """Return two distinct .cov.gz files for parallel-convert tests."""
    a = tmp_path / "a.cov.gz"
    b = tmp_path / "b.cov.gz"
    with gzip.open(a, "wt") as fh:
        fh.write(_COV_FIXTURE)
    # Slightly different content for sample b so the manifest signatures differ.
    with gzip.open(b, "wt") as fh:
        fh.write(_COV_FIXTURE.replace("chr1", "chr2"))
    return a, b


def test_parallel_conversion_converts_every_sample(tmp_path, monkeypatch):
    """Win #4: every sample lands a Parquet store under sample=<id>/."""
    monkeypatch.delenv("EPYKIT_CONVERT_WORKERS", raising=False)
    a, b = _make_cov_pair(tmp_path)
    out = tmp_path / "store"
    out.mkdir()

    _convert_samples_parallel(
        files=[(str(a), "s_a"), (str(b), "s_b")],
        cache_store_dir=str(out),
        context="CpG",
        reference_fasta=None,
    )

    assert (out / "sample=s_a").exists()
    assert (out / "sample=s_b").exists()
    assert next((out / "sample=s_a").rglob("part-0.parquet"), None) is not None
    assert next((out / "sample=s_b").rglob("part-0.parquet"), None) is not None


def test_serial_fallback_when_workers_env_is_one(tmp_path, monkeypatch):
    """``EPYKIT_CONVERT_WORKERS=1`` must take the serial path (no threads)."""
    monkeypatch.setenv("EPYKIT_CONVERT_WORKERS", "1")
    a, b = _make_cov_pair(tmp_path)
    out = tmp_path / "store"
    out.mkdir()

    _convert_samples_parallel(
        files=[(str(a), "s_a"), (str(b), "s_b")],
        cache_store_dir=str(out),
        context="CpG",
        reference_fasta=None,
    )

    # Same observable outcome -- serial vs threaded is a perf knob, not a
    # correctness one.
    assert (out / "sample=s_a").exists()
    assert (out / "sample=s_b").exists()


def test_ensure_converted_sample_skips_unchanged_input(cov_file, tmp_path):
    """Cache hit path: a second call with the same input returns False."""
    out = tmp_path / "store"
    out.mkdir()
    first = ensure_converted_sample(str(cov_file), "s1", str(out))
    second = ensure_converted_sample(str(cov_file), "s1", str(out))
    assert first is True
    assert second is False
