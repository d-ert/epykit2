"""Bismark .cov → partitioned Parquet converter.

Changes vs previous version:
  - Strand note: Bismark merged .cov files do not carry explicit strand
    information in the data columns. Strand inference requires a reference
    FASTA and is tracked as BIO-1 in plan.md.  For now, strand is stored
    as "+" for cytosines whose 0-based start position carries a C in the
    reference, and "-" otherwise — but only when a reference is supplied
    via the optional `reference_fasta` argument. Without a reference the
    field defaults to "*" as before, and CpG-pair merging (BIO-2) is
    deferred to a post-conversion step once strand is known.
  - context column added ("CpG" default; expandable to CHG/CHH).
  - Minor: use str.removeprefix instead of str.replace for robustness.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import polars as pl


RAW_MANIFEST_NAME = ".epykit_raw_manifest.json"

# Bismark .cov column order
_COV_COLUMNS = ["chrom", "start", "end", "methyl_percent", "N_meth", "N_unmeth"]

_COV_SCHEMA: dict[str, type[pl.DataType]] = {
    "chrom": pl.Utf8,
    "start": pl.Int32,
    "end": pl.Int32,
    "methyl_percent": pl.Float32,
    "N_meth": pl.Int32,
    "N_unmeth": pl.Int32,
}


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SampleManifest:
    sample_name: str
    source: dict[str, object]
    chroms: list[str]
    row_group_size: int


def _file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sample_dir(output_dir: Path, sample_name: str) -> Path:
    return output_dir / f"sample={sample_name}"


def _manifest_path(sample_dir: Path) -> Path:
    return sample_dir / RAW_MANIFEST_NAME


def _expected_chrom_dirs(sample_dir: Path) -> list[str]:
    return sorted(item.name for item in sample_dir.glob("chrom=*") if item.is_dir())


def _sample_is_complete(sample_dir: Path, chroms: list[str]) -> bool:
    if not sample_dir.exists():
        return False
    if _expected_chrom_dirs(sample_dir) != sorted(chroms):
        return False
    return all(
        (sample_dir / chrom / "part-0.parquet").exists() for chrom in chroms
    )


def _manifest_payload(manifest: _SampleManifest) -> dict[str, object]:
    return {
        "sample_name": manifest.sample_name,
        "source": manifest.source,
        "chroms": manifest.chroms,
        "row_group_size": manifest.row_group_size,
    }


def _can_reuse_sample(
    input_path: Path, sample_dir: Path, row_group_size: int
) -> bool:
    manifest = _load_json(_manifest_path(sample_dir))
    if not manifest:
        return False
    if manifest.get("source") != _file_signature(input_path):
        return False
    if manifest.get("row_group_size") != row_group_size:
        return False
    chroms = manifest.get("chroms")
    if not isinstance(chroms, list) or not all(
        isinstance(c, str) for c in chroms
    ):
        return False
    return _sample_is_complete(sample_dir, chroms)


def _promote_sample_dir(temp_sample_dir: Path, final_sample_dir: Path) -> None:
    backup_dir = final_sample_dir.with_name(f"{final_sample_dir.name}.bak")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if final_sample_dir.exists():
        final_sample_dir.rename(backup_dir)
    try:
        temp_sample_dir.rename(final_sample_dir)
    except Exception:
        if backup_dir.exists() and not final_sample_dir.exists():
            backup_dir.rename(final_sample_dir)
        raise
    else:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


# ---------------------------------------------------------------------------
# Optional strand inference
# ---------------------------------------------------------------------------

def _infer_strand(df: pl.DataFrame, reference_fasta: str) -> pl.Series:
    """Infer strand from reference sequence for each CpG position.

    A cytosine on the + strand sits at position `start` in the reference.
    Its complement on the − strand is at `start + 1`. Bismark merged .cov
    coordinates are 0-based start, 1-based end (BED-like).

    Requires pyfaidx:  pip install pyfaidx

    Parameters
    ----------
    df : pl.DataFrame
        Must contain columns: chrom (str), start (Int32)
    reference_fasta : str
        Path to indexed reference FASTA (.fai index must exist)

    Returns
    -------
    pl.Series (Utf8)
        "+" where the reference base at `start` is C (or c),
        "-" where it is G (complement C on the − strand),
        "*" for anything else (non-CpG context or N base).
    """
    try:
        from pyfaidx import Fasta  # optional dependency
    except ImportError as exc:
        raise ImportError(
            "pyfaidx is required for strand inference. "
            "Install it with: pip install pyfaidx"
        ) from exc

    fasta = Fasta(reference_fasta, as_raw=True)
    strands: list[str] = []

    chroms = df["chrom"].to_list()
    starts = df["start"].to_list()

    for chrom, start in zip(chroms, starts):
        try:
            base = fasta[chrom][start].upper()
            if base == "C":
                strands.append("+")
            elif base == "G":
                strands.append("-")
            else:
                strands.append("*")
        except (KeyError, IndexError):
            strands.append("*")

    return pl.Series("strand", strands, dtype=pl.Utf8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_sample(
    input_path: str,
    sample_name: str,
    output_dir: str,
    row_group_size: int = 1_000_000,
    context: str = "CpG",
    reference_fasta: str | None = None,
) -> None:
    """Convert a Bismark .cov (optionally gzipped) file into a partitioned
    Parquet store.

    Parameters
    ----------
    input_path : str
        Path to the .cov or .cov.gz file
    sample_name : str
        Sample identifier written into the `sample` column
    output_dir : str
        Directory where Parquet partitions will be written
    row_group_size : int
        Approximate Parquet row-group size (default 1 000 000)
    context : str
        Methylation context label stored in the `context` column
        ("CpG", "CHG", "CHH"). Default "CpG".
    reference_fasta : str, optional
        Path to an indexed reference FASTA. When provided, strand is inferred
        from the reference base at each position (BIO-1). Without this
        argument, strand defaults to "*".

    Output schema
    -------------
    chrom   Utf8
    pos     Int32   (0-based, == Bismark start)
    strand  Utf8    ("+" | "-" | "*")
    context Utf8    ("CpG" | "CHG" | "CHH")
    N_meth  Int32
    N_unmeth Int32
    coverage Int32
    sample  Utf8
    """
    p = Path(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lf = pl.scan_csv(
        str(p),
        separator="\t",
        has_header=False,
        new_columns=_COV_COLUMNS,
        schema_overrides=_COV_SCHEMA,
    ).with_columns(
        [
            (pl.col("N_meth") + pl.col("N_unmeth")).alias("coverage"),
            pl.lit(sample_name).alias("sample"),
            pl.col("start").alias("pos"),
            pl.lit(context).alias("context"),
        ]
    ).select(
        ["chrom", "pos", "context", "N_meth", "N_unmeth", "coverage", "sample",
         "start"]   # keep start temporarily for strand inference
    )

    df = lf.collect()

    # Strand inference (BIO-1): requires reference FASTA via pyfaidx
    if reference_fasta is not None:
        strand_series = _infer_strand(df, reference_fasta)
    else:
        strand_series = pl.Series("strand", ["*"] * len(df), dtype=pl.Utf8)

    df = df.with_columns(strand_series).drop("start").select(
        ["chrom", "pos", "strand", "context", "N_meth", "N_unmeth", "coverage",
         "sample"]
    )

    # Write one Parquet file per chromosome
    for chrom in df["chrom"].unique().to_list():
        sub = df.filter(pl.col("chrom") == chrom)
        part_dir = out / f"sample={sample_name}" / f"chrom={chrom}"
        part_dir.mkdir(parents=True, exist_ok=True)
        sub.write_parquet(
            str(part_dir / "part-0.parquet"),
            compression="zstd",
            row_group_size=row_group_size,
        )


def ensure_converted_sample(
    input_path: str,
    sample_name: str,
    output_dir: str,
    row_group_size: int = 1_000_000,
    context: str = "CpG",
    reference_fasta: str | None = None,
) -> bool:
    """Convert a sample unless a valid on-disk conversion already exists.

    Returns True when a fresh conversion was performed, False when the
    existing partitioned store was reused without changes.
    """
    source_path = Path(input_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    final_sample_dir = _sample_dir(output_root, sample_name)
    if _can_reuse_sample(source_path, final_sample_dir, row_group_size):
        return False

    temp_root = output_root.parent / f".{output_root.name}.{sample_name}.tmp"
    if temp_root.exists():
        shutil.rmtree(temp_root)

    try:
        convert_sample(
            input_path,
            sample_name,
            str(temp_root),
            row_group_size=row_group_size,
            context=context,
            reference_fasta=reference_fasta,
        )
        temp_sample_dir = _sample_dir(temp_root, sample_name)
        chroms = _expected_chrom_dirs(temp_sample_dir)
        manifest = _SampleManifest(
            sample_name=sample_name,
            source=_file_signature(source_path),
            chroms=chroms,
            row_group_size=row_group_size,
        )
        _write_json(_manifest_path(temp_sample_dir), _manifest_payload(manifest))
        _promote_sample_dir(temp_sample_dir, final_sample_dir)
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root)

    return True


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Convert Bismark .cov to partitioned Parquet"
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--context",
        default="CpG",
        choices=["CpG", "CHG", "CHH"],
        help="Methylation context (default: CpG)",
    )
    ap.add_argument(
        "--reference-fasta",
        default=None,
        help="Optional reference FASTA for strand inference (requires pyfaidx)",
    )
    args = ap.parse_args()
    convert_sample(
        args.input,
        args.sample_id,
        args.output_dir,
        context=args.context,
        reference_fasta=args.reference_fasta,
    )
