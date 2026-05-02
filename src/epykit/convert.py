import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import polars as pl


RAW_MANIFEST_NAME = ".epykit_raw_manifest.json"


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

    return all((sample_dir / chrom / "part-0.parquet").exists() for chrom in chroms)


def _manifest_payload(manifest: _SampleManifest) -> dict[str, object]:
    return {
        "sample_name": manifest.sample_name,
        "source": manifest.source,
        "chroms": manifest.chroms,
        "row_group_size": manifest.row_group_size,
    }


def _can_reuse_sample(input_path: Path, sample_dir: Path, row_group_size: int) -> bool:
    manifest = _load_json(_manifest_path(sample_dir))
    if not manifest:
        return False

    if manifest.get("source") != _file_signature(input_path):
        return False

    if manifest.get("row_group_size") != row_group_size:
        return False

    chroms = manifest.get("chroms")
    if not isinstance(chroms, list) or not all(isinstance(chrom, str) for chrom in chroms):
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


def convert_sample(input_path: str, sample_name: str, output_dir: str, row_group_size: int = 1_000_000):
    """Convert a Bismark .cov (possibly gzipped) file into a partitioned Parquet store.

    Parameters
    - input_path: path to the .cov or .cov.gz file
    - sample_name: sample identifier to write into `sample` column
    - output_dir: directory where Parquet partitions will be written
    - row_group_size: approximate row-group size for Parquet writer
    """
    p = Path(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # columns expected from Bismark .cov: chr, start, end, methylation_percent, count_methylated, count_unmethylated
    cols = ["chrom", "start", "end", "methyl_percent", "N_meth", "N_unmeth"]

    # Use Polars lazy CSV scanner for streaming conversion. Collect to memory
    # and write per-sample/per-chrom Parquet files because `sink_parquet`
    # doesn't expose `partition_by` in this Polars version.
    lf = pl.scan_csv(
        str(p),
        separator="\t",
        has_header=False,
        new_columns=cols,
        schema_overrides={
            "chrom": pl.Utf8,
            "start": pl.Int32,
            "end": pl.Int32,
            "methyl_percent": pl.Float32,
            "N_meth": pl.Int32,
            "N_unmeth": pl.Int32,
        },
    ).with_columns([
        (pl.col("N_meth") + pl.col("N_unmeth")).alias("coverage"),
        pl.lit(sample_name).alias("sample"),
        pl.col("start").alias("pos"),
        pl.lit("*").alias("strand"),
    ]).select(["chrom", "pos", "strand", "N_meth", "N_unmeth", "coverage", "sample"])

    df = lf.collect()

    # Write one Parquet file per chromosome under sample=<sample>/chrom=<chrom>/
    for chrom in df.select("chrom").unique().to_series().to_list():
        sub = df.filter(pl.col("chrom") == chrom)
        part_dir = out / f"sample={sample_name}" / f"chrom={chrom}"
        part_dir.mkdir(parents=True, exist_ok=True)
        out_path = part_dir / "part-0.parquet"
        sub.write_parquet(
            str(out_path),
            compression="zstd",
            row_group_size=row_group_size,
        )


def ensure_converted_sample(
    input_path: str,
    sample_name: str,
    output_dir: str,
    row_group_size: int = 1_000_000,
) -> bool:
    """Convert a sample unless a valid on-disk conversion already exists.

    Returns True when a conversion was performed, False when the existing
    partitioned store was reused.
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
        convert_sample(input_path, sample_name, str(temp_root), row_group_size=row_group_size)
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
    # Basic test-run when executed directly
    import argparse

    ap = argparse.ArgumentParser(description="Convert Bismark .cov to partitioned Parquet")
    ap.add_argument("--input", required=True)
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    convert_sample(args.input, args.sample_id, args.output_dir)
