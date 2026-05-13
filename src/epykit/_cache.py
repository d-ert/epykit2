"""Shared on-disk cache manifest helpers.

Each pipeline step (convert, filter, normalize) writes a JSON manifest
alongside its Parquet output. Subsequent runs compare the recorded
source signature + params against the current inputs; if they match
and the expected Parquet partitions still exist, the step is skipped
(logged as "cached").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def file_signature(path: Path) -> dict[str, Any]:
    """Path + size + mtime fingerprint of a single file."""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sample_dir(output_dir: Path, sample_name: str) -> Path:
    return output_dir / f"sample={sample_name}"


def expected_chrom_dirs(sd: Path) -> list[str]:
    return sorted(item.name for item in sd.glob("chrom=*") if item.is_dir())


def sample_is_complete(sd: Path, chroms: list[str]) -> bool:
    """True iff `sd` has exactly the expected chrom dirs and each has part-0."""
    if not sd.exists():
        return False
    if expected_chrom_dirs(sd) != sorted(chroms):
        return False
    return all((sd / chrom / "part-0.parquet").exists() for chrom in chroms)


def upstream_sample_signature(input_sample_dir: Path) -> dict[str, Any]:
    """Fingerprint of an upstream sample directory in a partitioned store.

    Prefers any existing pipeline manifest inside the dir (raw / filtered /
    normalized) — its content already captures the upstream lineage cheaply.
    Falls back to the on-disk chrom partition listing if no manifest exists.
    """
    for name in (
        ".epykit_raw_manifest.json",
        ".epykit_filter_manifest.json",
        ".epykit_normalize_manifest.json",
    ):
        mp = input_sample_dir / name
        if mp.exists():
            return {"manifest": name, "content": load_json(mp)}

    chroms = expected_chrom_dirs(input_sample_dir)
    return {
        "manifest": None,
        "chroms": chroms,
        "parts": {
            chrom: [
                file_signature(p)
                for p in sorted((input_sample_dir / chrom).glob("part-*.parquet"))
            ]
            for chrom in chroms
        },
    }
