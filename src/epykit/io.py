from __future__ import annotations

import csv
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from .convert import ensure_converted_sample
from .methyldata import MethylData

logger = logging.getLogger(__name__)


def _default_convert_workers() -> int:
    """Default thread count for parallel sample conversion.

    Sample conversions are independent and dominated by Polars CSV decode +
    gzip + Parquet write, all of which release the GIL. A modest pool size
    (half the logical cores, capped at 8) keeps disk contention manageable
    on typical workstations and HPC nodes alike. Override with the env var
    ``EPYKIT_CONVERT_WORKERS`` (set to ``1`` to disable parallelism).
    """
    override = os.environ.get("EPYKIT_CONVERT_WORKERS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    cpu = os.cpu_count() or 1
    return max(1, min(8, cpu // 2 or 1))


def _convert_samples_parallel(
    files: list[tuple[str, str]],
    cache_store_dir: str,
    context: str,
    reference_fasta: str | None,
) -> None:
    """Run ``ensure_converted_sample`` for many samples concurrently.

    Logs per-sample status as each future completes, preserving the
    informational output of the previous serial loop. Falls back to a
    serial loop when ``EPYKIT_CONVERT_WORKERS=1`` so the path is easy to
    bisect when diagnosing issues.
    """
    workers = _default_convert_workers()
    if workers == 1 or len(files) <= 1:
        for path, sample_id in files:
            converted = ensure_converted_sample(
                path, sample_id, cache_store_dir,
                context=context, reference_fasta=reference_fasta,
            )
            logger.info("  %s: %s", sample_id,
                        "converted" if converted else "cached")
        return

    logger.info("Converting %d sample(s) with %d worker thread(s)",
                len(files), workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_id = {
            pool.submit(
                ensure_converted_sample,
                path, sample_id, cache_store_dir,
                context=context, reference_fasta=reference_fasta,
            ): sample_id
            for path, sample_id in files
        }
        for fut in as_completed(future_to_id):
            sample_id = future_to_id[fut]
            try:
                converted = fut.result()
            except Exception:
                logger.exception("  %s: conversion failed", sample_id)
                raise
            logger.info("  %s: %s", sample_id,
                        "converted" if converted else "cached")


def _count_store_rows(store_dir: str) -> int | None:
    try:
        import pyarrow.parquet as pq

        total = 0
        for path in Path(store_dir).rglob("part-*.parquet"):
            total += pq.read_metadata(str(path)).num_rows
        return total
    except Exception:
        return None


def read_bismark(
    samplesheet: str,
    treatment_group: str,
    control_group: str,
    assembly: str = "unknown",
    store_dir: str = "methyl_store",
    context: str = "CpG",
    reference_fasta: str | None = None,
) -> MethylData:
    """Read a samplesheet and create a MethylData analysis object.

    Expected samplesheet columns: sample_id, group, path
    """
    with open(samplesheet) as handle:
        rows = list(csv.DictReader(handle))

    required = {"sample_id", "group", "path"}
    if not rows:
        raise ValueError("samplesheet contains no rows")
    missing_cols = required - set(rows[0].keys())
    if missing_cols:
        raise ValueError(f"samplesheet missing required columns: {sorted(missing_cols)}")

    obs_rows: list[dict] = []
    files: list[tuple[str, str]] = []
    for row in rows:
        group = row["group"]
        if group not in (treatment_group, control_group):
            continue
        treatment = 1 if group == treatment_group else 0
        obs_row = {
            "sample_id": row["sample_id"],
            "group": group,
            "treatment": treatment,
            "path": row["path"],
        }
        for key, value in row.items():
            if key not in {"sample_id", "group", "path"}:
                obs_row[key] = value
        obs_rows.append(obs_row)
        files.append((row["path"], row["sample_id"]))

    if not obs_rows:
        raise ValueError(
            "No samples matched treatment/control groups from samplesheet. "
            f"treatment_group={treatment_group}, control_group={control_group}"
        )

    # Organize stores under a .cache subdirectory for a cleaner output layout
    analysis_root = Path(store_dir)
    cache_store_dir = str(analysis_root / ".cache" / "raw")
    
    Path(cache_store_dir).mkdir(parents=True, exist_ok=True)
    _convert_samples_parallel(
        files, cache_store_dir, context, reference_fasta,
    )

    n_sites_raw = _count_store_rows(cache_store_dir)
    uns = {
        "samplesheet": samplesheet,
        "pipeline": "bismark",
        "epykit_version": "0.1.0",
    }
    if n_sites_raw is not None:
        uns["n_sites_raw"] = n_sites_raw
    uns["_store_history"] = [
        {"step": "raw", "path": cache_store_dir, "n_sites": n_sites_raw}
    ]

    md = MethylData(
        obs=pl.DataFrame(obs_rows),
        store=cache_store_dir,
        assembly=assembly,
        context=context,
        uns=uns,
    )
    md._analysis_root = str(analysis_root)
    return md


def load(path: str) -> MethylData:
    """Load a previously saved MethylData analysis directory."""
    return MethylData.load(path)


def _candidate_sample_ids_from_filename(name: str) -> list[str]:
    candidates = {name}
    suffixes = [
        ".deduplicated.bismark.cov.gz",
        ".bismark.cov.gz",
        ".deduplicated.cov.gz",
        ".cov.gz",
        ".deduplicated.bismark.cov",
        ".bismark.cov",
        ".deduplicated.cov",
        ".cov",
    ]
    for suf in suffixes:
        if name.endswith(suf):
            candidates.add(name[: -len(suf)])
    if "." in name:
        candidates.add(name.split(".")[0])
    return sorted(candidates, key=len, reverse=True)


def read_nfcore_methylseq(
    run_dir: str,
    treatment_group: str,
    control_group: str,
    assembly: str = "unknown",
    store_dir: str = "methyl_store",
    context: str = "CpG",
    samplesheet_name: str = "samplesheet.csv",
) -> MethylData:
    """Load methylation data directly from an nf-core/methylseq run directory."""
    run = Path(run_dir).resolve()
    cov_dir = run / "results" / "bismark" / "deduplicated"
    samplesheet_path = run / samplesheet_name

    if not cov_dir.exists():
        raise FileNotFoundError(
            f"Expected nf-core/methylseq bismark directory at: {cov_dir}"
        )
    if not samplesheet_path.exists():
        raise FileNotFoundError(
            f"Expected samplesheet at: {samplesheet_path}"
        )

    cov_files = sorted(cov_dir.glob("*.cov.gz")) + sorted(cov_dir.glob("*.cov"))
    if not cov_files:
        raise FileNotFoundError(f"No .cov/.cov.gz files found in {cov_dir}")

    sample_to_cov: dict[str, str] = {}
    for p in cov_files:
        for candidate in _candidate_sample_ids_from_filename(p.name):
            sample_to_cov.setdefault(candidate, str(p))

    with open(samplesheet_path) as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Samplesheet '{samplesheet_path}' is empty")

    if "sample_id" not in rows[0].keys() or "group" not in rows[0].keys():
        raise ValueError(
            "nf-core samplesheet requires at least 'sample_id' and 'group' columns"
        )

    obs_rows: list[dict] = []
    convert_files: list[tuple[str, str]] = []
    # Organize stores under a .cache subdirectory for a cleaner output layout
    analysis_root = Path(store_dir)
    cache_store_dir = str(analysis_root / ".cache" / "raw")

    Path(cache_store_dir).mkdir(parents=True, exist_ok=True)
    for row in rows:
        group = row["group"]
        if group not in (treatment_group, control_group):
            continue

        sample_id = row["sample_id"]
        cov_path = sample_to_cov.get(sample_id)
        if cov_path is None:
            raise FileNotFoundError(
                f"Could not match sample '{sample_id}' to .cov file in {cov_dir}. "
                f"Detected candidates: {sorted(sample_to_cov.keys())[:15]}"
            )

        obs_row = {
            "sample_id": sample_id,
            "group": group,
            "treatment": 1 if group == treatment_group else 0,
            "path": cov_path,
        }
        for key, value in row.items():
            if key not in {"sample_id", "group"}:
                obs_row[key] = value
        obs_rows.append(obs_row)

        logger.info("  %s (%s) <- %s", sample_id, group, cov_path)
        convert_files.append((cov_path, sample_id))

    if not obs_rows:
        raise ValueError(
            "No samples matched treatment/control groups from nf-core samplesheet. "
            f"treatment_group={treatment_group}, control_group={control_group}"
        )

    _convert_samples_parallel(
        convert_files, cache_store_dir, context, reference_fasta=None,
    )

    n_sites_raw = _count_store_rows(cache_store_dir)
    uns = {
        "pipeline": "nf-core/methylseq",
        "nfcore_run": str(run),
        "samplesheet": str(samplesheet_path),
        "epykit_version": "0.1.0",
        "n_sites_raw": n_sites_raw,
        "_store_history": [{"step": "raw", "path": cache_store_dir, "n_sites": n_sites_raw}],
    }
    md = MethylData(
        obs=pl.DataFrame(obs_rows),
        store=cache_store_dir,
        assembly=assembly,
        context=context,
        uns=uns,
    )
    md._analysis_root = str(analysis_root)
    return md
