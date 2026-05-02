"""Scratch script for testing the current methylation workflow on the small samplesheet.

This script mirrors the current CLI flow:
1. Convert each sample in small_samplesheet.csv into a partitioned Parquet store
2. Filter the store with the current coverage thresholds
3. Print per-sample summaries
4. Run DMC between the control and cd55 groups

Outputs are written into local scratch directories so the script can be rerun safely.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
import polars as pl

from epykit.convert import ensure_converted_sample
from epykit.dmc import apply_multiple_testing_correction, process_chromosomes_dmc
from epykit.filter import filter_sites, sample_summary


ROOT = Path(__file__).resolve().parent
SAMPLE_SHEET = ROOT / "samplesheet.csv"
RAW_STORE = ROOT / "scratch_store"
FILTERED_STORE = ROOT / "scratch_store_filtered"
DMC_OUTPUT = ROOT / "scratch_dmc.parquet"


def _reset_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _read_samplesheet(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = _read_samplesheet(SAMPLE_SHEET)
    if not rows:
        raise ValueError(f"No rows found in {SAMPLE_SHEET}")

    DMC_OUTPUT.unlink(missing_ok=True)
    _reset_path(FILTERED_STORE)

    print(f"Preparing {len(rows)} samples from {SAMPLE_SHEET.name}...")
    for row in rows:
        input_path = (ROOT / row["path"]).resolve()
        sample_id = row["sample_id"]
        converted = ensure_converted_sample(str(input_path), sample_id, str(RAW_STORE))
        if converted:
            print(f"  {sample_id}: converted")
        else:
            print(f"  {sample_id}: using cached conversion")

    print(f"Filtering converted store into {FILTERED_STORE.name}...")
    filter_sites(str(RAW_STORE), str(FILTERED_STORE), min_coverage=10)

    print("Sample summaries:")
    for row in rows:
        summary = sample_summary(str(FILTERED_STORE), row["sample_id"])
        print(f"\n{row['sample_id']}")
        print(summary)

    samples_by_group: dict[str, list[str]] = {}
    for row in rows:
        samples_by_group.setdefault(row["group"], []).append(row["sample_id"])

    control_samples = samples_by_group.get("control", [])
    treatment_samples = samples_by_group.get("cd55", [])
    if not control_samples or not treatment_samples:
        raise ValueError(
            f"Expected control and cd55 groups in {SAMPLE_SHEET.name}, got {samples_by_group}"
        )

    print("Running DMC...")
    results = process_chromosomes_dmc(
        str(FILTERED_STORE),
        treatment_samples,
        control_samples,
        test="fisher",
        unite=True,
    )
    results = apply_multiple_testing_correction(results, method="fdr_bh")
    results.write_parquet(str(DMC_OUTPUT))

    sig = results.filter(pl.col('qvalue') < 0.05)
    sig.write_csv('scratch_dmc.sig.csv')


    print(f"DMC results written to {DMC_OUTPUT}")
    print(results.head(10))


if __name__ == "__main__":
    main()