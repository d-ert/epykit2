"""Scratch script for exercising the full epykit workflow on local sample data.

This script intentionally covers the newer package additions:
1. Convert + filter + per-sample summaries
2. DMC with both fisher and beta_binomial tests
3. DMR calling with overlapping sliding windows
4. BSmooth-style methylation smoothing
5. Feature + CpG-island annotation
6. QC reporting, including CHH-based bisulfite conversion rate

All outputs are written to local scratch files/directories so reruns are safe.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import polars as pl

from epykit.annotate import annotate_cpg_islands, annotate_features
from epykit.convert import ensure_converted_sample
from epykit.dmc import apply_multiple_testing_correction, process_chromosomes_dmc
from epykit.dmr import call_dmr_sliding_window, smooth_methylation_bsmooth
from epykit.filter import filter_sites, sample_summary
from epykit.qc import (
    bisulfite_conversion_rate,
    coverage_uniformity,
    global_methylation_report,
)


ROOT = Path(__file__).resolve().parent

# --- Annotation Files (Set your actual paths here) ---
GTF_PATH = ROOT / "raw_data/gencode.v49.chr_patch_hapl_scaff.annotation.gtf"
BED_PATH = ROOT / "raw_data/hg38_cpg_islands.bed"

SAMPLE_SHEET = ROOT / "samplesheet.csv"
RAW_STORE = ROOT / "scratch_store"
FILTERED_STORE = ROOT / "scratch_store_filtered"

DMC_FISHER_OUTPUT = ROOT / "scratch_dmc.fisher.parquet"
DMC_BB_OUTPUT = ROOT / "scratch_dmc.beta_binomial.parquet"

DMC_FISHER_SIG_OUTPUT = ROOT / "scratch_dmc.fisher.sig.csv"
DMC_BB_SIG_OUTPUT = ROOT / "scratch_dmc.beta_binomial.sig.csv"

DMR_OUTPUT = ROOT / "scratch_dmr.parquet"
SMOOTH_OUTPUT = ROOT / "scratch_smooth.parquet"

DMC_ANNOTATED_OUTPUT = ROOT / "scratch_dmc.annotated.parquet"
DMR_ANNOTATED_OUTPUT = ROOT / "scratch_dmr.annotated.parquet"

QC_GLOBAL_OUTPUT = ROOT / "scratch_qc.global.parquet"
QC_COVERAGE_OUTPUT = ROOT / "scratch_qc.coverage.parquet"

CHH_STORE = ROOT / "scratch_chh_store"


def _reset_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _read_samplesheet(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _prepare_synthetic_chh_store(sample_id: str) -> None:
    """Create a tiny CHH-context store so conversion-rate QC can be exercised."""
    _reset_path(CHH_STORE)
    chh_part = CHH_STORE / f"sample={sample_id}" / "chrom=chrSynthetic"
    chh_part.mkdir(parents=True, exist_ok=True)

    # 0.33% methylation -> expected conversion rate ~99.67%
    chh_df = pl.DataFrame(
        {
            "pos":[1, 2, 3, 4, 5],
            "N_meth":[0, 0, 1, 0, 0],
            "coverage":[60, 60, 60, 60, 60],
        }
    )
    chh_df.write_parquet(str(chh_part / "part-0.parquet"))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="scratch.py",
        description="End-to-end scratch workflow for epykit.",
    )
    ap.add_argument(
        "--samplesheet",
        default=str(SAMPLE_SHEET),
        help="Path to samplesheet CSV (default: samplesheet.csv)",
    )
    ap.add_argument(
        "--gtf",
        default=str(GTF_PATH),
        help="Path to a real GTF/GFF3 file for annotate_features. If not found, function is skipped.",
    )
    ap.add_argument(
        "--cpg-islands-bed",
        dest="cpg_islands_bed",
        default=str(BED_PATH),
        help="Path to a real CpG-islands BED file for annotate_cpg_islands. If not found, function is skipped.",
    )
    ap.add_argument(
        "--no-annotate",
        action="store_true",
        help="Skip Phase 3 annotation entirely.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    samplesheet_path = Path(args.samplesheet)
    rows = _read_samplesheet(samplesheet_path)
    if not rows:
        raise ValueError(f"No rows found in {samplesheet_path}")

    # Reset scratch outputs that should be regenerated every run.
    _reset_path(FILTERED_STORE)
    _reset_path(CHH_STORE)

    for out_path in[
        DMC_FISHER_OUTPUT,
        DMC_BB_OUTPUT,
        DMC_FISHER_SIG_OUTPUT,
        DMC_BB_SIG_OUTPUT,
        DMR_OUTPUT,
        SMOOTH_OUTPUT,
        DMC_ANNOTATED_OUTPUT,
        DMR_ANNOTATED_OUTPUT,
        QC_GLOBAL_OUTPUT,
        QC_COVERAGE_OUTPUT,
    ]:
        out_path.unlink(missing_ok=True)

    print(f"Preparing {len(rows)} samples from {samplesheet_path.name}...")
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

    control_samples = samples_by_group.get("control",[])
    treatment_samples = samples_by_group.get("cd55",[])
    if not control_samples or not treatment_samples:
        raise ValueError(
            f"Expected control and cd55 groups in {samplesheet_path.name}, got {samples_by_group}"
        )

    print("\nRunning DMC (fisher)...")
    dmc_fisher = process_chromosomes_dmc(
        str(FILTERED_STORE),
        treatment_samples,
        control_samples,
        test="fisher",
        unite=True,
    )
    dmc_fisher = apply_multiple_testing_correction(dmc_fisher, method="fdr_bh")
    dmc_fisher.write_parquet(str(DMC_FISHER_OUTPUT))
    dmc_fisher.filter(pl.col("qvalue") < 0.05).write_csv(str(DMC_FISHER_SIG_OUTPUT))
    print(f"DMC (fisher) results written to {DMC_FISHER_OUTPUT.name}")
    print(dmc_fisher.head(10))

    print("\nRunning DMC (beta_binomial / mom)...")
    dmc_bb = process_chromosomes_dmc(
        str(FILTERED_STORE),
        treatment_samples,
        control_samples,
        test="beta_binomial",
        unite=True,
    )
    dmc_bb = apply_multiple_testing_correction(dmc_bb, method="fdr_bh")
    dmc_bb.write_parquet(str(DMC_BB_OUTPUT))
    dmc_bb.filter(pl.col("qvalue") < 0.05).write_csv(str(DMC_BB_SIG_OUTPUT))
    print(f"DMC (beta_binomial) results written to {DMC_BB_OUTPUT.name}")
    print(dmc_bb.head(10))

    print("\nRunning DMR calling from fisher DMC results...")
    dmr_results = call_dmr_sliding_window(
        dmc_fisher,
        window_bp=500,
        step_bp=250,
        min_cpgs=5,
        min_sites_significant=3,
        alpha=0.05,
        min_abs_meth_diff=0.1,
    )
    dmr_results.write_parquet(str(DMR_OUTPUT))
    print(f"DMR results written to {DMR_OUTPUT.name}; total={len(dmr_results):,}")
    if len(dmr_results) > 0:
        print(dmr_results.head(10))

    print("\nRunning BSmooth-style smoothing (subset of samples)...")
    smooth_samples = [r["sample_id"] for r in rows[: min(3, len(rows))]]
    smooth_df = smooth_methylation_bsmooth(
        str(FILTERED_STORE),
        smooth_samples,
        bandwidth=1000,
    )
    smooth_df.write_parquet(str(SMOOTH_OUTPUT))
    print(
        f"Smoothed beta output written to {SMOOTH_OUTPUT.name}; "
        f"rows={len(smooth_df):,}"
    )

    if args.no_annotate:
        print("\nSkipping annotation (--no-annotate).")
    else:
        gtf_path = Path(args.gtf)
        bed_path = Path(args.cpg_islands_bed)
        
        print("\nAnnotating DMC/DMR...")
        dmc_annot = dmc_fisher.head(min(20_000, len(dmc_fisher)))

        # GTF features annotation
        if gtf_path.exists():
            print(f"  Annotating features with GTF: {gtf_path.name}")
            dmc_annot = annotate_features(
                dmc_annot,
                annotation_gtf=str(gtf_path),
                promoter_upstream_bp=2_000,
                promoter_downstream_bp=200,
            )
        else:
            print(f"  [Skipped] GTF not found at {gtf_path}")

        # BED CpG Islands annotation
        if bed_path.exists():
            print(f"  Annotating CpG islands with BED: {bed_path.name}")
            dmc_annot = annotate_cpg_islands(dmc_annot, cpg_island_bed=str(bed_path))
        else:
            print(f"  [Skipped] BED not found at {bed_path}")

        dmc_annot.write_parquet(str(DMC_ANNOTATED_OUTPUT))
        print(f"Annotated DMC written to {DMC_ANNOTATED_OUTPUT.name}")

        # Safely print available columns
        cols_to_print = [c for c in["chrom", "pos", "gene_id", "feature_type", "cpg_context"] if c in dmc_annot.columns]
        print(dmc_annot.select(cols_to_print).head(10))

        if len(dmr_results) > 0:
            dmr_annot = dmr_results
            if gtf_path.exists():
                dmr_annot = annotate_features(
                    dmr_annot,
                    annotation_gtf=str(gtf_path),
                    promoter_upstream_bp=2_000,
                    promoter_downstream_bp=200,
                )
            if bed_path.exists():
                dmr_annot = annotate_cpg_islands(dmr_annot, cpg_island_bed=str(bed_path))
                
            dmr_annot.write_parquet(str(DMR_ANNOTATED_OUTPUT))
            print(f"Annotated DMR written to {DMR_ANNOTATED_OUTPUT.name}")

    print("\nRunning QC reports...")
    all_samples = [r["sample_id"] for r in rows]
    global_qc = global_methylation_report(str(FILTERED_STORE), all_samples)
    global_qc.write_parquet(str(QC_GLOBAL_OUTPUT))
    print(f"Global methylation QC written to {QC_GLOBAL_OUTPUT.name}")
    print(global_qc)

    cov_frames: list[pl.DataFrame] =[]
    for sample_id in all_samples:
        cov_df = coverage_uniformity(str(FILTERED_STORE), sample_id)
        cov_frames.append(cov_df)
    if cov_frames:
        coverage_qc = pl.concat(cov_frames)
        coverage_qc.write_parquet(str(QC_COVERAGE_OUTPUT))
        print(f"Coverage QC written to {QC_COVERAGE_OUTPUT.name}")
        print(coverage_qc.head(20))

    # Exercise bisulfite conversion estimator with a tiny synthetic CHH store.
    chh_sample = all_samples[0]
    _prepare_synthetic_chh_store(chh_sample)
    conv_rate = bisulfite_conversion_rate(
        str(FILTERED_STORE),
        sample=chh_sample,
        chh_context_store=str(CHH_STORE),
    )
    print(
        f"Bisulfite conversion rate ({chh_sample}, synthetic CHH store): "
        f"{conv_rate:.4%}"
    )

    print("\nScratch workflow complete.")


if __name__ == "__main__":
    main()