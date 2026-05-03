"""Scratch script for exercising the full epykit workflow on local sample data.

Verbose logging has been added around the annotation phase to help diagnose
OOM / early-termination issues.  Every major step prints:
  - elapsed time
  - row counts / DataFrame shapes
  - current RSS memory (requires psutil; falls back to nothing if absent)
  - full tracebacks on any exception
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import shutil
import sys
import time
import traceback
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

GTF_PATH  = ROOT / "raw_data/gencode.v49.chr_patch_hapl_scaff.annotation.gtf"
BED_PATH  = ROOT / "raw_data/hg38_cpg_islands.bed"

SAMPLE_SHEET = ROOT / "samplesheet.csv"
RAW_STORE    = ROOT / "scratch_store"
FILTERED_STORE = ROOT / "scratch_store_filtered"

DMC_FISHER_OUTPUT    = ROOT / "scratch_dmc.fisher.parquet"
DMC_BB_OUTPUT        = ROOT / "scratch_dmc.beta_binomial.parquet"
DMC_FISHER_SIG_OUTPUT = ROOT / "scratch_dmc.fisher.sig.csv"
DMC_BB_SIG_OUTPUT    = ROOT / "scratch_dmc.beta_binomial.sig.csv"

DMR_OUTPUT    = ROOT / "scratch_dmr.parquet"
SMOOTH_OUTPUT = ROOT / "scratch_smooth.parquet"

DMC_ANNOTATED_OUTPUT = ROOT / "scratch_dmc.annotated.parquet"
DMR_ANNOTATED_OUTPUT = ROOT / "scratch_dmr.annotated.parquet"

QC_GLOBAL_OUTPUT   = ROOT / "scratch_qc.global.parquet"
QC_COVERAGE_OUTPUT = ROOT / "scratch_qc.coverage.parquet"

CHH_STORE = ROOT / "scratch_chh_store"


# ---------------------------------------------------------------------------
# Logging helpers (mirror the ones in annotate.py so output looks consistent)
# ---------------------------------------------------------------------------

def _mem_str() -> str:
    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss
        return f"  [mem {rss / 1e9:.2f} GB RSS]"
    except Exception:
        return ""


def log(msg: str) -> None:
    print(f"[scratch] {msg}{_mem_str()}", flush=True)


def log_df(label: str, df) -> None:
    try:
        rows = len(df)
        if hasattr(df, "estimated_size"):
            mem_mb = df.estimated_size() / 1e6
            log(f"  {label}: {rows:,} rows  {mem_mb:.1f} MB")
        elif hasattr(df, "memory_usage"):
            mem_mb = df.memory_usage(deep=True).sum() / 1e6
            log(f"  {label}: {rows:,} rows  {mem_mb:.1f} MB")
        else:
            log(f"  {label}: {rows:,} rows")
    except Exception:
        log(f"  {label}: (could not measure)")


def section(title: str) -> None:
    log("")
    log("=" * 60)
    log(title)
    log("=" * 60)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _reset_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _read_samplesheet(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _prepare_synthetic_chh_store(sample_id: str) -> None:
    _reset_path(CHH_STORE)
    chh_part = CHH_STORE / f"sample={sample_id}" / "chrom=chrSynthetic"
    chh_part.mkdir(parents=True, exist_ok=True)
    chh_df = pl.DataFrame({
        "pos":      [1, 2, 3, 4, 5],
        "N_meth":   [0, 0, 1, 0, 0],
        "coverage": [60, 60, 60, 60, 60],
    })
    chh_df.write_parquet(str(chh_part / "part-0.parquet"))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="scratch.py")
    ap.add_argument("--samplesheet",      default=str(SAMPLE_SHEET))
    ap.add_argument("--gtf",              default=str(GTF_PATH))
    ap.add_argument("--cpg-islands-bed",  dest="cpg_islands_bed", default=str(BED_PATH))
    ap.add_argument("--no-annotate",      action="store_true")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Check for psutil upfront
    try:
        import psutil
        log(f"psutil available — memory tracking enabled")
    except ImportError:
        log("psutil not installed — memory numbers will be absent.  "
            "Run: pip install psutil")

    samplesheet_path = Path(args.samplesheet)
    rows = _read_samplesheet(samplesheet_path)
    if not rows:
        raise ValueError(f"No rows found in {samplesheet_path}")

    _reset_path(FILTERED_STORE)
    _reset_path(CHH_STORE)
    for out_path in [
        DMC_FISHER_OUTPUT, DMC_BB_OUTPUT,
        DMC_FISHER_SIG_OUTPUT, DMC_BB_SIG_OUTPUT,
        DMR_OUTPUT, SMOOTH_OUTPUT,
        DMC_ANNOTATED_OUTPUT, DMR_ANNOTATED_OUTPUT,
        QC_GLOBAL_OUTPUT, QC_COVERAGE_OUTPUT,
    ]:
        out_path.unlink(missing_ok=True)

    # ---------------------------------------------------------------
    section("Phase 1: Convert + filter")
    # ---------------------------------------------------------------
    log(f"Preparing {len(rows)} samples ...")
    for row in rows:
        input_path = (ROOT / row["path"]).resolve()
        sample_id  = row["sample_id"]
        t0 = time.time()
        converted  = ensure_converted_sample(str(input_path), sample_id, str(RAW_STORE))
        log(f"  {sample_id}: {'converted' if converted else 'cached'}  ({time.time()-t0:.1f}s)")

    log(f"Filtering -> {FILTERED_STORE.name} ...")
    t0 = time.time()
    filter_sites(str(RAW_STORE), str(FILTERED_STORE), min_coverage=10)
    log(f"Filter done in {time.time()-t0:.1f}s")

    for row in rows:
        summary = sample_summary(str(FILTERED_STORE), row["sample_id"])
        log(f"\nSummary for {row['sample_id']}:")
        print(summary, flush=True)

    samples_by_group: dict[str, list[str]] = {}
    for row in rows:
        samples_by_group.setdefault(row["group"], []).append(row["sample_id"])

    control_samples   = samples_by_group.get("control",   [])
    treatment_samples = samples_by_group.get("cd55",      [])
    if not control_samples or not treatment_samples:
        raise ValueError(f"Expected control and cd55 groups; got {samples_by_group}")

    # ---------------------------------------------------------------
    section("Phase 2a: DMC (fisher)")
    # ---------------------------------------------------------------
    t0 = time.time()
    dmc_fisher = process_chromosomes_dmc(
        str(FILTERED_STORE), treatment_samples, control_samples,
        test="fisher", unite=True,
    )
    dmc_fisher = apply_multiple_testing_correction(dmc_fisher, method="fdr_bh")
    dmc_fisher.write_parquet(str(DMC_FISHER_OUTPUT))
    dmc_fisher.filter(pl.col("qvalue") < 0.05).write_csv(str(DMC_FISHER_SIG_OUTPUT))
    log(f"Fisher DMC done in {time.time()-t0:.1f}s")
    log_df("dmc_fisher", dmc_fisher)
    print(dmc_fisher.head(10), flush=True)

    # ---------------------------------------------------------------
    section("Phase 2b: DMC (beta_binomial)")
    # ---------------------------------------------------------------
    t0 = time.time()
    dmc_bb = process_chromosomes_dmc(
        str(FILTERED_STORE), treatment_samples, control_samples,
        test="beta_binomial", unite=True,
    )
    dmc_bb = apply_multiple_testing_correction(dmc_bb, method="fdr_bh")
    dmc_bb.write_parquet(str(DMC_BB_OUTPUT))
    dmc_bb.filter(pl.col("qvalue") < 0.05).write_csv(str(DMC_BB_SIG_OUTPUT))
    log(f"Beta-binomial DMC done in {time.time()-t0:.1f}s")
    log_df("dmc_bb", dmc_bb)

    # ---------------------------------------------------------------
    section("Phase 2c: DMR calling")
    # ---------------------------------------------------------------
    t0 = time.time()
    dmr_results = call_dmr_sliding_window(
        dmc_fisher, window_bp=500, step_bp=250,
        min_cpgs=5, min_sites_significant=3,
        alpha=0.05, min_abs_meth_diff=0.1,
    )
    dmr_results.write_parquet(str(DMR_OUTPUT))
    log(f"DMR done in {time.time()-t0:.1f}s  total={len(dmr_results):,}")
    if len(dmr_results) > 0:
        print(dmr_results.head(10), flush=True)

    # ---------------------------------------------------------------
    section("Phase 2d: BSmooth smoothing")
    # ---------------------------------------------------------------
    t0 = time.time()
    smooth_samples = [r["sample_id"] for r in rows[: min(3, len(rows))]]
    smooth_df = smooth_methylation_bsmooth(
        str(FILTERED_STORE), smooth_samples, bandwidth=1000,
    )
    smooth_df.write_parquet(str(SMOOTH_OUTPUT))
    log(f"Smoothing done in {time.time()-t0:.1f}s  rows={len(smooth_df):,}")

    # ---------------------------------------------------------------
    section("Phase 3: Annotation")
    # ---------------------------------------------------------------
    if args.no_annotate:
        log("Skipping annotation (--no-annotate)")
    else:
        gtf_path = Path(args.gtf)
        bed_path = Path(args.cpg_islands_bed)

        # Subset to first 20k DMC sites for the annotation run
        dmc_annot = dmc_fisher.head(min(20_000, len(dmc_fisher)))
        log(f"Annotation input subset: {len(dmc_annot):,} sites")
        log_df("dmc_annot (input to annotation)", dmc_annot)

        # ---- GTF feature annotation --------------------------------
        if gtf_path.exists():
            log(f"GTF path: {gtf_path}")
            log(f"GTF file size: {gtf_path.stat().st_size / 1e6:.1f} MB")
            log("Starting annotate_features ...")
            t0 = time.time()
            try:
                dmc_annot = annotate_features(
                    dmc_annot,
                    annotation_gtf=str(gtf_path),
                    promoter_upstream_bp=2_000,
                    promoter_downstream_bp=200,
                )
                log(f"annotate_features returned in {time.time()-t0:.1f}s")
                log_df("dmc_annot (after feature annotation)", dmc_annot)
                if "feature_type" in dmc_annot.columns:
                    ft_counts = (
                        dmc_annot.group_by("feature_type")
                        .agg(pl.len().alias("n"))
                        .sort("n", descending=True)
                    )
                    log("Feature type breakdown:")
                    print(ft_counts, flush=True)
            except Exception:
                log(f"ERROR in annotate_features after {time.time()-t0:.1f}s:")
                traceback.print_exc(file=sys.stdout)
                sys.stdout.flush()
                log("Continuing without feature annotation ...")
            finally:
                gc.collect()
        else:
            log(f"GTF not found at {gtf_path} — skipping feature annotation")

        # ---- CpG island annotation ---------------------------------
        if bed_path.exists():
            log(f"BED path: {bed_path}")
            log(f"BED file size: {bed_path.stat().st_size / 1e6:.1f} MB")
            log("Starting annotate_cpg_islands ...")
            t0 = time.time()
            try:
                dmc_annot = annotate_cpg_islands(dmc_annot, cpg_island_bed=str(bed_path))
                log(f"annotate_cpg_islands returned in {time.time()-t0:.1f}s")
                log_df("dmc_annot (after CpG island annotation)", dmc_annot)
                if "cpg_context" in dmc_annot.columns:
                    ctx_counts = (
                        dmc_annot.group_by("cpg_context")
                        .agg(pl.len().alias("n"))
                        .sort("n", descending=True)
                    )
                    log("CpG context breakdown:")
                    print(ctx_counts, flush=True)
            except Exception:
                log(f"ERROR in annotate_cpg_islands after {time.time()-t0:.1f}s:")
                traceback.print_exc(file=sys.stdout)
                sys.stdout.flush()
                log("Continuing without CpG island annotation ...")
            finally:
                gc.collect()
        else:
            log(f"BED not found at {bed_path} — skipping CpG island annotation")

        # ---- Write annotated DMC -----------------------------------
        log(f"Writing annotated DMC -> {DMC_ANNOTATED_OUTPUT.name} ...")
        dmc_annot.write_parquet(str(DMC_ANNOTATED_OUTPUT))
        cols_to_print = [c for c in ["chrom", "pos", "gene_id", "feature_type", "cpg_context"]
                         if c in dmc_annot.columns]
        print(dmc_annot.select(cols_to_print).head(10), flush=True)

        # ---- Annotate DMRs -----------------------------------------
        if len(dmr_results) > 0:
            log(f"Annotating {len(dmr_results):,} DMRs ...")
            dmr_annot = dmr_results

            if gtf_path.exists():
                log("Starting annotate_features (DMRs) ...")
                t0 = time.time()
                try:
                    dmr_annot = annotate_features(
                        dmr_annot,
                        annotation_gtf=str(gtf_path),
                        promoter_upstream_bp=2_000,
                        promoter_downstream_bp=200,
                    )
                    log(f"DMR annotate_features done in {time.time()-t0:.1f}s")
                except Exception:
                    log(f"ERROR in annotate_features (DMRs) after {time.time()-t0:.1f}s:")
                    traceback.print_exc(file=sys.stdout)
                    sys.stdout.flush()
                finally:
                    gc.collect()

            if bed_path.exists():
                log("Starting annotate_cpg_islands (DMRs) ...")
                t0 = time.time()
                try:
                    dmr_annot = annotate_cpg_islands(dmr_annot, cpg_island_bed=str(bed_path))
                    log(f"DMR annotate_cpg_islands done in {time.time()-t0:.1f}s")
                except Exception:
                    log(f"ERROR in annotate_cpg_islands (DMRs) after {time.time()-t0:.1f}s:")
                    traceback.print_exc(file=sys.stdout)
                    sys.stdout.flush()
                finally:
                    gc.collect()

            dmr_annot.write_parquet(str(DMR_ANNOTATED_OUTPUT))
            log(f"Annotated DMR written to {DMR_ANNOTATED_OUTPUT.name}")
        else:
            log("No DMRs to annotate")

    # ---------------------------------------------------------------
    section("Phase 4: QC")
    # ---------------------------------------------------------------
    all_samples = [r["sample_id"] for r in rows]
    t0 = time.time()
    global_qc = global_methylation_report(str(FILTERED_STORE), all_samples)
    global_qc.write_parquet(str(QC_GLOBAL_OUTPUT))
    log(f"Global QC done in {time.time()-t0:.1f}s")
    print(global_qc, flush=True)

    cov_frames: list[pl.DataFrame] = []
    for sample_id in all_samples:
        cov_df = coverage_uniformity(str(FILTERED_STORE), sample_id)
        cov_frames.append(cov_df)
    if cov_frames:
        coverage_qc = pl.concat(cov_frames)
        coverage_qc.write_parquet(str(QC_COVERAGE_OUTPUT))
        log(f"Coverage QC written to {QC_COVERAGE_OUTPUT.name}")
        print(coverage_qc.head(20), flush=True)

    chh_sample = all_samples[0]
    _prepare_synthetic_chh_store(chh_sample)
    conv_rate = bisulfite_conversion_rate(
        str(FILTERED_STORE), sample=chh_sample, chh_context_store=str(CHH_STORE),
    )
    log(f"Bisulfite conversion rate ({chh_sample}): {conv_rate:.4%}")

    section("Scratch workflow COMPLETE")


if __name__ == "__main__":
    main()