import argparse
import logging
from pathlib import Path
from .convert import convert_sample
from . import filter, dmc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _cmd_convert(args: argparse.Namespace):
    """Handler for 'convert' subcommand."""
    convert_sample(
        args.input,
        args.sample_id,
        args.output_dir,
        context=args.context,
        reference_fasta=args.reference_fasta,
        merge_cpg=args.merge_cpg,
    )


def _cmd_filter(args: argparse.Namespace):
    """Handler for 'filter' subcommand."""
    filter.filter_sites(
        args.methylstore,
        args.output_dir,
        min_coverage=args.min_coverage,
        max_coverage_quantile=args.max_coverage_quantile,
        blacklist_bed=args.blacklist_bed,
        sample=args.sample,
    )


def _cmd_sample_summary(args: argparse.Namespace):
    """Handler for 'summary' subcommand."""
    df = filter.sample_summary(args.methylstore, args.sample, output_path=args.output)
    if not args.output:
        print(df)


def _cmd_dmc(args: argparse.Namespace):
    """Handler for 'dmc' subcommand."""
    import csv

    with open(args.samplesheet) as f:
        reader = csv.DictReader(f)
        samples_by_group: dict[str, list[str]] = {}
        for row in reader:
            group     = row["group"]
            sample_id = row["sample_id"]
            samples_by_group.setdefault(group, []).append(sample_id)

    treatment_samples = samples_by_group.get(args.treatment_group)
    control_samples   = samples_by_group.get(args.control_group)

    if not treatment_samples:
        raise ValueError(f"No samples found for group '{args.treatment_group}'")
    if not control_samples:
        raise ValueError(f"No samples found for group '{args.control_group}'")

    print(f"Treatment samples: {treatment_samples}")
    print(f"Control samples:   {control_samples}")

    results = dmc.process_chromosomes_dmc(
        args.methylstore,
        treatment_samples,
        control_samples,
        test=args.test,
        unite=args.unite,
    )
    results = dmc.apply_multiple_testing_correction(results, method="fdr_bh")
    results.write_parquet(args.output)
    print(f"DMC results written to {args.output}")


def _cmd_dmr(args: argparse.Namespace):
    """Handler for 'dmr' subcommand."""
    import polars as pl
    from .dmr import call_dmr_sliding_window

    dmc_results = pl.read_parquet(args.dmc_results)
    dmr_results = call_dmr_sliding_window(
        dmc_results,
        window_bp=args.window_bp,
        step_bp=args.step_bp,
        min_cpgs=args.min_cpgs,
        min_sites_significant=args.min_sites_significant,
        alpha=args.alpha,
        min_abs_meth_diff=args.min_abs_meth_diff,
    )
    dmr_results.write_parquet(args.output)
    print(f"DMR results written to {args.output}")
    print(f"Total DMRs called: {len(dmr_results):,}")
    if len(dmr_results) > 0:
        n_hyper = int((dmr_results["dmr_type"] == "hyper").sum())
        n_hypo  = int((dmr_results["dmr_type"] == "hypo").sum())
        print(f"  Hyper: {n_hyper:,}  Hypo: {n_hypo:,}")
        print(dmr_results.head(10))


def _cmd_annotate(args: argparse.Namespace):
    """Handler for 'annotate' subcommand."""
    import polars as pl

    sites = pl.read_parquet(args.input)

    if args.gtf:
        from .annotate import annotate_features
        sites = annotate_features(
            sites,
            annotation_gtf=args.gtf,
            promoter_upstream_bp=args.promoter_upstream_bp,
            promoter_downstream_bp=args.promoter_downstream_bp,
        )
        print("Gene feature annotation complete.")

    if args.cpg_islands:
        from .annotate import annotate_cpg_islands
        sites = annotate_cpg_islands(sites, cpg_island_bed=args.cpg_islands)
        print("CpG island annotation complete.")

    sites.write_parquet(args.output)
    print(f"Annotated results written to {args.output}")


def _cmd_qc_report(args: argparse.Namespace):
    """Handler for 'qc-report' subcommand."""
    import polars as pl
    from .qc import global_methylation_report, coverage_uniformity

    samples = args.samples.split(",")

    print("=== Global methylation report ===")
    meth_report = global_methylation_report(args.methylstore, samples)
    print(meth_report)
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        meth_report.write_parquet(str(out / "global_methylation.parquet"))

    print("\n=== Coverage uniformity report ===")
    cov_frames: list[pl.DataFrame] = []
    for sample in samples:
        try:
            cov_df = coverage_uniformity(args.methylstore, sample)
            print(f"\n{sample}")
            print(cov_df)
            cov_frames.append(cov_df)
        except ValueError as exc:
            print(f"  Warning: {exc}")

    if cov_frames and args.output_dir:
        combined = pl.concat(cov_frames)
        combined.write_parquet(str(Path(args.output_dir) / "coverage_uniformity.parquet"))
        print(f"\nQC reports written to {args.output_dir}")


def _cmd_smooth(args: argparse.Namespace):
    """Handler for 'smooth' subcommand (BSmooth-style)."""
    from .dmr import smooth_methylation_bsmooth

    samples = args.samples.split(",")
    result  = smooth_methylation_bsmooth(
        args.methylstore, samples, bandwidth=args.bandwidth
    )
    result.write_parquet(args.output)
    print(f"Smoothed betas written to {args.output}")
    print(f"Total sites: {len(result):,}")


def main():
    ap  = argparse.ArgumentParser(
        prog="epykit", description="Methylation Parquet store tools"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ------------------------------------------------------------------
    # convert
    # ------------------------------------------------------------------
    p_conv = sub.add_parser("convert", help="Convert a Bismark .cov file to Parquet")
    p_conv.add_argument("--input",        required=True)
    p_conv.add_argument("--sample-id",    required=True)
    p_conv.add_argument("--output-dir",   required=True)
    p_conv.add_argument(
        "--context", choices=["CpG", "CHG", "CHH"], default="CpG",
    )
    p_conv.add_argument("--reference-fasta")
    p_conv.add_argument("--merge-cpg",    dest="merge_cpg", action="store_true")
    p_conv.add_argument("--no-merge-cpg", dest="merge_cpg", action="store_false")
    p_conv.set_defaults(merge_cpg=None, func=_cmd_convert)

    # ------------------------------------------------------------------
    # filter
    # ------------------------------------------------------------------
    p_filt = sub.add_parser("filter", help="Filter low-coverage CpGs")
    p_filt.add_argument("--methylstore",           required=True)
    p_filt.add_argument("--output-dir",            required=True)
    p_filt.add_argument("--min-coverage",          type=int,   default=10)
    p_filt.add_argument("--max-coverage-quantile", type=float, default=0.999)
    p_filt.add_argument("--blacklist-bed")
    p_filt.add_argument("--sample")
    p_filt.set_defaults(func=_cmd_filter)

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------
    p_sum = sub.add_parser("summary", help="Per-sample summary statistics")
    p_sum.add_argument("--methylstore", required=True)
    p_sum.add_argument("--sample",      required=True)
    p_sum.add_argument("--output")
    p_sum.set_defaults(func=_cmd_sample_summary)

    # ------------------------------------------------------------------
    # dmc
    # ------------------------------------------------------------------
    p_dmc = sub.add_parser("dmc", help="Differential methylation calling (per-CpG)")
    p_dmc.add_argument("--methylstore",      required=True)
    p_dmc.add_argument("--samplesheet",      required=True,
                       help="CSV: sample_id, group, path")
    p_dmc.add_argument("--treatment-group",  required=True)
    p_dmc.add_argument("--control-group",    required=True)
    p_dmc.add_argument("--output",           required=True)
    p_dmc.add_argument(
        "--test",
        choices=["fisher", "beta_binomial"],
        default="fisher",
        help="Statistical test: 'fisher' (default) or 'beta_binomial' (≥6 reps)",
    )
    p_dmc.add_argument(
        "--unite", action="store_true", default=True,
        help="Use only sites covered in all samples (default: True)",
    )
    p_dmc.set_defaults(func=_cmd_dmc)

    # ------------------------------------------------------------------
    # dmr
    # ------------------------------------------------------------------
    p_dmr = sub.add_parser("dmr", help="DMR calling from DMC results")
    p_dmr.add_argument("--dmc-results",          required=True,
                       help="Parquet file from 'epykit dmc'")
    p_dmr.add_argument("--output",               required=True)
    p_dmr.add_argument("--window-bp",            type=int,   default=500)
    p_dmr.add_argument("--step-bp",              type=int,   default=250)
    p_dmr.add_argument("--min-cpgs",             type=int,   default=5)
    p_dmr.add_argument("--min-sites-significant",type=int,   default=3)
    p_dmr.add_argument("--alpha",                type=float, default=0.05)
    p_dmr.add_argument("--min-abs-meth-diff",    type=float, default=0.1)
    p_dmr.set_defaults(func=_cmd_dmr)

    # ------------------------------------------------------------------
    # annotate
    # ------------------------------------------------------------------
    p_ann = sub.add_parser(
        "annotate", help="Annotate DMC/DMR results with genomic features"
    )
    p_ann.add_argument("--input",   required=True,
                       help="Parquet file from 'epykit dmc' or 'epykit dmr'")
    p_ann.add_argument("--output",  required=True)
    p_ann.add_argument(
        "--gtf",
        help="Ensembl/UCSC GTF/GFF3 for gene feature annotation",
    )
    p_ann.add_argument(
        "--cpg-islands",
        help="UCSC CpGIsland BED file for CpG context annotation",
    )
    p_ann.add_argument("--promoter-upstream-bp",   type=int, default=2000)
    p_ann.add_argument("--promoter-downstream-bp", type=int, default=200)
    p_ann.set_defaults(func=_cmd_annotate)

    # ------------------------------------------------------------------
    # qc-report
    # ------------------------------------------------------------------
    p_qc = sub.add_parser("qc-report", help="QC and coverage uniformity report")
    p_qc.add_argument("--methylstore", required=True)
    p_qc.add_argument(
        "--samples", required=True,
        help="Comma-separated list of sample IDs",
    )
    p_qc.add_argument(
        "--output-dir",
        help="Directory for Parquet QC output files (optional)",
    )
    p_qc.set_defaults(func=_cmd_qc_report)

    # ------------------------------------------------------------------
    # smooth
    # ------------------------------------------------------------------
    p_sm = sub.add_parser("smooth", help="BSmooth-style LOESS beta smoothing")
    p_sm.add_argument("--methylstore", required=True)
    p_sm.add_argument(
        "--samples", required=True,
        help="Comma-separated list of sample IDs",
    )
    p_sm.add_argument("--output",    required=True,
                      help="Output Parquet file with smoothed betas")
    p_sm.add_argument("--bandwidth", type=int, default=1000,
                      help="Smoothing bandwidth in bp (default 1000)")
    p_sm.set_defaults(func=_cmd_smooth)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
