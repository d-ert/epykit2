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

    # Parse treatment/control sample lists
    with open(args.samplesheet) as f:
        reader = csv.DictReader(f)
        samples_by_group = {}
        for row in reader:
            group = row["group"]
            sample_id = row["sample_id"]
            if group not in samples_by_group:
                samples_by_group[group] = []
            samples_by_group[group].append(sample_id)

    treatment_samples = samples_by_group.get(args.treatment_group)
    control_samples = samples_by_group.get(args.control_group)

    if not treatment_samples:
        raise ValueError(f"No samples found for group '{args.treatment_group}'")
    if not control_samples:
        raise ValueError(f"No samples found for group '{args.control_group}'")

    print(f"Treatment samples: {treatment_samples}")
    print(f"Control samples: {control_samples}")

    # Run DMC
    results = dmc.process_chromosomes_dmc(
        args.methylstore,
        treatment_samples,
        control_samples,
        test="fisher",
        unite=args.unite,
    )

    # Apply multiple testing correction
    results = dmc.apply_multiple_testing_correction(results, method="fdr_bh")

    # Write results
    results.write_parquet(args.output)
    print(f"DMC results written to {args.output}")


def main():
    ap = argparse.ArgumentParser(prog="epykit", description="Methylation Parquet store tools")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Convert subcommand
    p_conv = sub.add_parser("convert", help="Convert a Bismark .cov file to Parquet")
    p_conv.add_argument("--input", required=True, help="Input Bismark .cov or .cov.gz file")
    p_conv.add_argument("--sample-id", required=True, help="Sample identifier")
    p_conv.add_argument("--output-dir", required=True, help="Output Parquet directory")
    p_conv.add_argument(
        "--context",
        choices=["CpG", "CHG", "CHH"],
        default="CpG",
        help="Cytosine context (default: CpG)",
    )
    p_conv.add_argument(
        "--reference-fasta",
        help="Optional reference FASTA for strand inference",
    )
    p_conv.add_argument(
        "--merge-cpg",
        dest="merge_cpg",
        action="store_true",
        help="Merge CpG dyad pairs into canonical sites",
    )
    p_conv.add_argument(
        "--no-merge-cpg",
        dest="merge_cpg",
        action="store_false",
        help="Disable CpG dyad merging",
    )
    p_conv.set_defaults(merge_cpg=None)
    p_conv.set_defaults(func=_cmd_convert)

    # Filter subcommand
    p_filt = sub.add_parser("filter", help="Filter low-coverage CpGs from Parquet store")
    p_filt.add_argument("--methylstore", required=True, help="Input Parquet methylstore")
    p_filt.add_argument("--output-dir", required=True, help="Output Parquet directory")
    p_filt.add_argument(
        "--min-coverage", type=int, default=10, help="Minimum coverage threshold (default: 10)"
    )
    p_filt.add_argument(
        "--max-coverage-quantile",
        type=float,
        default=0.999,
        help="Quantile for max coverage (default: 0.999)",
    )
    p_filt.add_argument(
        "--blacklist-bed", help="Optional BED file with regions to exclude"
    )
    p_filt.add_argument("--sample", help="Filter only this sample (default: all)")
    p_filt.set_defaults(func=_cmd_filter)

    # Summary subcommand
    p_sum = sub.add_parser("summary", help="Compute summary statistics for a sample")
    p_sum.add_argument("--methylstore", required=True, help="Parquet methylstore")
    p_sum.add_argument("--sample", required=True, help="Sample identifier")
    p_sum.add_argument("--output", help="Output Parquet file (default: stdout)")
    p_sum.set_defaults(func=_cmd_sample_summary)

    # DMC subcommand
    p_dmc = sub.add_parser("dmc", help="Differential methylation calling")
    p_dmc.add_argument(
        "--methylstore", required=True, help="Filtered Parquet methylstore"
    )
    p_dmc.add_argument(
        "--samplesheet",
        required=True,
        help="CSV with columns: sample_id, group, path",
    )
    p_dmc.add_argument(
        "--treatment-group",
        required=True,
        help="Group name for treatment samples",
    )
    p_dmc.add_argument(
        "--control-group",
        required=True,
        help="Group name for control samples",
    )
    p_dmc.add_argument(
        "--output", required=True, help="Output Parquet file with DMC results"
    )
    p_dmc.add_argument(
        "--unite",
        action="store_true",
        default=True,
        help="Use only sites in all samples (default: True)",
    )
    p_dmc.set_defaults(func=_cmd_dmc)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

