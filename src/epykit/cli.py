"""Command-line entry point for epykit.

Default DMC test is ``lr`` everywhere (CLI, Python API, docstrings) — the
quasi-binomial likelihood-ratio chi-square with per-site McCullagh-Nelder
dispersion. This matches methylKit's ``calculateDiffMeth(overdispersion="MN",
test="Chisq")`` and is the recommended path at n >= 2 replicates per group.

Historical note: earlier development iterations defaulted to ``fisher`` (n<6
pooled), then briefly to ``logit_t``, then to ``score``. ``lr`` is now the
single canonical default; the other engines (``score``, ``logit_t``, ``glm``,
``beta_binomial``, ``cmh``, ``fisher``) remain available via ``--test``.

CLI surface:
* ``dmc`` — per-CpG calling with ``--test {lr,score,glm,logit_t,beta_binomial,
  cmh,fisher}``, ``--min-samples-case`` / ``--min-samples-control`` filters,
  and ``--allow-n1`` to opt into the (anti-conservative) Fisher fallback when
  there are fewer than 2 replicates per group.
* ``dmr`` — ``--method {tile,sliding_window}``. The tile path takes a
  methylstore + samplesheet and pools reads per tile; the sliding-window path
  takes a DMC parquet and combines per-CpG p-values.
"""

import argparse
import logging
from pathlib import Path
from .convert import convert_sample
from . import filter, dmc


def _read_samplesheet_groups(samplesheet: str, treatment_group: str, control_group: str):
    import csv

    with open(samplesheet) as f:
        reader = csv.DictReader(f)
        samples_by_group: dict[str, list[str]] = {}
        for row in reader:
            group     = row["group"]
            sample_id = row["sample_id"]
            samples_by_group.setdefault(group, []).append(sample_id)

    treatment_samples = samples_by_group.get(treatment_group)
    control_samples   = samples_by_group.get(control_group)

    if not treatment_samples:
        raise ValueError(f"No samples found for group '{treatment_group}'")
    if not control_samples:
        raise ValueError(f"No samples found for group '{control_group}'")

    return treatment_samples, control_samples


def _cmd_convert(args: argparse.Namespace):
    """Handler for 'convert' subcommand."""
    convert_sample(
        args.input,
        args.sample_id,
        args.output_dir,
        context=args.context,
        reference_fasta=args.reference_fasta,
        merge_strands=args.merge_cpg,  # FIX-9: CLI flag is --merge-cpg; param is merge_strands
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


def _cli_n1_and_footgun_checks(args, unit: str = "sites") -> None:
    """Mirror tl.* guards on the CLI side (B6 + B8)."""
    treatment_samples, control_samples = args._samples  # set by caller
    if min(len(treatment_samples), len(control_samples)) < 2 and not args.allow_n1:
        raise SystemExit(
            f"error: at least 2 replicates per group required "
            f"(treatment={len(treatment_samples)}, control={len(control_samples)}). "
            f"Pass --allow-n1 to opt into the Fisher fallback."
        )
    import warnings
    if args.test == "fisher":
        warnings.warn(
            "test='fisher' is anti-conservative; prefer 'lr' at n >= 2.",
            UserWarning, stacklevel=2,
        )
    if (not args.unite) and args.min_samples_case == 0 and args.min_samples_control == 0:
        warnings.warn(
            f"--no-unite + min_samples_*=0 will test {unit} covered in only "
            f"one sample per group. Recommended: --min-samples-case 2 "
            f"--min-samples-control 2.",
            UserWarning, stacklevel=2,
        )


def _cmd_dmc(args: argparse.Namespace):
    """Handler for 'dmc' subcommand."""
    treatment_samples, control_samples = _read_samplesheet_groups(
        args.samplesheet, args.treatment_group, args.control_group
    )
    args._samples = (treatment_samples, control_samples)
    _cli_n1_and_footgun_checks(args, unit="sites")

    print(f"Treatment samples: {treatment_samples}")
    print(f"Control samples:   {control_samples}")
    print(f"Test:              {args.test}")
    print(f"Unite mode:        {'intersect' if args.unite else 'union'}")
    if args.min_samples_case or args.min_samples_control:
        print(
            f"Per-site guards:   min_samples_case={args.min_samples_case}, "
            f"min_samples_control={args.min_samples_control}"
        )

    results = dmc.process_chromosomes_dmc(
        args.methylstore,
        treatment_samples,
        control_samples,
        test=args.test,
        unite=args.unite,
        min_samples_treatment=args.min_samples_case,
        min_samples_control=args.min_samples_control,
    )
    results = dmc.apply_multiple_testing_correction(results, method="fdr_bh")
    results.write_parquet(args.output)
    print(f"DMC results written to {args.output}")
    n_sig = int((results["qvalue"] < 0.05).sum()) if "qvalue" in results.columns else 0
    print(f"  Total sites tested: {len(results):,}")
    print(f"  Significant (q<0.05): {n_sig:,}")


def _cmd_dmr(args: argparse.Namespace):
    """Handler for 'dmr' subcommand."""
    import polars as pl
    from .dmr import call_dmr_sliding_window, call_dmr_tile_based

    if args.method == "tile":
        # --- BIO-5: tile-based path. Needs methylstore + samplesheet. ---
        if not args.methylstore or not args.samplesheet:
            raise ValueError(
                "method=tile requires --methylstore, --samplesheet, "
                "--treatment-group and --control-group."
            )

        treatment_samples, control_samples = _read_samplesheet_groups(
            args.samplesheet, args.treatment_group, args.control_group
        )
        args._samples = (treatment_samples, control_samples)
        _cli_n1_and_footgun_checks(args, unit="tiles")

        print(f"Treatment samples: {treatment_samples}")
        print(f"Control samples:   {control_samples}")
        print(f"Tile size:         {args.tile_size_bp} bp")
        print(f"Test:              {args.test}")

        dmr_results = call_dmr_tile_based(
            methylstore_path=args.methylstore,
            samples_treatment=treatment_samples,
            samples_control=control_samples,
            tile_size_bp=args.tile_size_bp,
            test=args.test,
            min_cpgs_per_tile=args.min_cpgs_per_tile,
            alpha=args.alpha,
            min_abs_meth_diff=args.min_abs_meth_diff,
            unite=args.unite,
            min_samples_treatment=args.min_samples_case,
            min_samples_control=args.min_samples_control,
        )
    else:
        # --- Legacy sliding-window path: takes a DMC parquet ---
        if not args.dmc_results:
            raise ValueError("method=sliding_window requires --dmc-results.")

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
    if len(dmr_results) > 0 and "dmr_type" in dmr_results.columns:
        n_hyper = int((dmr_results["dmr_type"] == "hyper").sum())
        n_hypo  = int((dmr_results["dmr_type"] == "hypo").sum())
        n_mixed = int((dmr_results["dmr_type"] == "mixed").sum())
        print(f"  Hyper: {n_hyper:,}  Hypo: {n_hypo:,}  Mixed: {n_mixed:,}")
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
    """Handler for 'smooth' subcommand (Gaussian-kernel smoothing)."""
    from .dmr import smooth_methylation_gaussian

    samples = args.samples.split(",")
    smooth_path = args.output
    smooth_methylation_gaussian(
        args.methylstore,
        samples,
        bandwidth=args.bandwidth,
        output_path=smooth_path,
    )
    print(f"Smoothed betas written to {smooth_path}")


def _configure_logging(verbosity: int) -> None:
    """Configure logging only when running as a CLI.

    Library code never calls ``logging.basicConfig``; doing so at import time
    would override the host application's logging configuration. The CLI is
    allowed to configure logging because the user has explicitly invoked it.

    ``verbosity`` is the net of ``-v`` (count) minus ``-q`` (count):
      0  → INFO (default)
      ≥1 → DEBUG
      ≤−1 → WARNING
    """
    if verbosity >= 1:
        level = logging.DEBUG
    elif verbosity <= -1:
        level = logging.WARNING
    else:
        level = logging.INFO
    # Guard against overriding handlers a host program (e.g. tests, notebooks)
    # may already have installed.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    else:
        logging.getLogger().setLevel(level)


def main():
    ap  = argparse.ArgumentParser(
        prog="epykit", description="Methylation Parquet store tools"
    )
    ap.add_argument("-v", "--verbose", action="count", default=0,
                    help="Increase logging verbosity (-v: DEBUG)")
    ap.add_argument("-q", "--quiet", action="count", default=0,
                    help="Decrease logging verbosity (-q: WARNING and above)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # convert
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

    # filter
    p_filt = sub.add_parser("filter", help="Filter low-coverage CpGs")
    p_filt.add_argument("--methylstore",           required=True)
    p_filt.add_argument("--output-dir",            required=True)
    p_filt.add_argument("--min-coverage",          type=int,   default=10)
    p_filt.add_argument("--max-coverage-quantile", type=float, default=0.999)
    p_filt.add_argument("--blacklist-bed")
    p_filt.add_argument("--sample")
    p_filt.set_defaults(func=_cmd_filter)

    # summary
    p_sum = sub.add_parser("summary", help="Per-sample summary statistics")
    p_sum.add_argument("--methylstore", required=True)
    p_sum.add_argument("--sample",      required=True)
    p_sum.add_argument("--output")
    p_sum.set_defaults(func=_cmd_sample_summary)

    # dmc
    p_dmc = sub.add_parser("dmc", help="Differential methylation calling (per-CpG)")
    p_dmc.add_argument("--methylstore",      required=True)
    p_dmc.add_argument("--samplesheet",      required=True,
                       help="CSV: sample_id, group, path")
    p_dmc.add_argument("--treatment-group",  required=True)
    p_dmc.add_argument("--control-group",    required=True)
    p_dmc.add_argument("--output",           required=True)
    p_dmc.add_argument(
        "--test",
        choices=["lr", "score", "glm", "logit_t", "beta_binomial", "cmh", "fisher"],
        default="lr",
        help=(
            "Statistical test (default: lr). "
            "lr — Quasi-binomial likelihood-ratio chi-square with per-site "
            "McCullagh-Nelder dispersion. Matches methylKit's "
            "calculateDiffMeth(overdispersion='MN', test='Chisq'). "
            "Recommended default at n>=2. "
            "score — Quasi-binomial score test on the same dispersion-corrected "
            "accumulators as lr; marginally more powerful but mildly "
            "anti-conservative at the boundaries. "
            "glm — Binomial GLM with covariates (requires a design via the "
            "Python API: ep.tl.dmr(..., design='~ treatment + sex + batch')). "
            "logit_t — Welch t on logit(beta), variance-stabilising fallback. "
            "beta_binomial — Welch t on raw betas (NOT a true beta-binomial "
            "GLM; superseded by lr/score). "
            "cmh — Cochran-Mantel-Haenszel on per-pair strata. "
            "fisher — Fisher exact on reads pooled across replicates "
            "(anti-conservative, kept for backward compatibility; warns)."
        ),
    )
    p_dmc.add_argument(
        "--no-unite", action="store_false", dest="unite", default=True,
        help="Include sites seen in at least one sample (default: only sites in all samples)",
    )
    p_dmc.add_argument(
        "--min-samples-case", type=int, default=0,
        help=(
            "Per-site minimum number of treatment samples with non-zero "
            "coverage. Sites failing the threshold are NaN'd before FDR. "
            "Useful with --no-unite (BIO-7)."
        ),
    )
    p_dmc.add_argument(
        "--min-samples-control", type=int, default=0,
        help="Per-site minimum number of control samples with non-zero coverage.",
    )
    p_dmc.add_argument(
        "--allow-n1", action="store_true", default=False,
        help=(
            "Allow n<2 per group: fall back to Fisher exact on pooled reads. "
            "Default is to refuse — between-replicate variance is ignored "
            "and p-values are anti-conservative under this fallback."
        ),
    )
    p_dmc.set_defaults(func=_cmd_dmc)

    # dmr
    p_dmr = sub.add_parser("dmr", help="DMR calling (tile-based or sliding-window)")
    p_dmr.add_argument(
        "--method",
        choices=["tile", "sliding_window"],
        default="tile",
        help=(
            "DMR algorithm. "
            "'tile' (default, methylKit parity, BIO-5) pools reads across "
            "CpGs within each fixed-size tile and runs one test per tile. "
            "'sliding_window' takes a precomputed DMC parquet and combines "
            "per-CpG p-values with signed Stouffer's Z (legacy)."
        ),
    )
    p_dmr.add_argument("--output", required=True)

    # Tile-method options
    p_dmr.add_argument("--methylstore",
                       help="(tile only) Path to filtered Parquet methylstore.")
    p_dmr.add_argument("--samplesheet",
                       help="(tile only) CSV: sample_id, group, path.")
    p_dmr.add_argument("--treatment-group",
                       help="(tile only) Group label for treatment samples.")
    p_dmr.add_argument("--control-group",
                       help="(tile only) Group label for control samples.")
    p_dmr.add_argument("--tile-size-bp",       type=int,   default=1000,
                       help="(tile only) Tile width in bp. Default 1000 (methylKit default).")
    p_dmr.add_argument("--min-cpgs-per-tile",  type=int,   default=5,
                       help="(tile only) Minimum CpGs per tile per sample.")
    p_dmr.add_argument(
        "--test", choices=["lr", "score", "glm", "logit_t", "beta_binomial", "cmh", "fisher"],
        default="lr",
        help="(tile only) Statistical test applied to tile-level counts. "
             "Default 'lr' matches methylKit overdispersion='MN' test='Chisq'.",
    )
    p_dmr.add_argument(
        "--no-unite", action="store_false", dest="unite", default=True,
        help="(tile only) Test tiles covered in at least one sample (default: intersect).",
    )
    p_dmr.add_argument("--min-samples-case",    type=int, default=0,
                       help="(tile only) Per-tile minimum treatment samples.")
    p_dmr.add_argument("--min-samples-control", type=int, default=0,
                       help="(tile only) Per-tile minimum control samples.")
    p_dmr.add_argument(
        "--allow-n1", action="store_true", default=False,
        help=(
            "(tile only) Allow n<2 per group: fall back to Fisher exact on "
            "pooled reads. Default is to refuse — between-replicate variance "
            "is ignored and p-values are anti-conservative."
        ),
    )

    # Sliding-window-method options
    p_dmr.add_argument("--dmc-results",
                       help="(sliding_window only) Parquet file from 'epykit dmc'")
    p_dmr.add_argument("--window-bp",            type=int,   default=500)
    p_dmr.add_argument("--step-bp",              type=int,   default=250)
    p_dmr.add_argument("--min-cpgs",             type=int,   default=5)
    p_dmr.add_argument("--min-sites-significant",type=int,   default=3)

    # Shared filters
    p_dmr.add_argument("--alpha",                type=float, default=0.05)
    p_dmr.add_argument("--min-abs-meth-diff",    type=float, default=0.1)
    p_dmr.set_defaults(func=_cmd_dmr)

    # annotate
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

    # qc-report
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

    # smooth
    p_sm = sub.add_parser(
        "smooth",
        help="Gaussian-kernel methylation beta smoothing (approximates BSmooth)",
        description=(
            "Gaussian-kernel methylation beta smoothing. Approximates BSmooth "
            "via scipy.ndimage.gaussian_filter1d on a regular grid — not a "
            "true local LOESS. ~500x faster than statsmodels LOESS."
        ),
    )
    p_sm.add_argument("--methylstore", required=True)
    p_sm.add_argument(
        "--samples", required=True,
        help="Comma-separated list of sample IDs",
    )
    p_sm.add_argument("--output",    required=True,
                      help="Output directory for smoothed beta chunks")
    p_sm.add_argument("--bandwidth", type=int, default=1000,
                      help="Smoothing bandwidth in bp (default 1000)")
    p_sm.set_defaults(func=_cmd_smooth)

    args = ap.parse_args()
    _configure_logging(verbosity=args.verbose - args.quiet)
    args.func(args)


if __name__ == "__main__":
    main()