from __future__ import annotations

import polars as pl

from .annotate import annotate_cpg_islands, annotate_features
from .dmc import apply_multiple_testing_correction, process_chromosomes_dmc
from .dmr import call_dmr_sliding_window
from .methyldata import MethylData
from .qc import bisulfite_conversion_rate, coverage_uniformity, global_methylation_report


def _auto_test(md: MethylData) -> str:
    min_group = min(len(md.treatment_ids), len(md.control_ids))
    return "beta_binomial" if min_group >= 6 else "fisher"


def qc(md: MethylData, chh_context_store: str | None = None) -> None:
    """Populate md.obs with per-sample QC metrics and cache QC tables in md.uns."""
    samples = md.obs.get_column("sample_id").to_list()

    global_report = global_methylation_report(md.store, samples)
    cov_reports = [coverage_uniformity(md.store, sample) for sample in samples]
    cov_report = pl.concat(cov_reports) if cov_reports else pl.DataFrame()

    obs = md.obs
    cpG_report = global_report.filter(pl.col("context") == "CpG")
    if len(cpG_report) > 0:
        obs = obs.join(
            cpG_report.select([pl.col("sample").alias("sample_id"), "global_methylation"]),
            on="sample_id",
            how="left",
        )

    if len(cov_report) > 0:
        cov_genome = cov_report.filter(pl.col("chrom") == "genome")
        obs = obs.join(
            cov_genome.select([
                "sample",
                "mean_coverage",
                "frac_ge_1x",
                "frac_ge_5x",
                "frac_ge_10x",
                "low_coverage_flag",
            ]).rename({"sample": "sample_id"}),
            on="sample_id",
            how="left",
        )

    if chh_context_store:
        conv = []
        for sample in samples:
            try:
                rate = bisulfite_conversion_rate(md.store, sample, chh_context_store)
            except Exception:
                rate = None
            conv.append({"sample_id": sample, "bisulfite_conversion_rate": rate})
        obs = obs.join(pl.DataFrame(conv), on="sample_id", how="left")

    md.obs = obs
    md.uns["qc_global_methylation"] = global_report
    md.uns["qc_coverage_uniformity"] = cov_report


def dmc(
    md: MethylData,
    test: str = "auto",
    chromosomes: list[str] | None = None,
) -> None:
    """Run DMC calling and store result in md.varm['dmc_<test>']."""
    selected_test = _auto_test(md) if test == "auto" else test
    unite = md.uns.get("unite", {}).get("type", "intersect") == "intersect"

    result = process_chromosomes_dmc(
        methylstore_path=md.store,
        samples_case=md.treatment_ids,
        samples_control=md.control_ids,
        test=selected_test,
        chromosomes=chromosomes,
        unite=unite,
    )
    result = apply_multiple_testing_correction(result, method="fdr_bh")

    key = f"dmc_{selected_test}"
    md.varm[key] = result
    md.uns["dmc"] = {
        "test_requested": test,
        "test_used": selected_test,
        "n_sites": len(result),
        "unite": unite,
    }


def dmr(
    md: MethylData,
    window_bp: int = 500,
    step_bp: int = 250,
    min_cpgs: int = 5,
    min_sites_significant: int = 3,
    alpha: float = 0.05,
    min_abs_meth_diff: float = 0.1,
) -> None:
    """Run DMR calling using the current DMC result and store in md.uns['dmr']."""
    dmc_df = md.dmc
    if dmc_df is None:
        raise ValueError("No DMC results available. Run ep.tl.dmc(md) first.")

    dmr_df = call_dmr_sliding_window(
        dmc_results=dmc_df,
        window_bp=window_bp,
        step_bp=step_bp,
        min_cpgs=min_cpgs,
        min_sites_significant=min_sites_significant,
        alpha=alpha,
        min_abs_meth_diff=min_abs_meth_diff,
    )
    md.uns["dmr"] = dmr_df
    md.uns["dmr_params"] = {
        "window_bp": window_bp,
        "step_bp": step_bp,
        "min_cpgs": min_cpgs,
        "min_sites_significant": min_sites_significant,
        "alpha": alpha,
        "min_abs_meth_diff": min_abs_meth_diff,
    }


def annotate(
    md: MethylData,
    gtf: str | None = None,
    cpg_islands: str | None = None,
    promoter_upstream_bp: int = 2000,
    promoter_downstream_bp: int = 200,
) -> None:
    """Annotate DMC/DMR outputs in place."""
    if not gtf and not cpg_islands:
        raise ValueError("Provide at least one of gtf or cpg_islands")

    for key, df in list(md.varm.items()):
        if not key.startswith("dmc"):
            continue
        ann = df
        if gtf:
            ann = annotate_features(
                ann,
                annotation_gtf=gtf,
                promoter_upstream_bp=promoter_upstream_bp,
                promoter_downstream_bp=promoter_downstream_bp,
            )
        if cpg_islands:
            ann = annotate_cpg_islands(ann, cpg_island_bed=cpg_islands)
        md.varm[key] = ann

    if "dmr" in md.uns and isinstance(md.uns["dmr"], pl.DataFrame) and gtf:
        md.uns["dmr"] = annotate_features(
            md.uns["dmr"],
            annotation_gtf=gtf,
            promoter_upstream_bp=promoter_upstream_bp,
            promoter_downstream_bp=promoter_downstream_bp,
        )

    md.uns["annotation"] = {
        "gtf": gtf,
        "cpg_islands": cpg_islands,
        "promoter_upstream_bp": promoter_upstream_bp,
        "promoter_downstream_bp": promoter_downstream_bp,
    }
