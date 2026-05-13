"""Multi-contrast epykit analysis: cd55, cd81, empty vs control + cd55 vs cd81.

The samplesheet has 16 samples across 4 groups (cd55, cd81, control, empty),
each with 2 donors (d3, d4) and 2 technical replicates per donor — a clean
2 x 2 x 4 balanced design. That makes ``donor`` a well-conditioned covariate
for paired analysis: 4 samples per group, 4 residual df after `~ treatment +
donor`.

For each contrast we run:
    1. read_bismark            (Parquet conversion + MethylData)
    2. filter / normalize      (coverage QC, median-scale)
    3. unite (intersect)       (CpGs covered in every sample)
    4. qc                      (bisulfite conv, global methylation, coverage)
    5. dmc (auto -> lr)        (per-CpG quasi-binomial LR with MN dispersion)
    6. unpaired DMR (tile)     (kept as md.uns["dmr_unpaired"])
    7. paired DMR              (treatment + donor; canonical md.uns["dmr"])
    8. annotate                (gene features + CpG islands)
    9. save                    (results/{contrast})
   10. plots                   (volcano, MA, manhattan, coverage hist,
                                genomic-context bar, CpG-island pie, PCA,
                                methylation heatmap)

The Parquet store under ``methyl_store_test/`` is shared across contrasts:
each .cov is converted once at first sight and re-used. The .cache/filtered
and .cache/normalized subdirectories are rewritten per contrast because the
sample sets differ, which is expected (and cheap given the converted store).
"""

from pathlib import Path
import gc
import logging

import matplotlib
matplotlib.use("Agg", force=True)


# ---------------------------------------------------------------------------
# Logging
#   - INFO from epykit.* everywhere
#   - DEBUG for the GTF parse cache + filter progress (the things we usually
#     want to see when something's slow)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename="log_output_multigroup.txt",
)
logging.getLogger("epykit.annotate").setLevel(logging.DEBUG)
logging.getLogger("epykit.filter").setLevel(logging.DEBUG)


import epykit as ep
import polars as pl


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAMPLESHEET = "full_samplesheet.csv"
STORE_DIR   = "epykit_test_full"
ASSEMBLY    = "hg38"

GTF         = "raw_data/gencode.v49.chr_patch_hapl_scaff.annotation.gtf"
CPG_ISLANDS = "raw_data/hg38_cpg_islands.bed"

# Pairwise contrasts. epykit's pipeline is 2-group, so a 4-group design like
# {cd55, cd81, control, empty} gets decomposed into the contrasts of interest:
#   - each construct vs the control (the standard "is this CD-target gene
#     driving methylation changes?" question)
#   - empty-vector vs control (negative control for the construct backbone)
#   - cd55 vs cd81 (direct construct-vs-construct effect)
CONTRASTS = [
    ("cd55",  "control"),
    ("cd81",  "control"),
    ("empty", "control"),
    ("cd55",  "cd81"),
]

# Filter / normalization parameters (same as scratch2 so results are
# directly comparable across the two pipelines)
LO_COUNT       = 10
HI_PERC        = 99.9
NORM_METHOD    = "median"

# DMR parameters
TILE_BP        = 500
MIN_CPGS       = 5

# Plot list. Each entry: (suffix, plot_fn, needs_extra_deps).
PLOTS = [
    ("volcano",             ep.pl.volcano,             False),
    ("ma_plot",              ep.pl.ma_plot,              False),
    ("manhattan",           ep.pl.manhattan,           False),
    ("coverage_histogram",  ep.pl.coverage_histogram,  False),
    ("genomic_context_bar", ep.pl.genomic_context_bar, False),
    ("cpg_island_pie",      ep.pl.cpg_island_pie,      False),
    ("pca",                 ep.pl.pca,                 True),   # scikit-learn
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _hline(label: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n  {label}\n{bar}")


def _try_plot(fn, md, save_name: str, label: str, needs_extra: bool) -> None:
    """Run one plot and report status without aborting the contrast."""
    try:
        fn(md, save=save_name)
        print(f"  [ok]   {label}")
    except ImportError:
        if needs_extra:
            print(f"  [skip] {label} (missing optional dep)")
        else:
            raise
    except Exception as exc:
        print(f"  [FAIL] {label}: {exc}")


# ---------------------------------------------------------------------------
# Per-contrast pipeline
# ---------------------------------------------------------------------------
def run_contrast(treatment: str, control: str) -> None:
    name = f"{treatment}_vs_{control}"
    _hline(f"CONTRAST: {name}")

    # --- Ingest ------------------------------------------------------------
    md = ep.read_bismark(
        SAMPLESHEET,
        treatment_group=treatment,
        control_group=control,
        assembly=ASSEMBLY,
        store_dir=STORE_DIR,
    )
    print(md)

    # --- Preprocess --------------------------------------------------------
    ep.pp.filter_coverage(md, lo_count=LO_COUNT, hi_perc=HI_PERC)
    ep.pp.normalize_coverage(md, method=NORM_METHOD)
    ep.pp.unite(md, type="intersect")

    # --- QC ---------------------------------------------------------------
    ep.tl.qc(md)

    # --- DMC --------------------------------------------------------------
    ep.tl.dmc(md, test="auto")
    total = len(md.dmc)
    sig_df = md.dmc.filter(pl.col("qvalue") < 0.05)
    sig = sig_df.height
    print(f"  DMC: {sig:,} significant of {total:,} ({100*sig/total:.2f}%)")
    if sig > 0:
        print(sig_df.select("meth_diff").describe())

    # --- Donor covariate (regex extracts 'd3' / 'd4' from sample_id) ------
    md.obs = md.obs.with_columns(
        pl.col("sample_id").str.extract(r"_(d\d+)_", 1).alias("donor")
    )
    print("\n  Sample -> donor map:")
    print(md.obs.select(["sample_id", "group", "treatment", "donor"]))

    # --- Unpaired tile DMR ------------------------------------------------
    ep.tl.dmr(md, tile_size_bp=TILE_BP, min_cpgs_per_tile=MIN_CPGS)
    dmr_unpaired = md.uns["dmr"].clone()
    print(f"  Unpaired DMR: {len(dmr_unpaired):,} tiles, "
          f"{dmr_unpaired.filter(pl.col('qvalue') < 0.05).height:,} significant")

    # --- Paired tile DMR (donor as covariate) -----------------------------
    # With 4 samples per group split evenly across two donors, the model
    # `~ treatment + donor` has 8 - 3 = 5 residual df per tile -- safely
    # well-conditioned. Covariate-adjustment should add power if donor
    # variance is non-trivial.
    ep.tl.dmr(
        md,
        tile_size_bp=TILE_BP,
        min_cpgs_per_tile=MIN_CPGS,
        covariates=["donor"],
    )
    dmr_paired = md.uns["dmr"].clone()
    print(f"  Paired DMR (donor): {len(dmr_paired):,} tiles, "
          f"{dmr_paired.filter(pl.col('qvalue') < 0.05).height:,} significant")
    print(f"    formula      : {md.uns['dmr_params']['formula_used']}")
    print(f"    design terms : {md.uns['dmr_params']['design_terms']}")
    print(f"    test used    : {md.uns['dmr_params']['test']}")

    # Stash both DMR tables so the saved object preserves them. The "active"
    # md.uns["dmr"] remains the paired analysis (more interesting biologically).
    md.uns["dmr_unpaired"] = dmr_unpaired
    md.uns["dmr_paired"]   = dmr_paired

    # Top paired DMRs by qvalue (GLM path adds coef_treatment / coef_se)
    paired_sig = dmr_paired.filter(pl.col("qvalue") < 0.05).sort("qvalue")
    if len(paired_sig) > 0:
        cols = ["chrom", "start", "end", "meth_diff",
                "coef_treatment", "coef_se", "pvalue", "qvalue"]
        keep = [c for c in cols if c in paired_sig.columns]
        print("\n  Top 5 paired DMRs by qvalue:")
        print(paired_sig.select(keep).head(5))

    # --- Annotate ---------------------------------------------------------
    ep.tl.annotate(md, gtf=GTF, cpg_islands=CPG_ISLANDS)

    # --- Save -------------------------------------------------------------
    md.save(name)
    save_path = f"{STORE_DIR}/results/{name}"
    print(f"  saved to {save_path}")

    # Free the in-memory object before reloading the smaller annotated one
    # for plotting (avoids holding two MethylDatas at once).
    del md, dmr_unpaired, dmr_paired
    gc.collect()

    # --- Plots ------------------------------------------------------------
    md = ep.load(save_path)
    print(f"\n  Plotting ({name}):")

    for suffix, fn, needs_extra in PLOTS:
        _try_plot(fn, md, save_name=f"{name}_{suffix}",
                  label=suffix, needs_extra=needs_extra)

    # heatmap is its own call because it takes an extra arg
    try:
        ep.pl.methylation_heatmap(md, n_top=500, save=f"{name}_heatmap")
        print(f"  [ok]   methylation_heatmap")
    except Exception as exc:
        print(f"  [FAIL] methylation_heatmap: {exc}")

    del md
    gc.collect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("epykit version :", ep.__version__)
    print("samplesheet    :", Path(SAMPLESHEET).resolve())
    print("store          :", Path(STORE_DIR).resolve())
    print("contrasts      :", CONTRASTS)

    for treatment, control in CONTRASTS:
        try:
            run_contrast(treatment, control)
        except Exception as exc:
            # Don't let one contrast take down the whole pipeline; log and
            # continue with the next one.
            logging.getLogger(__name__).exception(
                "Contrast %s vs %s failed", treatment, control,
            )
            print(f"\n[FAIL] {treatment} vs {control}: {exc}\n")

    print("\nAll contrasts complete.")


if __name__ == "__main__":
    main()
