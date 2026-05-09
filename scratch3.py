"""Scratch3: Full sample analysis with all groups from full_samplesheet"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import epykit as ep
import polars as pl

print("=" * 80)
print("ANALYZING FULL SAMPLESHEET WITH ALL GROUPS")
print("=" * 80)

# Load all samples from full_samplesheet
md = ep.read_bismark(
    "full_samplesheet.csv",
    treatment_group="cd55",
    control_group="control",
    assembly="hg38",
    store_dir="methyl_store_full",
)

print("\nLoaded methylation data:")
print(md)
print(f"\nGroups present: {md.obs['group'].unique()}")
print(f"Sample counts: \n{md.obs.group_by('group').agg(pl.col('sample_id').count())}")

# Quality control and preprocessing
print("\n" + "-" * 80)
print("PREPROCESSING")
print("-" * 80)

ep.pp.filter_coverage(md, lo_count=10, hi_perc=99.9)
print("✓ Coverage filtering complete")

ep.pp.unite(md, type="union")
print("✓ Unite complete")

# QC analysis
print("\n" + "-" * 80)
print("QC ANALYSIS")
print("-" * 80)
ep.tl.qc(md)
print("✓ QC analysis complete")

# Differential methylation analysis: cd55 vs control
print("\n" + "-" * 80)
print("DIFFERENTIAL METHYLATION ANALYSIS: CD55 vs CONTROL")
print("-" * 80)

# Set treatment column for cd55 vs control
md.obs = md.obs.with_columns(
    treatment=pl.when(pl.col('group') == 'cd55').then(1).when(pl.col('group') == 'control').then(0).otherwise(None)
)
md.obs = md.obs.filter(pl.col('treatment').is_not_null())

ep.tl.dmc(md, test="auto")
ep.tl.dmr(md, window_bp=500, min_cpgs=5)

print(f"CD55 vs Control - Significant DMCs (qvalue < 0.05): {len(md.dmc.filter(pl.col('qvalue') < 0.05))}")
print(f"CD55 vs Control - Significant DMRs (combined_pvalue < 0.05): {len(md.uns['dmr'].filter(pl.col('combined_pvalue') < 0.05))}")

if len(md.dmc) > 0:
    print("\nTop 5 DMCs (CD55 vs Control):")
    print(md.dmc.sort('qvalue').head(5).select(['pos', 'meth_diff', 'pvalue', 'qvalue']))

# Differential methylation analysis: cd81 vs control
print("\n" + "-" * 80)
print("DIFFERENTIAL METHYLATION ANALYSIS: CD81 vs CONTROL")
print("-" * 80)

# Reload full samplesheet
md.obs = pl.read_csv('full_samplesheet.csv')
md.obs = md.obs.with_columns(treatment=pl.lit(0))  # Reset treatment

# Set treatment column for cd81 vs control
md.obs = md.obs.with_columns(
    treatment=pl.when(pl.col('group') == 'cd81').then(1).when(pl.col('group') == 'control').then(0).otherwise(None)
)
md.obs = md.obs.filter(pl.col('treatment').is_not_null())

ep.tl.dmc(md, test="auto")
ep.tl.dmr(md, window_bp=500, min_cpgs=5)

print(f"CD81 vs Control - Significant DMCs (qvalue < 0.05): {len(md.dmc.filter(pl.col('qvalue') < 0.05))}")
print(f"CD81 vs Control - Significant DMRs (combined_pvalue < 0.05): {len(md.uns['dmr'].filter(pl.col('combined_pvalue') < 0.05))}")

if len(md.dmc) > 0:
    print("\nTop 5 DMCs (CD81 vs Control):")
    print(md.dmc.sort('qvalue').head(5).select(['pos', 'meth_diff', 'pvalue', 'qvalue']))

# Differential methylation analysis: empty vs control
print("\n" + "-" * 80)
print("DIFFERENTIAL METHYLATION ANALYSIS: EMPTY vs CONTROL")
print("-" * 80)

# Reload full samplesheet
md.obs = pl.read_csv('full_samplesheet.csv')
md.obs = md.obs.with_columns(treatment=pl.lit(0))  # Reset treatment

# Set treatment column for empty vs control
md.obs = md.obs.with_columns(
    treatment=pl.when(pl.col('group') == 'empty').then(1).when(pl.col('group') == 'control').then(0).otherwise(None)
)
md.obs = md.obs.filter(pl.col('treatment').is_not_null())

ep.tl.dmc(md, test="auto")
ep.tl.dmr(md, window_bp=500, min_cpgs=5)

print(f"Empty vs Control - Significant DMCs (qvalue < 0.05): {len(md.dmc.filter(pl.col('qvalue') < 0.05))}")
print(f"Empty vs Control - Significant DMRs (combined_pvalue < 0.05): {len(md.uns['dmr'].filter(pl.col('combined_pvalue') < 0.05))}")

if len(md.dmc) > 0:
    print("\nTop 5 DMCs (Empty vs Control):")
    print(md.dmc.sort('qvalue').head(5).select(['pos', 'meth_diff', 'pvalue', 'qvalue']))

# Differential methylation analysis: cd55 vs cd81
print("\n" + "-" * 80)
print("DIFFERENTIAL METHYLATION ANALYSIS: CD55 vs CD81")
print("-" * 80)

# Reload full samplesheet
md.obs = pl.read_csv('full_samplesheet.csv')
md.obs = md.obs.with_columns(treatment=pl.lit(0))  # Reset treatment

# Set treatment column for cd55 vs cd81
md.obs = md.obs.with_columns(
    treatment=pl.when(pl.col('group') == 'cd55').then(1).when(pl.col('group') == 'cd81').then(0).otherwise(None)
)
md.obs = md.obs.filter(pl.col('treatment').is_not_null())

ep.tl.dmc(md, test="auto")
ep.tl.dmr(md, window_bp=500, min_cpgs=5)

print(f"CD55 vs CD81 - Significant DMCs (qvalue < 0.05): {len(md.dmc.filter(pl.col('qvalue') < 0.05))}")
print(f"CD55 vs CD81 - Significant DMRs (combined_pvalue < 0.05): {len(md.uns['dmr'].filter(pl.col('combined_pvalue') < 0.05))}")

if len(md.dmc) > 0:
    print("\nTop 5 DMCs (CD55 vs CD81):")
    print(md.dmc.sort('qvalue').head(5).select(['pos', 'meth_diff', 'pvalue', 'qvalue']))

# Annotation on full dataset
print("\n" + "-" * 80)
print("ANNOTATION")
print("-" * 80)

ep.tl.annotate(
    md,
    gtf="raw_data/gencode.v49.chr_patch_hapl_scaff.annotation.gtf",
    cpg_islands="raw_data/hg38_cpg_islands.bed"
)
print("✓ Annotation complete")

# Save full analysis
md.save("full_analysis")
print("\n✓ Full analysis saved to 'full_analysis'")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

n_cpgs = md.uns.get("n_sites_filtered") or md.uns.get("n_sites_raw", "?")
print(f"\nTotal CpGs analyzed: {n_cpgs}")
print(f"Total samples: {md.n_samples}")
print(f"Groups: {md.obs['group'].unique().to_list()}")
print(f"\nSample composition:")
print(md.obs.group_by('group').agg(
    pl.col('sample_id').count().alias('count'),
    pl.col('sample_id').n_unique().alias('unique_samples')
))

# Clear memory before loading for plotting
import gc
del md
gc.collect()
print("\n✓ Memory cleared")

# Reload for plotting
md = ep.load("methyl_store_full/results/full_analysis")
print("✓ Reloaded analysis for visualization")

# Generate plots
print("\n" + "=" * 80)
print("GENERATING PLOTS")
print("=" * 80)

ep.pl.coverage_histogram(md, save="coverage_histogram_full")
print("✓ Coverage histogram saved")

ep.pl.genomic_context_bar(md, save="genomic_context_bar_full")
print("✓ Genomic context bar plot saved")

ep.pl.cpg_island_pie(md, save="cpg_island_pie_full")
print("✓ CpG island pie chart saved")

ep.pl.pca(md, save="pca_plot_full")
print("✓ PCA plot saved")

ep.pl.methylation_heatmap(md, n_top=500, save="methylation_heatmap_full")
print("✓ Methylation heatmap saved")

print("\n" + "=" * 80)
print("ANALYSIS AND PLOTTING COMPLETE")
print("=" * 80)
