"Scratch2"
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import epykit as ep
import polars as pl

md = ep.read_bismark(
    "samplesheet.csv",
    treatment_group="cd55",
    control_group="control",
    assembly="hg38",
    store_dir="methyl_store_test",
)

print(md)

ep.pp.filter_coverage(md, lo_count=10, hi_perc=99.9)
ep.pp.normalize_coverage(md, method="median")        # ← new
ep.pp.unite(md, type="intersect")
#ep.pp.smooth(md, bandwidth=1000)

ep.tl.qc(md)
ep.tl.dmc(md, test="auto")

# After ep.tl.dmc(md, test="auto")
total = len(md.dmc)
sig = md.dmc.filter(pl.col('qvalue') < 0.05).height
pct = 100 * sig / total

print(f"Total sites tested: {total:,}")
print(f"Significant DMCs: {sig:,}")  
print(f"Percentage: {pct:.2f}%")
print(f"\nEffect size summary:")
print(md.dmc.filter(pl.col('qvalue') < 0.05).select('meth_diff').describe())

ep.tl.dmr(md, tile_size_bp=500, min_cpgs_per_tile=5)

# Check DMC results
print(md.dmc.filter(pl.col("qvalue") < 0.05))

# Check DMR results  
print(md.uns["dmr"].filter(pl.col("qvalue") < 0.05))



ep.tl.annotate(md, gtf="raw_data/gencode.v49.chr_patch_hapl_scaff.annotation.gtf", cpg_islands="raw_data/hg38_cpg_islands.bed")

md.save("cd55_analysis")

# Clear memory before plotting to avoid OOM issues
import gc
del md
gc.collect()

# Reload the preprocessed/annotated data for plotting (much faster than full pipeline)
md = ep.load("methyl_store_test/results/cd55_analysis")
print("\nLoaded preprocessed data for plotting")

# Plot differential methylation plots
try:
    ep.pl.volcano(md, save="volcano_plot")
    print("✓ Volcano plot saved")
except Exception as e:
    print(f"✗ Volcano plot failed: {e}")

try:
    ep.pl.ma_plot(md, save="ma_plot")
    print("✓ MA plot saved")
except Exception as e:
    print(f"✗ MA plot failed: {e}")

try:
    ep.pl.manhattan(md, save="manhattan_plot")
    print("✓ Manhattan plot saved")
except Exception as e:
    print(f"✗ Manhattan plot failed: {e}")

# Plot coverage histogram
try:
    ep.pl.coverage_histogram(md, save="coverage_histogram")
    print("✓ Coverage histogram saved")
except Exception as e:
    print(f"✗ Coverage histogram failed: {e}")

# Plot genomic context bar plot
try:
    ep.pl.genomic_context_bar(md, save="genomic_context_bar")
    print("✓ Genomic context bar plot saved")
except Exception as e:
    print(f"✗ Genomic context bar plot failed: {e}")

# Plot CpG island pie chart
try:
    ep.pl.cpg_island_pie(md, save="cpg_island_pie")
    print("✓ CpG island pie chart saved")
except Exception as e:
    print(f"✗ CpG island pie chart failed: {e}")

# Optional: PCA (requires scikit-learn)
try:
    ep.pl.pca(md, save="pca_plot")
    print("✓ PCA plot saved")
except ImportError:
    print("⊘ PCA skipped (requires: pip install scikit-learn)")
except Exception as e:
    print(f"✗ PCA failed: {e}")

# Optional: Methylation heatmap (now optimized)
try:
    ep.pl.methylation_heatmap(md, n_top=500, save="methylation_heatmap")
    print("✓ Methylation heatmap saved")
except Exception as e:
    print(f"✗ Methylation heatmap failed: {e}")

print("\nAnalysis complete.")