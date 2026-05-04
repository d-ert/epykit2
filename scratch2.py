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
ep.pp.unite(md, type="union")
ep.pp.smooth(md, bandwidth=1000)

ep.tl.qc(md)
ep.tl.dmc(md, test="auto")
ep.tl.dmr(md, window_bp=500, min_cpgs=5)

# Check DMC results
print(md.dmc.filter(pl.col("qvalue") < 0.05))

# Check DMR results  
print(md.uns["dmr"].filter(pl.col("mean_pvalue") < 0.05))



ep.tl.annotate(md, gtf="raw_data/gencode.v49.chr_patch_hapl_scaff.annotation.gtf", cpg_islands="raw_data/hg38_cpg_islands.bed")

md.save("cd55_analysis")


# plot volcano plot
ep.pl.volcano(md, dmc_key="dmc", title="Volcano Plot of DMCs", save_path="volcano_plot.png")
print("Volcano plot saved as volcano_plot.png")

# plot MA plot
ep.pl.ma_plot(md, dmc_key="dmc", title="MA Plot of DMCs", save_path="ma_plot.png")
print("MA plot saved as ma_plot.png")

# plot Manhattan plot
ep.pl.manhattan(md, dmc_key="dmc", title="Manhattan Plot of DMCs", save_path="manhattan_plot.png")
print("Manhattan plot saved as manhattan_plot.png")

# plot PCA
ep.pl.pca(md, title="PCA of Methylation Data", save_path="pca_plot.png")
print("PCA plot saved as pca_plot.png")

# plot coverage histogram
ep.pl.coverage_histogram(md, title="Coverage Histogram", save_path="coverage_histogram.png")
print("Coverage histogram saved as coverage_histogram.png")

# plot methylation heatmap
ep.pl.methylation_heatmap(md, title="Methylation Heatmap", save_path="methylation_heatmap.png")
print("Methylation heatmap saved as methylation_heatmap.png")

# plot genomic context bar plot
ep.pl.genomic_context_bar(md, title="Genomic Context Distribution", save_path="genomic_context_bar.png")
print("Genomic context bar plot saved as genomic_context_bar.png")

# plot CpG island pie chart
ep.pl.cpg_island_pie(md, title="CpG Island Distribution", save_path="cpg_island_pie.png")
print("CpG island pie chart saved as cpg_island_pie.png")

print("All plots generated and saved successfully.")
print("Analysis complete.")