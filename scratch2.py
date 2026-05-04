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

plot_names = [
    ("volcano", ep.pl.volcano),
    ("ma_plot", ep.pl.ma_plot),
    ("manhattan", ep.pl.manhattan),
    ("pca", ep.pl.pca),
    ("coverage_histogram", ep.pl.coverage_histogram),
    ("methylation_heatmap", ep.pl.methylation_heatmap),
    ("genomic_context_bar", ep.pl.genomic_context_bar),
    ("cpg_island_pie", ep.pl.cpg_island_pie),
]

for plot_name, plot_fn in plot_names:
    plot_fn(md, save=f"scratch2_{plot_name}")
    plot_path = Path("figures") / f"scratch2_{plot_name}.png"
    assert plot_path.exists(), f"Missing plot output: {plot_path}"
    print(f"{plot_name}: {plot_path}")