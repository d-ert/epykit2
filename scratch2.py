"Scratch2"
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