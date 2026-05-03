# epykit — scalable WGBS analysis in Python

`epykit` now supports an AnnData-inspired central object for WGBS workflows:
`MethylData`.

## New high-level API

```python
import epykit as ep

md = ep.read_bismark(
    "samplesheet.csv",
    treatment_group="cd55",
    control_group="control",
    assembly="hg38",
)

ep.pp.filter_coverage(md, lo_count=10, hi_perc=99.9)
ep.pp.unite(md, type="intersect")
ep.pp.smooth(md, bandwidth=1000)

ep.tl.qc(md)
ep.tl.dmc(md, test="auto")
ep.tl.dmr(md, window_bp=500, min_cpgs=5)
ep.tl.annotate(md, gtf="gencode.v49.gtf", cpg_islands="hg38_cpg_islands.bed")

ep.pl.volcano(md)
ep.pl.coverage_histogram(md)
ep.pl.methylation_heatmap(md, n_top=1000)

md.save("cd55_analysis")
md2 = ep.load("cd55_analysis")
```

## nf-core/methylseq direct entrypoint

```python
md = ep.read_nfcore_methylseq(
    "/path/to/nfcore_run",
    treatment_group="cd55",
    control_group="control",
    assembly="hg38",
)
```

This expects:

```text
run_dir/
  samplesheet.csv
  results/
    bismark/
      deduplicated/
        *.cov.gz
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For plotting support:

```bash
python -m pip install -e .[plotting]
```

### Samplesheet format

`ep.read_bismark(...)` expects CSV columns:

```csv
sample_id,group,path
S1,control,/path/to/S1.cov.gz
S2,cd55,/path/to/S2.cov.gz
```

Extra columns are preserved automatically in `md.obs` (e.g. `batch_id`,
`donor_age`, sequencing metadata) for downstream stratification.

## Legacy API

The original module-level functions remain exported for compatibility
(`convert_sample`, `filter_sites`, `process_chromosomes_dmc`, etc.).
