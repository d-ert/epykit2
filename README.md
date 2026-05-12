# epykit

Parquet-backed WGBS methylation analysis pipeline — from Bismark `.cov` files to differentially methylated cytosines (DMCs), differentially methylated regions (DMRs), and gene-feature annotation.

epykit ingests Bismark coverage output into a partitioned Parquet **methylstore** and runs the whole downstream analysis (QC → filtering → DMC → DMR → annotation → plotting) over that store with [polars](https://pola.rs) and lazy I/O. The Python API is organised in a scanpy-style `pp` / `tl` / `pl` namespace; a CLI mirrors the same operations for scripting.

> **Status:** version 0.1.0, pre-1.0. API may change. MIT licensed.

---

## Highlights

- **Partitioned Parquet methylstore.** Per-chromosome, per-sample columnar storage — never load a whole genome into RAM.
- **methylKit-parity statistics.** DMC backends include the quasi-binomial likelihood-ratio test (`lr`, default at n ≥ 2, matches `calculateDiffMeth(overdispersion="MN", test="Chisq")`), score test, GLM with covariates, Welch t on logit(β), Cochran–Mantel–Haenszel, beta-binomial, and Fisher exact for legacy comparisons.
- **Two DMR engines.** Tile-based (read-pooled, default) and per-CpG sliding-window with signed Stouffer's combining.
- **Replicate-aware throughout.** Per-site `min_samples_case` / `min_samples_control` guards, per-site or chromosome-level McCullagh–Nelder dispersion, optional covariate design matrices via a Wilkinson formula.
- **Annotation.** GTF gene features (promoter / 5'UTR / exon / intron / 3'UTR) and UCSC CpG-island context.
- **QC.** Bisulfite conversion rate from a CHH-context store, global methylation report, per-sample coverage uniformity.
- **Plotting.** matplotlib volcano, MA, Manhattan, coverage histogram, methylation heatmap, PCA, genomic-context bar, CpG-island pie.
- **CLI.** `epykit convert | filter | dmc | dmr | annotate | qc-report | smooth` — every stage scriptable from the shell.

---

## Installation

Requires Python ≥ 3.9.

```bash
# from the repo checkout
pip install -e .

# or with uv
uv pip install -e .

# dev install (adds pytest, pytest-cov)
pip install -e ".[dev]"
```

Core dependencies: `polars`, `pyarrow`, `numpy`, `scipy`, `numba`, `pyranges`, `pyfaidx`, `statsmodels`, `psutil`, `scikit-learn`, `matplotlib`, `seaborn`. The CLI is installed as the `epykit` console script.

---

## Quickstart

### 1. Samplesheet

epykit reads a CSV with three required columns. Any extra columns are kept on `md.obs` and are available as GLM covariates.

```csv
sample_id,group,path
ctrl_1,control,raw_data/bismark/ctrl_1.bismark.cov.gz
ctrl_2,control,raw_data/bismark/ctrl_2.bismark.cov.gz
cd55_1,cd55,raw_data/bismark/cd55_1.bismark.cov.gz
cd55_2,cd55,raw_data/bismark/cd55_2.bismark.cov.gz
```

### 2. End-to-end analysis (Python API)

```python
import epykit as ep
import polars as pl

# Ingest: converts each .cov to per-chromosome Parquet under
# methyl_store_test/.cache/raw/ and returns a MethylData object.
md = ep.read_bismark(
    "samplesheet.csv",
    treatment_group="cd55",
    control_group="control",
    assembly="hg38",
    store_dir="methyl_store_test",
)
print(md)

# Preprocessing (pp.*) — each step repoints md.store at a cached store.
ep.pp.filter_coverage(md, lo_count=10, hi_perc=99.9)
ep.pp.normalize_coverage(md, method="median")
ep.pp.unite(md, type="intersect")        # or "union" + min_samples_* guards

# Tools (tl.*) — populate md.obs / md.varm / md.uns.
ep.tl.qc(md)                              # populates md.obs with QC metrics
ep.tl.dmc(md, test="auto")                # md.varm["dmc_lr"] (n≥2) or "dmc_fisher" (n=1)
ep.tl.dmr(md, tile_size_bp=500, min_cpgs_per_tile=5)   # tile-based, default

# Inspect.
total = len(md.dmc)
sig   = md.dmc.filter(pl.col("qvalue") < 0.05).height
print(f"DMCs: {sig:,} / {total:,} ({100 * sig / total:.2f}%)")
print(md.uns["dmr"].filter(pl.col("qvalue") < 0.05))

# Annotation.
ep.tl.annotate(
    md,
    gtf="raw_data/gencode.v49.annotation.gtf",
    cpg_islands="raw_data/hg38_cpg_islands.bed",
)

# Persist the analysis (obs / varm / uns + a manifest pointing at the
# methylstore cache layers) so plotting can be re-run without redoing the
# full pipeline.
md.save("cd55_analysis")

# Plotting (pl.*) — works on a freshly loaded MethylData.
md = ep.load("methyl_store_test/results/cd55_analysis")
ep.pl.volcano(md,            save="volcano_plot")
ep.pl.ma_plot(md,            save="ma_plot")
ep.pl.manhattan(md,          save="manhattan_plot")
ep.pl.coverage_histogram(md, save="coverage_histogram")
ep.pl.genomic_context_bar(md, save="genomic_context_bar")
ep.pl.cpg_island_pie(md,     save="cpg_island_pie")
ep.pl.methylation_heatmap(md, n_top=500, save="methylation_heatmap")
ep.pl.pca(md,                save="pca_plot")
```

`scratch2.py` at the repo root is a copy of this flow used during development — feel free to run it as a smoke test against `raw_data/`.

### 3. Covariate-adjusted DMR (optional)

When `md.obs` has additional columns (sex, batch, age, …), pass them through to a binomial GLM tile model:

```python
ep.tl.dmr(
    md,
    method="tile",
    design="~ treatment + sex + batch",   # Wilkinson formula
    treatment_col="treatment",            # column tested for non-zero coefficient
)
```

If you pass `design=` or `covariates=`, the engine forces the `glm` backend regardless of `test="auto"`.

---

## CLI

The `epykit` script mirrors the Python pipeline. Every subcommand takes `--methylstore` (the partitioned Parquet directory) and writes Parquet output.

| Subcommand   | Purpose |
|--------------|---------|
| `convert`    | Bismark `.cov[.gz]` → partitioned Parquet (`--input`, `--sample-id`, `--output-dir`, `--context {CpG,CHG,CHH}`) |
| `filter`     | Coverage / blacklist filtering (`--min-coverage`, `--max-coverage-quantile`, `--blacklist-bed`) |
| `summary`    | Per-sample summary statistics |
| `dmc`        | Per-CpG differential methylation (`--test {score,logit_t,beta_binomial,cmh,fisher}`, `--samplesheet`, `--treatment-group`, `--control-group`) |
| `dmr`        | DMR calling — `--method tile` (default, methylstore-driven) or `--method sliding_window` (DMC-driven) |
| `annotate`   | Add gene-feature (`--gtf`) and CpG-island (`--cpg-islands`) annotation to a DMC/DMR Parquet |
| `qc-report`  | Global methylation + coverage uniformity report |
| `smooth`     | BSmooth-style LOESS β smoothing |

Run `epykit <subcommand> --help` for the full flag list. The `dmc` and `dmr --method tile` subcommands share the `--min-samples-case` / `--min-samples-control` / `--no-unite` semantics with the Python API.

---

## Input formats

- **Bismark `.cov` / `.cov.gz`** — 6-column 0-based BED-like:
  `chrom`, `start`, `end`, `methylation_percent`, `count_methylated`, `count_unmethylated`.
- **Samplesheet** (CSV) — required columns `sample_id`, `group`, `path`. Any extra column is preserved on `md.obs` and can be referenced as a GLM covariate.
- **GTF** — Ensembl/GENCODE/UCSC; gene features are extracted via [pyranges].
- **CpG-island BED** — UCSC `cpgIslandExt` 4-column BED.

---

## Output layout

`read_bismark(..., store_dir="methyl_store_test")` produces:

```
methyl_store_test/
├── .cache/
│   ├── raw/                      # converted .cov → Parquet
│   │   └── chromosome=chr1/
│   │       ├── sample=ctrl_1/part-0.parquet
│   │       └── sample=cd55_1/part-0.parquet
│   ├── filtered/                 # after pp.filter_coverage
│   └── normalized/               # after pp.normalize_coverage
└── results/
    └── cd55_analysis/            # md.save() target
        ├── obs.parquet
        ├── varm/dmc_lr_annotated.parquet
        ├── uns/dmr.parquet
        └── manifest.json
```

DMC frames carry at minimum: `chromosome`, `start`, `end`, `meth_diff`, `pvalue`, `qvalue`, plus per-test extras (`t_stat`, `chi2`, `phi_hat`, …) and, after `tl.annotate`, `feature_type` / `gene_id` / `cpg_context`. Tile-DMR frames add `tile_id`, `n_cpgs`, `dmr_type ∈ {hyper, hypo, mixed}`.

---

## Module map

| Module | Role |
|--------|------|
| `methyldata.py` | `MethylData` dataclass — `obs`, `store`, `varm`, `uns`; `.dmc` / `.treatment_ids` / `.control_ids` properties; `save()` / `load()` round-trip |
| `io.py`         | `read_bismark`, `read_nfcore_methylseq`, `load` |
| `convert.py`    | `.cov` → partitioned Parquet (`convert_sample`, `ensure_converted_sample`) |
| `filter.py`     | Coverage filter, coverage-quantile normalisation, blacklist intersect, sample summary |
| `pp.py`         | High-level preprocessing wrappers (`filter_coverage`, `normalize_coverage`, `unite`, `smooth`) — record state on `md.uns` |
| `dmc.py`        | Streaming per-CpG accumulators + statistical engines (lr, score, glm, logit_t, beta_binomial, cmh, fisher), BH correction |
| `dmr.py`        | `call_dmr_tile_based`, `call_dmr_sliding_window`, `smooth_methylation_bsmooth` |
| `annotate.py`   | `annotate_features` (GTF promoter / UTR / exon / intron / intergenic), `annotate_cpg_islands` |
| `qc.py`         | `bisulfite_conversion_rate`, `global_methylation_report`, `coverage_uniformity` |
| `tl.py`         | High-level orchestrators: `tl.qc`, `tl.dmc`, `tl.dmr`, `tl.annotate` |
| `pl/`           | Plotting submodules — `pl.qc`, `pl.differential`, `pl.genomic`, `pl.clustering` |
| `cli.py`        | `epykit` CLI entry point |
| `_glm.py`       | Wilkinson-formula → design matrix builder for covariate-adjusted GLM |
| `_style.py`     | Shared matplotlib palette / theme |

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT.
