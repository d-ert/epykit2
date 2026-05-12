Comparing epykit against methylKit, bsseq + dmrseq, DSS, methylpy, RnBeads, and the scanpy-style ecosystem it borrows from — here's what's genuinely absent that users of those tools would expect.

> All of this is *additional* to the items in the prior roadmap (tests, methylKit-parity benchmark, CpG shore/shelf, real BSmooth LOESS, multi-default cleanup). Don't restart those.

---

## Statistical methods present elsewhere, absent here

| Feature | Where it exists | Why it matters | Effort |
|---|---|---|---|
| **Multi-group / >2-condition contrasts** (F-test across groups) | methylKit `calculateDiffMethPerChr` with multiple treatments, DSS `DMLtest.multiFactor` | Studies with 3+ conditions (e.g. WT / het / KO) currently force users into pairwise hacks. Your GLM path can already fit it — just expose the contrast specification. | Low — `_glm.py` already does batched LRT; add `tl.dmc(contrasts=...)` |
| **Continuous covariate as primary effect** (e.g. age, dosage) | DSS `DMLtest`, methylKit covariate slot, RnBeads | All current entry points assume binary treatment. The GLM accepts continuous covariates as confounders but not as *the* effect of interest. | Low (parameter rewiring) |
| **Differential variability** (iEVORA / BiVA) | iEVORA, methylSpectrum | Cancer / aging signatures often show variance changes without mean changes. A few hundred lines, big scientific differentiator. | Medium |
| **Effect-size shrinkage** (apeglm-style on methylation differences) | dmrseq, DSS (Bayesian shrinkage on dispersion *and* effect) | Low-coverage sites currently produce huge meth_diffs that survive q < 0.05 by luck. Shrinkage reduces this. | Medium |
| **Empirical-null / permutation FDR** | dmrseq's permutation-based p-values, methylKit `getMethylDiff(qvalue.method="qvalue")` | Asymptotic q-values are sometimes wildly miscalibrated on WGBS. Adding a `tl.dmr(empirical_fdr=True, n_perm=100)` is a strong validation knob. | Medium |
| **Wald CIs on `meth_diff`** | methylKit reports them; DSS does too | The GLM in `_glm.py` already computes `se_beta` (line 304). Surfacing 95% CIs is a one-column addition. | Trivial |
| **HMM- / segmentation-based DMR** | methylKitDB `methylKit::regionCounts`, methylpy DMR-find HMM, dmrseq is segmentation-aware | Tile-based DMRs miss boundaries; HMM finds them. Different tool for a different question. | High |
| **Beta-binomial done properly** | DSS, MOABS, methylSig | Today `test="beta_binomial"` is, by your own docstring, a Welch t on raw betas — not a BB likelihood. Either implement true BB or rename to `welch_t`. | Medium |

---

## Biological analyses present elsewhere, absent here

These are the features bioinformaticians comparing tools will look for first.

- **Custom-region aggregation (user-supplied BED).** methylKit's `regionCounts` lets the user pass any BED (promoters, enhancers, ATAC-peaks, ChIP peaks, super-enhancers) and aggregates reads to those intervals. This is the single most-requested missing piece relative to methylKit. Today you can only aggregate to fixed tiles. → `pp.aggregate_regions(md, regions_bed=...)`.
- **Allele-specific methylation (ASM).** methylpy `allc_to_bed`, methHaplo. Requires linkage to a VCF and read-level info, so it's a substantial addition — but it opens imprinting and X-inactivation analyses.
- **PMD detection** (partially methylated domains, Mb-scale). methylpy `pmd`, methylKitDB. Defining HMR / PMD / LMR is standard for cancer methylomes.
- **HMR / LMR / UMR calling.** Below-threshold contiguous regions. ENmix, methylpy. Adjacent to PMD; both useful.
- **Methylation entropy / co-methylation.** methclone, methylpy `mch_level`. Quantifies within-cell heterogeneity from co-methylated reads. Needs BAM input, not .cov — a real architectural extension.
- **CHG / CHH as first-class.** Plants and ESC studies require this. The converter and methylstore are context-aware ([convert.py](epykit2/src/epykit/convert.py)), but `tl.dmc` / `tl.dmr` paths in [tl.py](epykit2/src/epykit/tl.py) implicitly assume CpG. A `context=` switch on the analysis namespace.
- **Cell-type deconvolution.** EpiDISH / CIBERSORT-style on bulk methylation. Heavily used in clinical EWAS. Provide reference matrices for blood / brain / placenta and a single deconvolve API call.
- **Epigenetic age clocks** (Horvath, Hannum, PhenoAge, GrimAge, DunedinPACE). A `qc.epigenetic_age(md, clock="horvath")` reading frozen coefficient tables would land you a clinical-genomics audience for almost no code.
- **Methylation imputation.** METHimpute, methyLImp. Especially important if you're going to support array data later or sparse-coverage WGBS.
- **eQTM / methyl-expression integration.** `tl.correlate(md, expression_matrix=...)`. Methylation-expression Pearson/Spearman with FDR. RnBeads has it.
- **Gene-set enrichment on DMR-associated genes.** methylGSA, missMethyl, ReactomePA wrapping. CpG-biased gene-set tests (genes with more CpGs are more likely "significant" — needs probability-of-being-tested correction).
- **TFBS / motif enrichment at DMR sequences.** Standard wrap of HOMER / MEME, or in-Python via `pyjaspar` + `pyranges` for JASPAR PWMs.
- **CpG island shore/shelf strata.** Already in your prior roadmap — flagging here because it's the methylKit-parity item users will immediately notice.

---

## I/O & ecosystem interop

The Parquet methylstore is your differentiator, but it's an island. Users live in a tool stack.

- **BedGraph / BigWig export.** This is the most painful missing piece for any user with an IGV / UCSC habit. `md.to_bedgraph(...)`, `md.to_bigwig(...)` (via `pyBigWig`). Also: export DMRs as BED and DMCs as VCF-like.
- **AnnData / MuData interop.** You already chose the scanpy namespace (`pp`/`tl`/`pl`) — close the loop. `md.to_anndata()` / `md.to_mudata()` would put epykit results in the same notebook as scRNA-seq / ATAC-seq downstream. Critical for multi-omics groups.
- **methylKit `methylRaw` / `methylDiff` interop.** Either a `methylKit_io.py` reader, or — more pragmatic — `md.to_methylkit_tabix(...)` so reviewers can re-run methylKit on your output.
- **HDF5 / Zarr backing alongside Parquet.** Zarr is the scanpy / single-cell direction. Parquet is great for OLAP; Zarr for random-access genomic ranges. Not urgent, but eventually.
- **MultiQC plugin.** A simple `qc.report_multiqc(md, output="multiqc_data/")` emitting MultiQC-readable JSON would let your QC numbers flow into every standard pipeline report.
- **nf-core/methylseq tighter integration.** `read_nfcore_methylseq` exists but is a samplesheet reader. Picking up the pipeline's QC JSON (Bismark report, Qualimap, preseq) and joining onto `md.obs` automatically would make epykit the natural downstream node.
- **VCF integration for ASM** (when/if ASM lands).
- **Tabix indexing on the Parquet partitions** for random-range queries (`md.region("chr7", 27_000_000, 27_500_000)`). Right now there's no obvious random-access API.

---

## Visualization gaps

The current `pl.*` set is solid for differential analysis but light on the *standard* WGBS figures.

- **TSS / gene-body metaplot** (mean methylation profile ±N kb around TSS, exon boundaries, CpG-island edges). This is in every WGBS paper. deeptools `plotProfile` is the de-facto standard outside R; replicate it.
- **DMR region boxplots** — for each top DMR, per-sample beta distribution as a strip/box plot. Lets users sanity-check "this DMR is real."
- **Genome-browser-style track plot** — pyGenomeTracks-style multi-track view of beta, coverage, and DMR calls in a window. Essential for talks.
- **Sample correlation heatmap** — Pearson/Spearman across samples on the methylation matrix.
- **UpSet / Venn** of DMC or DMR overlaps across contrasts (when you have >1 contrast).
- **UMAP / t-SNE** in addition to PCA, for many-sample studies (`pl.umap`).
- **Karyogram / chromosome painter** of DMR density per chromosome. Cancer methylomes especially.
- **QC dashboards** — one composite figure: per-sample conversion rate, coverage, global meth, methylation distribution. Currently these exist as individual plots; the user has to assemble.

---

## Reproducibility & UX gaps

- **Auto-generated HTML report** (à la RnBeads, MultiQC). One call: `md.report("report.html")` → samples, QC, DMCs, top DMRs with links, plots. Massive UX win.
- **Checkpoint / resume API.** The `_store_history` infra is there but users have no `md.resume_from("filtered")` to skip an expensive step.
- **Power / sample-size calculator.** DSS has `powerEst`. Methylation-specific because effect sizes are bounded; a `qc.power(meth_diff=0.1, coverage=10, n=...)` would be a teaching tool as well.
- **Sample swap / sex check / contamination detection** from methylation patterns (X-chromosome global meth → sex; ID SNP-CpGs → swap). Critical for clinical cohorts.

---

## Performance / scale

Less urgent but worth flagging since the Parquet backend invites these.

- **Polars LazyFrame end-to-end.** A lot of the current code calls `.collect()` early ([filter.py](epykit2/src/epykit/filter.py)). Pushing lazy evaluation through the DMC engine would let the same code run on 10× larger cohorts.
- **Dask / Ray** for cross-chromosome parallelism. The per-chromosome loop in `process_chromosomes_dmc` is embarrassingly parallel; today it's sequential.
- **GPU path** for the IRLS GLM (cupy). For 10k-sample EWAS-style designs the batched solve in `_glm.py:_solve_weighted_lsq` is the hot path.
- **Single-cell methylation (scBS-seq, snmC-seq)** support — different architecture (per-cell sparse data), but the Parquet partitioning model is actually a good fit. Future direction.

---

## If I had to pick six to add first

Ranked by **(user demand × low effort)**:

1. **Custom-region aggregation from a user BED.** Top methylKit feature. ~200 lines on top of the existing tile path.
2. **BedGraph / BigWig export.** Unblocks every IGV user. ~50 lines with `pyBigWig`.
3. **TSS / metaplot.** The single most-cited WGBS figure. ~150 lines, integrates with GTF cache already in `annotate.py`.
4. **AnnData export.** Locks in the scanpy-ecosystem positioning. ~100 lines.
5. **Multi-group / continuous-covariate primary contrasts.** GLM engine already supports it — surface it via `tl.dmc(contrasts=...)`.
6. **HTML report.** Closes the UX loop and gives every project a single shareable deliverable.

Each of these is roughly a day or two if the test scaffolding from the prior roadmap is in place. The first four would, on their own, take epykit from "internal-use methylKit-parity engine" to "credible Python alternative to methylKit + bsseq + dmrseq."