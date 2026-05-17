# Changelog

All notable changes to **epykit** are tracked here. Format roughly follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses
SemVer (`MAJOR.MINOR.PATCH`).

## [0.6.0] — HMM segmentation

One shared HMM engine, three callers on top. All three operate on the
existing methylstore (no new I/O path) and route through the 0.4
chrom-streaming dispatcher, so the distributed backend works for free.
No new dependencies — the HMM is hand-rolled in ~200 LoC of numpy.

### Added

#### Shared HMM engine
- **`epykit._hmm.segment(observations, n_states, ...)`**. Hand-rolled
  forward-backward + Viterbi for either Bernoulli or Gaussian
  emissions, with a sticky-chain transition prior (configurable
  ``self_loop`` and full ``transition_priors`` overrides).
- **`epykit._hmm.runs_of_state(viterbi, target_state, positions)`**
  extracts contiguous runs of a target state from the Viterbi path,
  optionally translated to genomic positions.

#### Callers
- **`ep.tl.pmd(md)`** — partially methylated domains. Per-sample,
  megabase-scale 2-state HMM on coverage-weighted smoothed β.
  Output in ``md.uns["pmd"]``.
- **`ep.tl.hmr(md)`** — hypo- and low-methylated regions
  (MethylSeekR-style). Per-sample 2-state HMM on raw per-CpG β.
  Tagging splits the runs into ``md.uns["hmr"]`` (dense, CpG-island-
  like) and ``md.uns["lmr"]`` (sparse, distal regulatory).
- **`ep.tl.dmr(md, method="hmm")`** — HMM-based DMR caller. Three-state
  Gaussian HMM on the per-CpG ``meth_diff`` signal from any DMC table.
  Schema-compatible with ``method="tile"``, so existing plot / export
  paths work unchanged.

### Deferred to 0.7+

Zarr storage backing, single-cell sparse stores, eQTM, motif/TFBS
enrichment, matrix-completion imputation beyond kNN, pyGenomeTracks
track plot, differential entropy CpG-window test.

---

## [0.5.0] — BAM-based read-level analyses

Adds the first analyses that need read-level methylation information.
Builds on the 0.4 engine lifts but doesn't change any existing default
behaviour. New optional `bam` extra (pysam, Linux/macOS only) gates
the new analyses; everything else stays installable on Windows.

### Added

#### BAM ingestion
- **`epykit.bam_io.read_methylation_calls(bam, ...)`**. Returns a
  long-form polars DataFrame with one row per (read, covered CpG):
  `(read_id, chrom, pos, methylation_status, base_qual, mapq,
  mate_pair_id, strand, allele_base)`. Two BAM dialects: Bismark `XM`
  tags and SAM-standard `MM`/`ML` tags (MethylDackel).
- **`bam` optional extra** in `pyproject.toml` (`pysam>=0.22`).
  Linux/macOS only — pysam has no Windows wheel.

#### Allele-specific methylation (ASM)
- **`ep.tl.asm(md, bam=..., vcf=...)`**. Per-CpG Fisher exact test of
  H1 vs H2 read methylation, with heterozygous SNVs from a VCF as
  phasing anchors. Result lands in `md.varm["asm"]` with columns
  matching the `dmc_*` family (`pvalue`, `qvalue`, `meth_diff`) so
  `pl.volcano(md, key="asm")` works without modification.
- Reuses `fisher_exact_vectorized` and `apply_multiple_testing_correction`
  from the DMC stack.

#### Methylation entropy
- **`ep.tl.entropy(md, bam=..., window_cpgs=4)`**. Per-CpG-window
  Shannon entropy over the observed read methylation patterns. Reads
  that cover all CpGs in the window contribute one binary pattern; the
  full distribution's Shannon entropy is normalised to `[0, 1]`.
  Result lands in `md.varm["entropy"]`.

### Deferred to 0.6

PMD / HMR / LMR callers and HMM-DMR all share an HMM segmentation
engine; they land together in the 0.6 release.

---

## [0.4.0] — engine lifts

Infrastructure release. Default behaviour unchanged (every 0.3 test
passes verbatim); every new capability is reached by an explicit kwarg
or optional extras install. No new analyses — those land in 0.5.

### Added

#### Compute backends
- **Distributed compute via Dask** (`tl.dmc(..., backend="dask", n_workers=4)`,
  `tl.dmr`, `tl.dvc`). Per-chromosome work submitted to a local cluster
  or an existing `dask.distributed.Client`. Requires the new
  `pip install 'epykit[distributed]'` extra. Results are bit-identical
  to the sequential path.
- **Distributed compute via Ray** (`backend="ray"`). Same surface as
  the Dask backend; uses `ray.remote` actors. Requires
  `pip install 'epykit[ray]'`.
- **GPU IRLS via CuPy** (`tl.dmc(test="glm", glm_backend="gpu")`). The
  batched binomial IRLS hot path in `_glm.py` now has a CuPy mirror
  (`_glm_gpu.py`). Requires `pip install 'epykit[gpu]'` (CUDA 12). The
  closed-form `lr` / `score` tests stay CPU-only by design.

#### Pipeline manifest + resume
- **Formal checkpoint / resume API.** Each `MethylData` analysis root
  now hosts a top-level `.epykit_manifest.json` recording completed
  pipeline stages with their input signatures and sidecar parquet
  paths. Call `ep.tl.dmc(md, ..., resumable=True)` twice with the same
  inputs and the second call loads the cached result instead of
  recomputing. `md.completed_stages` reports the recorded list;
  `md.resume_from("dmc_lr")` re-hydrates a fresh `MethylData` from the
  on-disk manifest.

#### Tabix-on-Parquet random access
- **`ep.query` module.** Three entry points —
  `query_region(store, chrom, start, end)`,
  `query_regions(store, regions_df)`, and
  `query_sites(store, sites_df)` — return long-form
  `(sample_id, chrom, pos, strand, N_meth, coverage, beta)` frames for
  arbitrary genomic loci. No new dependency: built on
  `pl.scan_parquet` predicate pushdown over the existing
  hive-partitioned store.

### Internal

- **Per-chromosome compute dispatcher** at
  `src/epykit/_compute.py:run_chrom_pipeline`. The chrom loops in
  `dmc.py`, `dmr.py` (tile), and `dvc.py` were refactored to route
  through this shared dispatcher. `backend="sequential"` (default) is
  bit-identical to the pre-0.4 in-line loop.
- **`irls_dispatch`** in `_glm.py` routes `irls_binomial_batch`
  between CPU (numpy) and GPU (CuPy via `_glm_gpu.py`).

### Deferred to 0.5+

Out of scope for 0.4: ASM, methylation entropy, PMD, HMR/LMR, HMM-DMR,
Zarr backing, single-cell sparse stores, eQTM, motif/TFBS enrichment,
matrix-completion imputation, pyGenomeTracks-style track plot.

---

## [Unreleased] — 0.3.0

### Added

#### Visualization
- **Karyogram / chromosome painter** (`ep.pl.karyogram`). One row per
  chromosome, megabase-binned mean of any per-CpG metric (`meth_diff`,
  `-log10_qvalue`, raw β). RdBu_r by default; symmetric colour limits
  for signed metrics.
- **DMR UpSet / Venn overlap** (`ep.pl.dmr_overlap`). 2-set inputs fall
  back to a 2-circle Venn; 3-6 set inputs render an UpSet plot
  (bar chart + dot matrix + per-set totals). Matplotlib-only; no
  upsetplot or matplotlib_venn dependency.
- **Gene-body metaplot** (`ep.pl.gene_body_metaplot`). Three-zone TSS /
  body / TES plot with flanking windows; body length is normalised
  across genes so a 50 kb and a 500 kb gene contribute equally.

#### Statistical features
- **DVR — Differentially Variable Regions** (`ep.tl.dvr`,
  `ep.call_dvr_density`). Region-level aggregation of `tl.dvc` output
  via per-tile DVC-density enrichment (one-sided binomial vs the
  genome-wide rate, BH-corrected). Avoids the variance-statistic
  combining problem that defeats Fisher / Stouffer's at the
  per-CpG level.
- **Effect-size shrinkage** (`ep.shrink_meth_diff`). Empirical-Bayes
  Normal-prior James-Stein-style shrinkage of `meth_diff` toward zero.
  Adds `meth_diff_shrunk`, `meth_diff_se`, `shrinkage_factor` columns
  to a DMC table. Same spirit as `ashr` / `apeglm`; pulls
  low-coverage sites harder than well-powered ones.
- **kNN β imputation** (`ep.impute_knn_beta`, `ep.impute_knn_anndata`).
  Per-chromosome inverse-distance-weighted kNN over genomic position;
  optional `max_distance_bp` cap so cross-CGI gaps don't pull
  long-distance neighbours.

#### Clocks / deconvolution scaffolding
- **Generic linear age-clock runner** (`ep.tl.age_clock`,
  `ep.age_clock`). Takes a user-supplied `(cpg_id, coefficient)` table
  and a probe → `(chrom, pos)` manifest, computes per-sample age, and
  writes the result into `md.obs`. Supports an optional `transform`
  argument (`"horvath"` for the standard ≥20-year piecewise
  anti-transform); other published clocks (Hannum, PhenoAge,
  DunedinPACE) plug in via their own coefficient CSVs. Coefficient
  tables and manifests are **not bundled** — licensing and probe
  vendor specifics differ per clock.
- **Reference-based cell-type deconvolution** (`ep.tl.deconvolve`,
  `ep.deconvolve`). Non-negative least squares solve against a
  user-supplied reference β matrix (EpiDISH / CIBERSORT / Houseman
  style). Long-format result on `md.uns['deconvolution']`; wide
  per-cell-type columns (`frac_<celltype>`) joined onto `md.obs`.

### Tests
- New: `test_viz_new.py` (8 tests), `test_dvr.py` (6 tests),
  `test_stats_new.py` (12 tests) covering shrinkage, kNN imputation
  end-to-end on AnnData, age-clock recovery on synthetic data, and
  deconvolution NNLS round-trip.

### Notes on what's *still* left on the 0.3+ roadmap

Architectural lifts that didn't fit this round and remain unimplemented:
ASM (per-read haplotype phasing from BAM), PMD / HMR / LMR callers,
methylation entropy, single-cell methylation, distributed compute
(Dask / Ray), GPU IRLS, HMM-based DMR, Zarr backing, Tabix-on-Parquet,
formal checkpoint / resume API. These each warrant their own focused
release; the karyogram + UpSet + gene-body trio plus DVR / shrinkage /
imputation / clocks / deconvolution covers the highest-ROI subset of
the roadmap.

---

## [0.2.0]

### Added
- **MethylDackel input adapter.** `ep.read_methyldackel(samplesheet, ...)` and
  `epykit convert --format methyldackel` ingest MethylDackel `.bedGraph[.gz]`
  output through the same partitioned-Parquet pipeline as Bismark. The
  conversion cache is format-aware so a Bismark store cannot be silently
  reused for MethylDackel input.
- **Permutation empirical FDR for DMC.** `ep.tl.dmc(..., empirical_fdr=True,
  n_perm=100)` shuffles treatment / control labels, re-runs the per-CpG
  DMC engine, and emits `empirical_pvalue` / `empirical_qvalue` columns —
  parity with the existing DMR permutation FDR. Refused with `formula=` /
  `contrast=` designs (label shuffling invalidates stratified models).
- **Bismark M-bias parser and plot.** `nfcore_qc.parse_bismark_mbias(path)`
  parses the Bismark M-bias text format into a long table; `pl.mbias_plot(
  {sample_id: df_or_path})` renders percent methylation per read position
  with context / R1 / R2 lines.
- **CLI `--version` flag.** `epykit --version` prints the installed
  `__version__` from `importlib.metadata`.
- **DVC engine re-export.** `ep.process_chromosomes_dvc` is now part of the
  public surface alongside the high-level `ep.tl.dvc` orchestrator.

### Changed
- **GLM degeneracy is visible.** When the batched `(X'WX)⁻¹` solve in
  `_glm.py` falls back to a per-site solve, the helper emits a
  `logger.warning` summarising affected sites instead of failing silently.
  Per-site GLM separation is logged at `info` (or `warning` when ≥5 % of
  sites separate) rather than `debug`.
- **Bisulfite conversion rate is reported, not applied.** Doc clarification
  in `qc.bisulfite_conversion_rate` and the README: epykit follows
  `bsseq` / `methylKit` defaults and does not rescale per-CpG counts by the
  conversion rate. The rate is surfaced through QC, MultiQC export, and
  the HTML report so users can gate on it.
- **Multi-group DMC accuracy test.** `tests/test_dmc_multigroup.py` now
  asserts power (≥30 %) and FDR (≤15 %) on the 3-group joint F-test
  fixture instead of only checking column presence. Continuous-covariate
  test is now an explicit structural check (engine runs, finite
  p-values, FDR isn't catastrophic) with the fixture limitation
  documented in the test.
- **DMR sensitivity / specificity tests.** `test_accuracy.py` gains
  `test_dmr_tile_sensitivity_and_fdp` and
  `test_dmr_sliding_window_sensitivity_and_fdp`, pinning both recovery
  rate and false-discovery proportion per method. Conditional
  `pytest.skip` calls in the existing DMR direction test are now hard
  assertions to surface regressions instead of masking them.

### Fixed
- `tests/test_dmc_multigroup.py` no longer silently swallows a `ValueError`
  in the contrast-resolution test; on `ValueError` the test now asserts
  the error message is informative and on success it asserts finite
  p-values are produced.

## [0.1.0]

- Initial release. Bismark `.cov` → partitioned Parquet methylstore;
  scanpy-style `pp` / `tl` / `pl` API; 8 DMC test backends; two DMR
  engines plus permutation FDR; DVC (iEVORA-style); covariate-aware
  contrasts; AnnData / MuData / methylKit / MultiQC interop;
  self-contained HTML report.
