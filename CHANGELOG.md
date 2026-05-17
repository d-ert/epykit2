# Changelog

All notable changes to **epykit** are tracked here. Format roughly follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses
SemVer (`MAJOR.MINOR.PATCH`).

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
