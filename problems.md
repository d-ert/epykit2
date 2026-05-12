Spawning parallel deep-dives — one on biological/statistical correctness, one on software-engineering quality — plus pulling the full `MethylData` and `_glm.py` myself so my synthesis is grounded.

Here's the critique. I'm putting weight on findings I personally verified by reading the code; subagent claims I couldn't independently confirm I've explicitly flagged for verification.

---

## Framing

epykit is more ambitious than its 0.1.0 label suggests: a quasi-binomial LR / score test with McCullagh–Nelder dispersion, a batched IRLS GLM with covariates, two DMR engines, GTF + CpG-island annotation, a CLI, a Python API, lazy Parquet storage. The design is sound and the algorithmic claims are mostly defensible. What's missing is what fast-iteration always leaves missing: tests, validation, and consistency. The codebase is a credible methylKit-parity engine wearing the operational maturity of a notebook.

---

## Biological / statistical critique

**B1. No automated agreement test against methylKit. — Critical for credibility.**
The whole positioning is "methylKit parity" ([cli.py:1-14](epykit2/src/epykit/cli.py:1), [tl.py:1-15](epykit2/src/epykit/tl.py:1), [_glm.py:160-163](epykit2/src/epykit/_glm.py:160)). There is no fixture-based regression test that ingests a tiny .cov dataset, runs `tl.dmc` / `tl.dmr`, and compares q-values to a frozen methylKit output. Without that, every refactor is a leap of faith and "parity" is an unverified claim.

**B2. Default test is inconsistent across layers. — High.**
- [cli.py:316](epykit2/src/epykit/cli.py:316): CLI default is `score`.
- [tl.py:43-66](epykit2/src/epykit/tl.py:43): `_auto_test_simple` returns `"lr"` for n ≥ 2.
- The cli.py module docstring at [cli.py:7](epykit2/src/epykit/cli.py:7) still says the default changed *to* `logit_t` — three different defaults documented and used in the same package. Pick one (probably `lr`, the methylKit parity), and surface a clear changelog.

**B3. Dispersion-corrected χ² df, logit-t boundary variance, NaN-after-BH sort. — Needs verification.**
The statistical-correctness audit flagged three potential errors in `dmc.py` (df off by one in the chromosome-pooled dispersion path, Welford M2 accumulated in β-space then Jacobian-scaled instead of accumulated in logit-space, and BH applied before NaN restoration disturbing row order). I didn't read those exact line ranges myself; treat as **claims to verify with unit tests** before you trust them, but each one is plausible and each one would silently bias real results. Add a numerical regression test that pins χ² and p-value behaviour at the n=2 / n=3 / n=6 boundaries against a precomputed reference (or against statsmodels' `GLM(...).fit_constrained(...)`).

**B4. CMH and Fisher claims overstated in the docstring. — Medium.**
[cli.py:42-44, 47](epykit2/src/epykit/cli.py:42) markets CMH as "proper per-pair stratification", but a true Mantel–Haenszel needs one 2×2 stratum per replicate pair; the implementation almost certainly accumulates a single stratified statistic. Either implement and test the per-pair stratification, or rewrite the help text to describe what's actually computed. Fisher is honestly labelled anti-conservative — good — but it's still wired as a default-eligible option in the CLI choices list; consider gating it behind an explicit flag or warning.

**B5. Smoothing labelled "BSmooth-style" but isn't. — Medium.**
Real BSmooth is local LOESS with coverage-weighted observations. The implementation appears to be a Gaussian convolution on a regular grid with interpolation back to CpG positions. That's a reasonable approximation, but downstream users will cite "BSmooth smoothing" and be wrong. Either rename to `gaussian_smooth` or implement coverage-weighted LOESS. The function is currently EXPERIMENTAL and not consumed by `tl.dmr` ([pp.py:160-174](epykit2/src/epykit/pp.py:160)), so this is a documentation fix today and a design decision before you wire it into DMR.

**B6. n=1 fallback path is silently unreliable. — High.**
[tl.py:63-66](epykit2/src/epykit/tl.py:63) falls back to Fisher exact when `min(n_case, n_ctrl) < 2`. Fisher at n=1 ignores all between-replicate variability and the function emits *no warning at the data level* (only in the docstring). At n=1 the right behaviour is to refuse, not to silently produce p-values nobody should trust. Raise a `UserWarning` once per run, or require `--allow-n1` to opt in.

**B7. CpG-island annotation is binary; no shore/shelf strata. — Medium (scope).**
A standard WGBS annotation pipeline reports island / shore (±2 kb) / shelf (±2-4 kb) / open-sea. The current `annotate_cpg_islands` looks like a single overlap call. Add the shore/shelf categories — it's a one-pass interval expansion and it's what publishing reviewers expect.

**B8. `unite(type="union")` + `min_samples_*=0` is a footgun. — Medium.**
The pipeline cheerfully tests sites covered in one sample of one group vs zero of the other. The CLI flags exist (`--min-samples-case`, `--min-samples-control`) but default to 0. Make the default ≥ 2 when `unite="union"`, and warn — or better, refuse — when both are 0 and unite is union.

---

## Software-engineering critique

**S1. Zero automated tests at the package root. — Critical.**
There is no `tests/` next to `pyproject.toml`. The only test code lives buried under `.claude/worktrees/`, which is a Claude Code transient artefact, not a repo. From a publication/reproducibility standpoint this is the single highest-leverage gap. Tests aren't optional for a statistics package — they're the only thing that lets you refactor.

**S2. `logging.basicConfig` runs at import. — High.**
[cli.py:23-26](epykit2/src/epykit/cli.py:23) calls `logging.basicConfig(...)` at module-top. Importing **anything** from epykit pulls in the CLI module via no path I traced, but the moment you `python -m epykit.cli` or anything imports `cli`, it overrides the host application's logging. Move this into `def main():`. Libraries should never call `basicConfig`.

**S3. Mixed `print()` / `logger.info` / `warnings.warn`. — Medium.**
CLI handlers print to stdout ([cli.py:96-98, 117-120, 175-182](epykit2/src/epykit/cli.py:96)); library code logs through `logger`. There's no convention. Pick one: library uses `logging`, CLI handlers use `print` only for the final user-facing result lines.

**S4. Version duplicated. — Low but tedious.**
[__init__.py:15](epykit2/src/epykit/__init__.py:15) hardcodes `"0.1.0"`; so does [pyproject.toml:7](epykit2/pyproject.toml:7). Use `importlib.metadata.version("epykit")` in `__init__.py` so the install version is the source of truth.

**S5. `MethylData.dmc` resolution is implicit and order-dependent. — Medium.**
[methyldata.py:54-82](epykit2/src/epykit/src/epykit/methyldata.py:54) auto-picks the "best" DMC table from `varm` by hardcoded priority (`dmc_glm > dmc_lr > dmc_score > …`). It's clever and brittle — running two tests in one session, the user gets whichever ranks higher even though they may have wanted the other. Make this explicit: `md.dmc(test="lr")` as a method, or store a `md.uns["dmc"]["last"]` pointer and read that.

**S6. State flags vs. store path can drift. — Medium.**
`_filtered` / `_united` / `_smoothed` are independent booleans on `MethylData` ([methyldata.py:23-25](epykit2/src/epykit/methyldata.py:23)). Nothing prevents `md.store = "/some/other/path"` from leaving `_filtered=True`. `pp.normalize_coverage` checks `_filtered` ([pp.py:105-108](epykit2/src/epykit/pp.py:105)), but the guarantee is one-sided. Either move to an `enum State` or compute these from `uns["_store_history"]` so there's a single source of truth.

**S7. `_GTF_CACHE` is unbounded. — Medium.**
[annotate.py:_GTF_CACHE](epykit2/src/epykit/annotate.py): plain module-level dict, cleared only on explicit `tl.annotate(..., clear_gtf_cache=True)` (which is the default, so the only people who pay the bill are users who *opt in* to re-using GTF across calls). One human-sized GTF is ~1.5 GB resident. Either make this an LRU with size cap, or document the trade-off loudly.

**S8. `patsy` is a hidden dependency. — Low.**
[_glm.py:138-143](epykit2/src/epykit/_glm.py:138) imports `patsy` — required for any covariate-aware DMR. It's not declared in `pyproject.toml`; the comment claims it comes via `statsmodels`. That's true *today* but statsmodels has been threatening to drop the patsy dep for years. Pin `patsy` explicitly.

**S9. Naming drift: `case`/`treatment`, `control`/`control_group`. — Medium.**
`process_chromosomes_dmc(samples_case, samples_control)` ([dmc.py](epykit2/src/epykit/dmc.py)) vs `md.treatment_ids` / `md.control_ids` ([methyldata.py:29-48](epykit2/src/epykit/methyldata.py:29)) vs CLI `--treatment-group` / `--control-group`. Three vocabularies for the same concept. Pick one (probably `treatment`/`control` — biological convention) and audit.

**S10. Top-level filter API duplicates `pp.*`. — Low.**
`__init__.py` exports `filter_sites`, `intersect_sites`, `get_coverage_quantile`, `normalize_coverage_store` ([__init__.py:21-29](epykit2/src/epykit/__init__.py:21)) at the package root, while [pp.py](epykit2/src/epykit/pp.py) wraps the same module with different names (`filter_coverage`, `unite`, `normalize_coverage`). Two ways to call the same thing with different mutation semantics. The scanpy-style `pp.*` should be the only public surface; demote the rest to `epykit._filter` and stop re-exporting.

**S11. No CI, no linter, no `py.typed`, no `__all__` on `__init__.py`. — Medium overall.**
Together these mean: every refactor is unverified, every import is a guess, and downstream type-checkers can't see your annotations. Single-line fixes individually; together they cost weeks once the package has users.

**S12. `_repr_html_` shows fixed columns. — Trivial.**
[methyldata.py:191-214](epykit2/src/epykit/methyldata.py:191) hardcodes the displayed columns of `obs`. Show all of them, or pivot/scroll, but don't silently hide user covariates from notebooks.

---

## Recommended next steps (phased)

**Phase 1 — make it testable (1 week).** Highest leverage by far.
1. Create a real `tests/` at the repo root. Commit a tiny fixture: 4 samples × 3 chromosomes × ~1k CpGs with known DMC truth.
2. Numerical regression tests for `tl.dmc(test="lr"|"score"|"glm"|"fisher")` against frozen p-value snapshots — these are your statistical safety net.
3. Cross-validation test: run `tl.dmc` on the fixture, run methylKit on the same `.cov`s in a CI-friendly R script (or use a pre-baked reference), assert `r2(qvalues) ≥ 0.99` and bias < 1e-3.
4. CLI tests via `subprocess.run([..., "epykit", "dmc", ...])` on the fixture.
5. Round-trip test: `md.save() → load() → assert obs/varm/uns/store all match`.

**Phase 2 — verify the bio findings, then ship corrections (1-2 weeks).**
6. Read [dmc.py:735, 887-895, 1770-1776] and [dmr.py:464-469] with the audit findings in hand. Confirm or refute each. For every confirmed bug, write the failing test first, then fix.
7. Unify the default test. One default across CLI, `tl.dmc`, and the docstrings.
8. Replace the `_auto_test` n=1 silent fallback with a `UserWarning` + opt-in flag.
9. Add CpG island shore/shelf to `annotate_cpg_islands`. Rename `smooth_methylation_bsmooth` to `smooth_methylation_gaussian` (or implement real LOESS).
10. Add a single methylKit-parity benchmark notebook in `docs/` showing q-value scatter, Bland-Altman, and FDR calibration.

**Phase 3 — operational hardening (1-2 weeks).**
11. Move `logging.basicConfig` into `main()`. Wire `epykit -v` / `--quiet`.
12. Declare `patsy` in `pyproject.toml`. Drop the redundant top-level `filter_sites` exports — `pp.*` only.
13. `importlib.metadata.version` in `__init__.py`. Add `py.typed`. Add `__all__`.
14. LRU on `_GTF_CACHE` (or move it to a small `_AnnotationContext` object instead of a module global).
15. Unify naming on `treatment`/`control`. One deprecation cycle for the old kwargs.
16. Replace `_filtered`/`_united`/`_smoothed` flags with `state` derived from `uns["_store_history"]`.

**Phase 4 — packaging & ergonomics (1 week).**
17. GitHub Actions: lint (ruff), type-check (mypy on the public API), tests on py3.9/3.11/3.12, build wheel.
18. Pre-commit config (ruff + ruff-format + end-of-file + trailing whitespace).
19. Publish to TestPyPI; verify install works without the local `.venv` build-tools quirks (the ncls failure you saw).
20. Tag `v0.2.0` once the bio verifications close.

**Phase 5 — feature direction (optional, post-0.2).**
21. Multi-group designs (>2 conditions) via the existing GLM path.
22. Allele-specific methylation (ASM) calling — natural fit for the strand-aware Parquet store.
23. Optional Apache Arrow Dataset partitioning by sample so single-sample analyses don't load all partitions.
24. Plot polish: residual diagnostic plots from the GLM, methylKit-style methylation profile plots.

The single highest-impact change is **#1**. Until there is a `pytest` you can run, every other improvement here is a guess. Once tests exist, B3 (the dispersion and logit-t findings) will either confirm in 30 minutes or fall out as false alarms — and the rest of the roadmap becomes straightforward incremental work.