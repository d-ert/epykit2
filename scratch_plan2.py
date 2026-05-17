"""Exercise every Plan 2 new feature on the real CD55-vs-control data.

Data: 6 Bismark .cov.gz samples (3 cd55 + 3 control, donors d3/d4, replicates
rep1/rep2). Reads from ``samplesheet.csv``. The Parquet store is built once
into ``plan2_store/`` and reused across the script.

Every section prints what it does, what's new in Plan 2, and a small slice
of the result so you can see the new columns / behaviour without trawling
through full output. Optional-dependency paths skip gracefully when the
extra isn't installed.

Run:    py -3 scratch_plan2.py
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import epykit as ep  # noqa: E402  (matplotlib backend set first)
import polars as pl  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLESHEET = "samplesheet.csv"
STORE_DIR   = "plan2_store"
OUT_DIR     = Path("plan2_scratch_out")
OUT_DIR.mkdir(exist_ok=True)

LO_COUNT = 10
HI_PERC  = 99.9

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename=str(OUT_DIR / "plan2_log.txt"),
    filemode="w",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def hline(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n  {title}\n{bar}")


def show(df, n: int = 5) -> None:
    if isinstance(df, pl.DataFrame):
        print(df.head(n))
    else:
        print(df)


# ---------------------------------------------------------------------------
# Load + preprocess (shared baseline)
# ---------------------------------------------------------------------------
def load_baseline():
    md = ep.read_bismark(
        SAMPLESHEET,
        treatment_group="cd55",
        control_group="control",
        assembly="hg38",
        store_dir=STORE_DIR,
    )
    # Synthesize a `donor` covariate from the sample_id (d3 / d4) so we
    # can demonstrate covariate-adjusted analysis on real data.
    md.obs = md.obs.with_columns(
        pl.col("sample_id").str.extract(r"_(d\d+)_", 1).alias("donor")
    )
    ep.pp.filter_coverage(md, lo_count=LO_COUNT, hi_perc=HI_PERC)
    ep.pp.unite(md, type="intersect")
    print("Loaded MethylData (filtered, intersected):")
    print(md)
    return md


# ---------------------------------------------------------------------------
# §1 Multi-group / continuous-covariate contrasts (formula + contrast)
# ---------------------------------------------------------------------------
def section1_contrasts(md):
    hline("§1 — Multi-group / covariate-adjusted contrasts (formula + contrast)")

    print("""
What's new:
  ep.tl.dmc(md, formula='~ group + donor', contrast='group')
  - formula on md.obs columns (any patsy expression)
  - contrast = column name, factor name, linear combination, or raw matrix
  - covariate-adjusts for donor; tests the group coefficient
  - works for >2 groups (joint F-test) and continuous primaries
""")

    # Binary contrast with covariate adjustment (this dataset is 2-group).
    # If you had 3+ levels, this same call would yield a joint F-test with
    # an f_stat / df1 / df2 column and per-level mean_beta_<level> columns.
    ep.tl.dmc(
        md,
        formula="~ group + donor",
        contrast="group",
        treatment_col="treatment",
    )
    df = md.varm["dmc_glm_contrast"]
    print(f"\nResult schema ({len(df):,} rows):")
    print(df.schema)
    print("\nTop 5 sites by qvalue:")
    cols_to_show = [c for c in (
        "chrom", "pos", "meth_diff",
        "meth_diff_ci_lo", "meth_diff_ci_hi",
        "pvalue", "qvalue",
        "coef_treatment", "coef_se",
        "f_stat", "df1", "df2",
    ) if c in df.columns]
    show(df.sort("qvalue").select(cols_to_show), 5)
    print(f"\nuns['dmc']: {md.uns['dmc']}")


# ---------------------------------------------------------------------------
# §2 Wald CIs on meth_diff + welch_t/bb_lr rename
# ---------------------------------------------------------------------------
def section2_ci_and_rename(md):
    hline("§2 — Wald CIs on meth_diff + welch_t / bb_lr rename")

    print("""
What's new:
  Every DMC test path now writes `meth_diff_ci_lo` / `meth_diff_ci_hi`.
  test='beta_binomial' renamed -> 'welch_t' (deprecation warning).
  test='bb_lr' is a brand-new TRUE quasi-binomial LRT (vs Welch t on betas).
""")

    # Run lr (default), welch_t (new canonical name), bb_lr (new true BB).
    for test in ("lr", "welch_t", "bb_lr"):
        ep.tl.dmc(md, test=test)
        df = md.get_dmc(test=test)
        sig = df.filter(pl.col("qvalue") < 0.05).height if df is not None else 0
        print(f"\n[test={test:<8}] total={len(df):,}  q<0.05={sig:,}")
        cols = [c for c in (
            "chrom", "pos", "meth_diff",
            "meth_diff_ci_lo", "meth_diff_ci_hi",
            "pvalue", "qvalue",
        ) if c in df.columns]
        show(df.sort("qvalue").select(cols), 3)

    # Show the deprecation behaviour explicitly.
    import warnings
    from epykit import dmc as _dmc_mod
    _dmc_mod._WELCH_T_RENAME_WARNED = False  # reset gate
    print("\nCalling test='beta_binomial' (should warn + route to welch_t):")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        ep.tl.dmc(md, test="beta_binomial")
    dep = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    for w in dep:
        print(f"  DeprecationWarning: {w.message}")
    print(f"  result key = dmc_welch_t (present: {'dmc_welch_t' in md.varm})")


# ---------------------------------------------------------------------------
# §3 Permutation-based empirical FDR on DMR
# ---------------------------------------------------------------------------
def section3_permutation_fdr(md):
    hline("§3 — Permutation-based empirical FDR on DMR")

    print("""
What's new:
  ep.tl.dmr(md, empirical_fdr=True, n_perm=N) reruns the tile-DMR engine
  on shuffled treatment labels N times and reports `empirical_pvalue` /
  `empirical_qvalue` columns derived from the null distribution.
  Costly but trustworthy when asymptotic q-values look borderline.
""")

    # Use a small n_perm and restrict to one chromosome to keep this fast.
    ep.tl.dmr(
        md,
        method="tile",
        tile_size_bp=1000,
        empirical_fdr=True,
        n_perm=10,
        perm_seed=42,
        chromosomes=["chr1"],
    )
    dmr = md.uns["dmr"]
    print(f"\nDMR rows: {len(dmr):,}")
    print(f"empirical_fdr params: "
          f"empirical_fdr={md.uns['dmr_params']['empirical_fdr']}, "
          f"n_perm={md.uns['dmr_params']['n_perm']}, "
          f"perm_seed={md.uns['dmr_params']['perm_seed']}")
    if len(dmr) > 0:
        cols = [c for c in (
            "chrom", "start", "end", "n_cpgs", "meth_diff",
            "pvalue", "qvalue",
            "empirical_pvalue", "empirical_qvalue",
        ) if c in dmr.columns]
        show(dmr.sort("qvalue").select(cols), 5)


# ---------------------------------------------------------------------------
# §4 Differential variability (tl.dvc, iEVORA-style)
# ---------------------------------------------------------------------------
def section4_dvc(md):
    hline("§4 — Differential Variability CpGs (tl.dvc)")

    print("""
What's new:
  ep.tl.dvc(md) finds CpGs whose BETWEEN-GROUP VARIANCE differs even when
  the means don't. Standard mean-based DMC analysis misses these entirely.
  iEVORA signature filter: q_variance<alpha AND p_mean>mean_filter_alpha.
""")

    ep.tl.dvc(md, test="bartlett")
    df = md.varm["dvc"]
    print(f"\nDVC table ({len(df):,} sites):")
    print(f"  n_dvc (variance-but-not-mean changes) = "
          f"{int(df.get_column('is_dvc').sum())}")
    print("\nTop 5 by q_variance:")
    show(df.sort("q_variance").head(5), 5)


# ---------------------------------------------------------------------------
# §5 Clinical / cohort QC pack
# ---------------------------------------------------------------------------
def section5_clinical_qc(md):
    hline("§5 — Clinical / cohort QC (sex_check, contamination, correlation, power)")

    print("""
What's new (all opt-in via run_* flags on tl.qc):
  - qc.sex_check          : infers sex from chrX mean β; flags swaps
  - qc.contamination_*    : score = fraction of intermediate-β CpGs
  - qc.sample_correlation : N x N Spearman/Pearson; spots mix-ups
  - qc.power              : DSS-style sample-size / power calculator
""")

    ep.tl.qc(
        md,
        run_sex_check=True,
        run_contamination=True,
        run_sample_correlation=True,
    )

    print("\nobs columns gained from clinical QC:")
    cols = [c for c in md.obs.columns if c in (
        "inferred_sex", "sex_mismatch",
        "contamination_score", "min_pairwise_corr",
        "mean_coverage", "global_methylation",
    )]
    show(md.obs.select(["sample_id"] + cols), 10)

    if "qc_sex_check" in md.uns:
        print("\nsex_check result:")
        show(md.uns["qc_sex_check"], 10)
    if "qc_sample_correlation" in md.uns:
        sc = md.uns["qc_sample_correlation"]
        n = sc.filter(pl.col("sample_a") != pl.col("sample_b")).height
        print(f"\nsample_correlation: {n} off-diagonal pairs cached on "
              f"md.uns['qc_sample_correlation']")
        print(sc.filter(pl.col("sample_a") != pl.col("sample_b"))
                .sort("correlation").head(3))

    print("\nPower / sample-size calculator examples:")
    print(f"  meth_diff=0.10, coverage=20, n=10  ->  power = "
          f"{ep.qc.power(meth_diff=0.10, coverage=20, n_per_group=10):.3f}")
    print(f"  meth_diff=0.10, coverage=20, target power=0.80  ->  n_needed = "
          f"{ep.qc.power(meth_diff=0.10, coverage=20, power=0.80)}")
    print(f"  meth_diff=0.05, coverage=10, target power=0.80  ->  n_needed = "
          f"{ep.qc.power(meth_diff=0.05, coverage=10, power=0.80)}")


# ---------------------------------------------------------------------------
# §6 Visualization pack
# ---------------------------------------------------------------------------
def section6_visuals(md):
    hline("§6 — Visualization pack (umap, correlation heatmap, dashboard, dmr_boxplot)")

    print("""
What's new:
  ep.pl.umap                  : sibling of pl.pca (needs umap-learn)
  ep.pl.sample_correlation    : clustered correlation heatmap
  ep.pl.qc_dashboard          : single composite figure
  ep.pl.dmr_boxplot           : per-DMR per-sample β strip plots
""")

    save_root = str(OUT_DIR / "fig")
    # umap is optional — skip cleanly if missing
    try:
        ep.pl.umap(md, n_neighbors=3, min_dist=0.3, save=f"{save_root}_umap")
        print("  [ok]   pl.umap         -> figures/" + f"{save_root}_umap.png")
    except ImportError as exc:
        print(f"  [skip] pl.umap         ({exc})")
    except Exception as exc:
        print(f"  [FAIL] pl.umap         {exc}")

    try:
        ep.pl.sample_correlation(md, method="spearman",
                                 save=f"{save_root}_sample_corr")
        print("  [ok]   pl.sample_correlation")
    except Exception as exc:
        print(f"  [FAIL] pl.sample_correlation: {exc}")

    try:
        ep.pl.qc_dashboard(md, save=f"{save_root}_qc_dashboard")
        print("  [ok]   pl.qc_dashboard")
    except Exception as exc:
        print(f"  [FAIL] pl.qc_dashboard: {exc}")

    # dmr_boxplot needs an existing DMR table; we have one from §3.
    try:
        ep.pl.dmr_boxplot(md, top_n=6, save=f"{save_root}_dmr_boxplot")
        print("  [ok]   pl.dmr_boxplot")
    except Exception as exc:
        print(f"  [FAIL] pl.dmr_boxplot: {exc}")


# ---------------------------------------------------------------------------
# §7 Ecosystem interop pack
# ---------------------------------------------------------------------------
def section7_interop(md):
    hline("§7 — Ecosystem interop (mudata, methylkit_tabix, multiqc, nfcore)")

    print("""
What's new:
  md.to_mudata()              : MuData wrapper (needs anndata + mudata extras)
  md.to_methylkit_tabix(...)  : write methylKit-readable tabix tables
  ep.report_multiqc(md, dir)  : emit *_mqc.json for MultiQC pickup
  ep.read_nfcore_methylseq_qc : Bismark / Qualimap parse from an nf-core run
""")

    # MuData: silently skipped if extras missing
    try:
        mu = md.to_mudata()
        print(f"  [ok]   to_mudata -> {type(mu).__name__} with mod keys: "
              f"{list(mu.mod.keys())}")
    except ImportError as exc:
        print(f"  [skip] to_mudata ({exc})")
    except Exception as exc:
        print(f"  [FAIL] to_mudata: {exc}")

    # methylKit tabix: always writes text + gzip; tabix index needs pysam.
    try:
        out_dir = md.to_methylkit_tabix(str(OUT_DIR / "methylkit_export"))
        files = sorted(Path(out_dir).glob("*.methylraw.txt.gz"))
        print(f"  [ok]   to_methylkit_tabix -> {len(files)} files in {out_dir}")
        for f in files[:3]:
            print(f"          {f.name}  ({f.stat().st_size:,} bytes)")
    except Exception as exc:
        print(f"  [FAIL] to_methylkit_tabix: {exc}")

    # MultiQC custom-content JSON
    try:
        mqc_dir = ep.report_multiqc(md, str(OUT_DIR / "multiqc"))
        files = sorted(Path(mqc_dir).glob("*_mqc.json"))
        print(f"  [ok]   report_multiqc -> {len(files)} files in {mqc_dir}")
        for f in files:
            print(f"          {f.name}")
    except Exception as exc:
        print(f"  [FAIL] report_multiqc: {exc}")

    # nf-core QC parser — there's no real run dir here, just demonstrate
    # the API call with an empty directory.
    try:
        stub = OUT_DIR / "fake_nfcore_run"
        stub.mkdir(exist_ok=True)
        qc_df = ep.read_nfcore_methylseq_qc(SAMPLESHEET, str(stub))
        print(f"  [ok]   read_nfcore_methylseq_qc -> {len(qc_df)} rows "
              f"(empty run dir, schema only)")
        show(qc_df, 3)
    except Exception as exc:
        print(f"  [FAIL] read_nfcore_methylseq_qc: {exc}")


# ---------------------------------------------------------------------------
# Bonus: the new region_beta helper
# ---------------------------------------------------------------------------
def section_bonus_region_beta(md):
    hline("Bonus — md.region_beta(chrom, start, end) per-region β query")

    print("""
What's new:
  md.region_beta(chrom, start, end) returns per-sample mean β within any
  interval. Used internally by pl.dmr_boxplot; exposed for ad-hoc queries.
""")

    dmr = md.uns.get("dmr")
    if isinstance(dmr, pl.DataFrame) and len(dmr) > 0:
        top = dmr.sort("qvalue").head(1).to_dicts()[0]
        chrom, start, end = top["chrom"], top["start"], top["end"]
        print(f"\nTop DMR: {chrom}:{start:,}-{end:,}  qvalue={top['qvalue']:.3g}")
        rb = md.region_beta(chrom, start, end)
        show(rb, 10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"epykit __version__ = {ep.__version__}")
    print(f"samplesheet        = {Path(SAMPLESHEET).resolve()}")
    print(f"output dir         = {OUT_DIR.resolve()}")

    md = load_baseline()

    section1_contrasts(md)
    section2_ci_and_rename(md)
    section3_permutation_fdr(md)
    section4_dvc(md)
    section5_clinical_qc(md)
    section6_visuals(md)
    section7_interop(md)
    section_bonus_region_beta(md)

    hline("Done")
    print(f"All outputs under {OUT_DIR.resolve()}")
    print("Log:               " + str((OUT_DIR / "plan2_log.txt").resolve()))

    del md
    gc.collect()


if __name__ == "__main__":
    main()
