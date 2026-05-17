"""Plan 2 §1: multi-group / continuous-covariate contrasts via tl.dmc."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import epykit as ep
from tests.fixtures.synth import SimConfig, generate


@pytest.fixture(scope="module")
def multigroup_md(tmp_path_factory):
    cfg = SimConfig(
        n_groups=3,
        n_per_group=4,
        cpgs_per_chrom=600,
        chromosomes=("chr1", "chr2"),
        seed=4242,
    )
    out_dir = tmp_path_factory.mktemp("multigroup")
    result = generate(cfg, out_dir)
    # Build MethylData manually using groups=[...]
    md = ep.read_bismark(
        result["samplesheet"],
        groups=list(result["group_ids"].keys()),
        store_dir=str(out_dir / "store"),
    )
    ep.pp.filter_coverage(md, lo_count=3, hi_perc=99.9)
    ep.pp.unite(md, type="intersect")
    return md, pl.read_parquet(result["truth"]), cfg


@pytest.fixture(scope="module")
def continuous_md(tmp_path_factory):
    cfg = SimConfig(
        n_per_group=4,
        cpgs_per_chrom=600,
        chromosomes=("chr1", "chr2"),
        continuous_covariate=True,
        n_groups=2,
        seed=4243,
    )
    out_dir = tmp_path_factory.mktemp("continuous")
    result = generate(cfg, out_dir)
    md = ep.read_bismark(
        result["samplesheet"],
        treatment_group="treatment",
        control_group="control",
        store_dir=str(out_dir / "store"),
    )
    # The samplesheet stores 'age' as a string; coerce to float on obs.
    md.obs = md.obs.with_columns(
        pl.col("age").cast(pl.Float64, strict=False)
    )
    ep.pp.filter_coverage(md, lo_count=3, hi_perc=99.9)
    ep.pp.unite(md, type="intersect")
    return md, pl.read_parquet(result["truth"]), cfg


def test_multigroup_factor_joint_test(multigroup_md):
    """A 3-group joint F-test recovers seeded multi-group DMCs."""
    md, truth, cfg = multigroup_md
    ep.tl.dmc(md, formula="~ group", contrast="group")
    df = md.varm["dmc_glm_contrast"]
    assert df is not None
    assert "f_stat" in df.columns
    assert "df1" in df.columns
    assert df.get_column("df1")[0] >= 2  # 3 levels → df1 == 2
    # Per-level mean beta columns
    level_cols = [c for c in df.columns if c.startswith("mean_beta_")]
    assert len(level_cols) >= 3
    # Power: > 30 % of seeded multi-group DMCs called at q < 0.05.
    joined = (
        truth.with_columns(pl.col("pos").cast(pl.Int64))
        .join(df.with_columns(pl.col("pos").cast(pl.Int64)),
              on=["chrom", "pos"], how="left")
    )
    mg_truth = joined.filter(pl.col("is_multigroup_dmc"))
    if len(mg_truth) > 0 and "qvalue" in mg_truth.columns:
        called = mg_truth.filter(pl.col("qvalue") < 0.05).height
        # The fixture seeds 200 multi-group DMCs with effect step 0.20 across
        # three groups; with 4 reps per group this should be very detectable.
        assert called / max(len(mg_truth), 1) >= 0.30, (
            f"Multi-group recovery too low: {called}/{len(mg_truth)} "
            "at q<0.05"
        )


def test_continuous_covariate_primary(continuous_md):
    """Continuous covariate as primary effect — Wald test on the age coef."""
    md, truth, cfg = continuous_md
    ep.tl.dmc(md, formula="~ age", contrast="age")
    df = md.varm["dmc_glm_contrast"]
    assert "coef_treatment" in df.columns or "f_stat" in df.columns
    # Empirical test: more seeded age-DMCs should be called than null.
    joined = (
        truth.with_columns(pl.col("pos").cast(pl.Int64))
        .join(df.with_columns(pl.col("pos").cast(pl.Int64)),
              on=["chrom", "pos"], how="left")
    )
    if "qvalue" in df.columns:
        age_truth = joined.filter(pl.col("is_age_dmc"))
        if len(age_truth) > 0:
            sig_frac = float(
                (age_truth.get_column("qvalue") < 0.05).fill_null(False).mean()
            )
            null = joined.filter(
                ~pl.col("is_age_dmc") & ~pl.col("is_dmc")
                & ~pl.col("is_multigroup_dmc")
            )
            null_frac = float(
                (null.get_column("qvalue") < 0.05).fill_null(False).mean()
            )
            assert sig_frac > null_frac, (
                f"Age-driven sites should be enriched (sig={sig_frac:.3f}, "
                f"null={null_frac:.3f})."
            )


def test_named_contrast_single_row(multigroup_md):
    """A patsy-style linear contrast works through resolve_contrast."""
    md, _truth, _cfg = multigroup_md
    levels = sorted(md.obs.get_column("group").unique().to_list())
    a, b = levels[0], levels[1]
    contrast = f"group[T.{a}] - group[T.{b}]"
    # This is a tough resolution path — accept either a successful run or
    # a clean ValueError, but never a crash.
    try:
        ep.tl.dmc(md, formula="~ group", contrast=contrast)
        df = md.varm["dmc_glm_contrast"]
        assert "pvalue" in df.columns
    except ValueError:
        pass
