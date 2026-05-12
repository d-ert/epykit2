"""Layer 3: statistical accuracy on the synthetic fixture.

These tests measure power (sensitivity), empirical FDR (specificity), and
effect-size bias for every DMC backend and both DMR engines against a
ground truth where we *know* which CpGs and which regions are differential.

Why thresholds rather than tight equality? The synthetic data has finite
sample size (8 samples × 5 chr × 2 000 CpGs) and stochastic noise. We pick
thresholds that:

  - any minimally-working implementation would clear easily, AND
  - any of the suspected bugs from the prior audit (dispersion df off by
    one, logit-t variance unbounded near β=0/1, NaN-after-BH sort) would
    almost certainly fail.

Each test exercises one backend so a regression points at the right place.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from tests.conftest import (
    SynthBundle,
    dmr_recovery,
    fdr_at_threshold,
    meth_diff_bias,
    power_at_threshold,
)


# Power / FDR thresholds. Calibrated against the medium fixture
# (4 vs 4 samples, Δβ=0.40, cov~20, replicate_sd=0.03, ~75 k post-filter
# CpGs, BH on 500 true DMCs).
#
# Observed power on the calibration run was LR ≈ 0.60, score ≈ 0.58,
# fisher ≈ 0.45. At 4-vs-4 with df_resid ≈ 6 in the F(1, df_resid)
# reference distribution, the achievable BH-q<0.05 power on 75 k tests
# tops out roughly there — bumping the fixture knobs further would risk
# making the test trivially easy.
#
# Thresholds are set ~5-10 % below observed to tolerate any future
# Monte Carlo wobble while still catching real regressions (especially
# the dispersion-df / logit-t variance bugs flagged in the prior audit).
POWER_MIN_LR = 0.55
POWER_MIN_SCORE = 0.50
POWER_MIN_GLM = 0.45
POWER_MIN_LOGIT_T = 0.20      # logit_t loses power on bounded betas
POWER_MIN_FISHER = 0.35       # high power per site, anti-conservative

FDR_MAX_STRICT = 0.10         # well-calibrated tests at alpha=0.05
FDR_MAX_FISHER = 0.25         # anti-conservative; allow more


def _run_dmc(md, *, test: str, **kwargs):
    """Run ``ep.tl.dmc`` and return the resulting table via ``get_dmc``."""
    import epykit as ep
    ep.tl.dmc(md, test=test, **kwargs)
    df = md.get_dmc(test=test)
    assert df is not None, f"tl.dmc(test={test!r}) wrote no varm entry"
    return df



# Per-backend DMC accuracy


def test_dmc_lr_power_and_fdr(synth_md_filtered, synth_bundle: SynthBundle):
    """The default LR backend: should recover most truly-differential sites
    with FDR close to the nominal alpha.
    """
    df = _run_dmc(synth_md_filtered, test="lr")
    power = power_at_threshold(df, synth_bundle.truth, alpha=0.05)
    fdr   = fdr_at_threshold(df, synth_bundle.truth, alpha=0.05)
    assert power >= POWER_MIN_LR, f"LR power too low: {power:.3f}"
    assert fdr   <= FDR_MAX_STRICT, f"LR FDR too high: {fdr:.3f}"


def test_dmc_score_power_and_fdr(synth_md_filtered, synth_bundle: SynthBundle):
    df = _run_dmc(synth_md_filtered, test="score")
    power = power_at_threshold(df, synth_bundle.truth, alpha=0.05)
    fdr   = fdr_at_threshold(df, synth_bundle.truth, alpha=0.05)
    assert power >= POWER_MIN_SCORE, f"score power too low: {power:.3f}"
    assert fdr   <= FDR_MAX_STRICT + 0.02, f"score FDR too high: {fdr:.3f}"


def test_dmc_logit_t_power_and_fdr(synth_md_filtered, synth_bundle: SynthBundle):
    """Welch t on logit(β) is the most variance-stabilising fallback.
    Power is intentionally lower than LR/score but FDR should stay calibrated.
    """
    df = _run_dmc(synth_md_filtered, test="logit_t")
    power = power_at_threshold(df, synth_bundle.truth, alpha=0.05)
    fdr   = fdr_at_threshold(df, synth_bundle.truth, alpha=0.05)
    assert power >= POWER_MIN_LOGIT_T, f"logit_t power too low: {power:.3f}"
    assert fdr   <= FDR_MAX_STRICT + 0.05, f"logit_t FDR too high: {fdr:.3f}"


def test_dmc_fisher_runs_and_is_powerful_but_anti_conservative(
    synth_md_filtered, synth_bundle: SynthBundle
):
    """Fisher exact pools across replicates; should be powerful but with
    inflated FDR. We assert it *runs* and emits a warning."""
    with pytest.warns(UserWarning, match=r"fisher.*anti-conservative|fisher"):
        df = _run_dmc(synth_md_filtered, test="fisher")
    power = power_at_threshold(df, synth_bundle.truth, alpha=0.05)
    fdr   = fdr_at_threshold(df, synth_bundle.truth, alpha=0.05)
    assert power >= POWER_MIN_FISHER, f"fisher power too low: {power:.3f}"
    # Document expected anti-conservativeness — we don't fail on high FDR.
    assert fdr <= FDR_MAX_FISHER, f"fisher FDR runaway: {fdr:.3f}"


def test_dmc_lr_meth_diff_bias_small(synth_md_filtered, synth_bundle: SynthBundle):
    """Recovered Δβ on truly-DMC sites is approximately unbiased.

    The synthetic fixture now uses **deterministically balanced** ±Δβ
    signs across both DMRs (half hyper, half hypo) and scattered DMCs
    (half hyper, half hypo), so the truth-table mean effect is exactly
    zero. Any residual mean bias must therefore come from the estimator
    itself — coverage weighting, boundary handling near β = 0.7, etc.

    The threshold of 0.04 is loose enough to absorb minor coverage-
    weighted-estimator quirks but tight enough to surface a real bug
    (a sign-convention flip would show |bias| ≈ |Δβ|). If the first
    green run reports a bias well under 0.02, this can be tightened.
    """
    df = _run_dmc(synth_md_filtered, test="lr")
    bias, mae = meth_diff_bias(df, synth_bundle.truth)
    assert abs(bias) < 0.04, f"meth_diff bias too large: {bias:.4f}"
    assert mae < 0.10,       f"meth_diff MAE too large: {mae:.4f}"


def test_dmc_auto_dispatches_to_lr_at_n_gte_2(synth_md_filtered):
    """With 4 replicates per group, test='auto' should resolve to 'lr'."""
    import epykit as ep
    ep.tl.dmc(synth_md_filtered, test="auto")
    used = synth_md_filtered.uns["dmc"]["test_used"]
    assert used == "lr", f"auto should resolve to lr at n>=2, got {used!r}"



# DMR accuracy


def test_dmr_tile_recovers_seeded_regions(synth_md_filtered, synth_bundle: SynthBundle):
    """Tile-based DMR should recover the majority of seeded DMR regions.

    Uses methylKit's default tile_size_bp=1000 (which holds ~4-5 CpGs at
    the fixture's CpG density) and min_cpgs_per_tile=2. With tighter
    settings most tiles fail the min_cpgs filter and are dropped before
    BH, capping recovery far below what the per-CpG signal would suggest.
    """
    import epykit as ep
    ep.tl.dmr(
        synth_md_filtered,
        method="tile",
        tile_size_bp=1000,
        min_cpgs_per_tile=2,
    )
    dmr_df = synth_md_filtered.uns["dmr"]
    n_recovered, n_seeded = dmr_recovery(
        dmr_df, synth_bundle.truth, synth_bundle.config, alpha=0.05,
    )
    # With 10 seeded DMRs, we want at least 5 detected. Loose threshold to
    # tolerate boundary effects (a DMR straddling two tiles may split).
    assert n_recovered >= max(1, n_seeded // 2), (
        f"tile DMR recovered only {n_recovered}/{n_seeded} seeded regions"
    )


def test_dmr_sliding_window_recovers_seeded_regions(
    synth_md_filtered, synth_bundle: SynthBundle
):
    """Sliding-window DMR (lower power than tile) should still recover a
    decent fraction of seeded DMRs after running DMC first.

    Uses window_bp=1000 / min_cpgs=2 to match the tile-test geometry —
    the fixture's CpG density (~225 bp mean spacing) makes a 500/3
    window+min_cpgs filter drop most candidate windows before BH.
    """
    import epykit as ep
    ep.tl.dmc(synth_md_filtered, test="lr")
    ep.tl.dmr(
        synth_md_filtered,
        method="sliding_window",
        window_bp=1000,
        step_bp=500,
        min_cpgs=2,
    )
    dmr_df = synth_md_filtered.uns["dmr"]
    n_recovered, n_seeded = dmr_recovery(
        dmr_df, synth_bundle.truth, synth_bundle.config, alpha=0.05,
    )
    # Sliding window is less powerful — accept a third.
    assert n_recovered >= max(1, n_seeded // 3), (
        f"sliding_window DMR recovered only {n_recovered}/{n_seeded} seeded regions"
    )


def test_dmr_tile_dmr_types_consistent_with_seed_direction(
    synth_md_filtered, synth_bundle: SynthBundle
):
    """For each seeded DMR, the called tile-DMR direction (hyper/hypo)
    should match the sign of the seeded effect size."""
    import epykit as ep
    ep.tl.dmr(
        synth_md_filtered,
        method="tile",
        tile_size_bp=1000,
        min_cpgs_per_tile=2,
    )
    dmr_df = synth_md_filtered.uns["dmr"]
    if "qvalue" in dmr_df.columns:
        sig = dmr_df.filter(pl.col("qvalue") < 0.05)
    else:
        sig = dmr_df
    if len(sig) == 0:
        pytest.skip("no significant DMRs called; direction test not applicable")

    # Build seed direction lookup from truth.
    seed_dirs = (
        synth_bundle.truth.filter(pl.col("in_dmr"))
        .group_by("dmr_id")
        .agg([
            pl.col("chrom").first(),
            pl.col("pos").min().alias("seed_start"),
            pl.col("pos").max().alias("seed_end"),
            pl.col("true_meth_diff").mean().alias("seed_effect"),
        ])
        .to_dicts()
    )

    matched = 0
    correct = 0
    for seed in seed_dirs:
        seed_sign = np.sign(seed["seed_effect"])
        for call in sig.to_dicts():
            if call.get("chrom") != seed["chrom"]:
                continue
            c_lo = call.get("start", call.get("pos"))
            c_hi = call.get("end", call.get("pos"))
            if c_lo is None or c_hi is None:
                continue
            if c_lo <= seed["seed_end"] and c_hi >= seed["seed_start"]:
                matched += 1
                call_diff = call.get("meth_diff") or call.get("mean_meth_diff")
                if call_diff is not None and np.sign(call_diff) == seed_sign:
                    correct += 1
                break
    if matched == 0:
        pytest.skip("no overlapping seeded DMRs in significant calls")
    # At least 80 % of matched DMRs should have correct direction.
    assert correct / matched >= 0.80, (
        f"DMR direction wrong on {matched - correct}/{matched} matched seeds"
    )



# Coverage-weighted Δβ consistency between case and control


def test_dmc_lr_per_group_mean_betas_are_finite_and_in_range(
    synth_md_filtered, synth_bundle: SynthBundle
):
    """``mean_beta_case`` / ``mean_beta_control`` columns should always
    be in [0, 1] and finite for sites that passed filtering."""
    df = _run_dmc(synth_md_filtered, test="lr")
    for col in ("mean_beta_case", "mean_beta_control"):
        if col not in df.columns:
            continue
        vals = df[col].to_numpy()
        finite = np.isfinite(vals)
        assert finite.mean() > 0.99, f"{col} mostly NaN"
        assert ((vals[finite] >= 0) & (vals[finite] <= 1)).all(), (
            f"{col} out of [0, 1]"
        )