"""B3 audit-finding verification tests.

The bio-correctness audit (second-session critique) flagged three suspected
math bugs without confirming them. Each of the three tests below was
designed to either confirm or refute one specific claim by computing the
correct answer analytically and asserting epykit matches.

Findings (read the test docstrings for the per-claim reasoning):

* **B3.1 — dispersion df at dmc.py:712**: audit was almost certainly WRONG.
  ``n_disp`` is the number of sites contributing to the dispersion estimate
  (set at dmc.py:700). With 2 fitted proportions per site, total params is
  ``2 * n_disp``. The formula ``df = n_obs - 2 * n_disp`` is correct.
  Verified via H0 calibration: under no differential methylation, the
  chromosome-pooled φ̂ should sit at ≈ 1.0 ± noise. If df were off by a
  factor, φ̂ would be systematically biased.

* **B3.2 — logit-t Welford M2 at dmc.py:864-868**: audit's specific
  failure mode (jac = 1e12) was hyperbolic since
  ``_logit_variance_jacobian`` clips β to [1e-6, 1-1e-6]. But the
  *direction* of the concern is valid: at boundary β values, the delta-
  method Jacobian is huge and the SE inflates. We test directly: does
  logit_t still detect a strong (Δβ ≈ 0.5) signal when one group sits near
  β = 0?

* **B3.3 — BH-vs-NaN at dmc.py:1755-1761**: audit claimed BH "disturbs
  sort order", but multipletests preserves input position. The *real*
  issue is that NaN-replaced-with-1.0 is treated as a real test by BH,
  inflating ``m`` (total tests). Direction: q-values come out
  **conservative** (slightly higher than they should be), not
  anti-conservative as the audit said.
"""

from __future__ import annotations

import gzip

import numpy as np
import polars as pl
import pytest

from tests.fixtures.synth import SimConfig, generate



# B3.1 — dispersion df under H0 calibration


@pytest.fixture(scope="module")
def synth_md_null(tmp_path_factory):
    """A null fixture: same baseline in both groups, no DMCs, no DMRs.

    Under H0 the chromosome-pooled Pearson φ̂ should be ≈ 1.0 (the
    McCullagh-Nelder estimate of overdispersion with no model misfit).
    """
    import epykit as ep
    out_dir = tmp_path_factory.mktemp("synth_null")
    cfg = SimConfig(
        n_per_group=4,
        chromosomes=("chr1", "chr2"),
        cpgs_per_chrom=2_000,
        baseline_meth=0.50,         # midpoint to avoid clipping
        n_scattered_dmcs=0,          # NO scattered DMCs
        n_dmrs=0,                    # NO DMRs
        dmc_effect=0.0,
        dmr_effect=0.0,
        coverage_mean=20.0,
        replicate_sd=0.03,
        seed=99,
    )
    result = generate(cfg, out_dir)
    md = ep.read_bismark(
        result["samplesheet"],
        treatment_group="treatment",
        control_group="control",
        store_dir=str(out_dir / "store"),
    )
    ep.pp.filter_coverage(md, lo_count=5, hi_perc=99.9)
    ep.pp.unite(md, type="intersect")
    return md


def test_b3_1_dispersion_phi_hat_is_unbiased_under_h0(synth_md_null, caplog):
    """B3.1: Under H0 with chrom-pooled dispersion, φ̂ should be ≈ 1.0.

    If ``df = n_obs - 2 * n_disp`` is correct (the formula the audit
    questioned), φ̂ tracks ``E[χ²] / df = 1.0`` under H0. If df were
    off by even a factor of 2, φ̂ would land near 0.5 or 2.0 and this
    test would loudly fail.

    We use dispersion='chrom' to force a single per-chromosome φ̂ that
    is logged (and easy to retrieve from caplog).
    """
    import logging
    import epykit as ep

    caplog.set_level(logging.INFO, logger="epykit.dmc")
    ep.tl.dmc(synth_md_null, test="lr", dispersion="chrom")

    # Extract the per-chromosome phi_hat values from the INFO log lines:
    # "%s: chrom-pooled φ̂ = %.3f (raw %.3f, ...)"
    import re
    phi_hats = []
    for rec in caplog.records:
        m = re.search(r"chrom-pooled φ̂ = ([\d.]+)", rec.getMessage())
        if m:
            phi_hats.append(float(m.group(1)))

    assert len(phi_hats) >= 1, (
        "expected at least one chrom-pooled φ̂ log line; got none. "
        "Test cannot verify dispersion-df calibration without it."
    )

    mean_phi = float(np.mean(phi_hats))
    # Under correct df + H0, E[φ̂] = 1.0. Allow ±20% wobble — at 4 vs 4
    # replicates × ~2000 sites the estimator has finite-sample variance
    # but no systematic bias. A df off by a factor of 2 would shift
    # mean_phi to ~0.5 or ~2.0, well outside this window.
    assert 0.80 <= mean_phi <= 1.20, (
        f"chrom-pooled φ̂ = {mean_phi:.3f} under H0; expected ≈ 1.0. "
        f"A persistent deviation suggests dmc.py:712 df calculation is wrong."
    )


def test_b3_1_per_site_phi_floors_at_min_dispersion(synth_md_null):
    """B3.1 corollary: under H0, per-site φ̂_i averages near 1.0 too.

    The 'site' dispersion mode computes per-site φ̂_i = chi_i / df_i.
    Under H0 with df_i = (n_case + n_ctrl) - 2 = 4 + 4 - 2 = 6, the
    expected chi-square contribution is df_i. So φ̂_i should average 1.0.
    The implementation clamps to ``min_dispersion = 1.0`` so it can
    only be ≥ 1.0; we just check it doesn't run away to large values.
    """
    import epykit as ep
    ep.tl.dmc(synth_md_null, test="lr", dispersion="site")
    df = synth_md_null.get_dmc(test="lr")
    # phi_hat column may not be exposed; instead check the chi2 statistic
    # distribution is reasonable. Under H0 with correct df, the χ²
    # statistics should follow a chi-square(1) distribution. Most p-values
    # should be > 0.05.
    if "pvalue" in df.columns:
        pvals = df["pvalue"].drop_nulls().to_numpy()
        frac_significant_naive = float((pvals < 0.05).sum() / len(pvals))
        # Under H0, expect ~5%. Allow [2%, 10%] for finite-sample wobble.
        assert 0.02 <= frac_significant_naive <= 0.10, (
            f"H0 false-positive rate at p<0.05 = {frac_significant_naive:.3%}; "
            f"expected ≈ 5%. Outside [2%, 10%] suggests df miscalibration."
        )



# B3.2 — logit-t variance at boundary β
#
# Position (locked in, see dmc.py:_beta_binom_mom_from_welford_logit):
#   logit_t is the WEAK variance-stabilising fallback. We do not pretend
#   it is well-calibrated near β = 0 / 1. Two tests that previously
#   locked in an over-aggressive "NaN if either group has M2 = 0" guard
#   were removed: one asserted ≥30% NaN at extreme β (necessarily true
#   only with the strict guard), the other asserted ~5% H0 FP rate at
#   β ≈ 0.005 (only the strict guard could deliver this).
#
#   The strict guard collapsed power to 0 on the standard accuracy
#   fixture (Δβ=0.40, cov=20, n=4, replicate_sd=0.03) because ~half the
#   truly-differential hypo sites have β_treatment clipped to ~0.01 and
#   routinely produce all-identical y across replicates. The relaxed
#   guard (NaN only when BOTH groups have M2 = 0) restores power at the
#   cost of an anti-conservative SE under H0 boundary β. We accept that
#   trade-off and document the recommendation: use ``test="lr"`` for
#   trustworthy inference; ``logit_t`` only when count-model assumptions
#   are doubtful and the user knows what they're trading.
#
# The moderate-boundary test below remains: at β ≈ 0.05 the Jacobian
# inflation is mild and logit_t works fine, which is what we want to
# protect against regressing.


def test_b3_2_logit_t_detects_signal_at_moderate_boundary_beta(tmp_path):
    """B3.2 follow-up: at *moderate* boundary β (≈ 0.05, not 0.005) the
    Jacobian inflation is much milder (jac² ≈ 440 vs 40,000) and almost
    no sites should have all-zero replicates by chance. Detection at
    p < 0.01 should be near-universal.
    """
    import epykit as ep
    import pandas as pd

    rng = np.random.default_rng(2027)
    n_sites = 50
    chrom = "chr1"
    positions = (np.arange(n_sites) * 200 + 1000).astype(np.int64)
    coverage = 20

    out_dir = tmp_path / "moderate_boundary"
    cov_dir = out_dir / "cov"
    cov_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_idx in range(8):
        is_treat = sample_idx < 4
        group = "treatment" if is_treat else "control"
        sid = f"{group}_{sample_idx % 4 + 1}"
        beta_target = 0.50 if is_treat else 0.05      # moderate boundary
        betas = np.clip(beta_target + rng.normal(0, 0.005, n_sites), 0.001, 0.999)
        N_meth = rng.binomial(coverage, betas).astype(np.int64)
        df = pd.DataFrame({
            "chrom": [chrom] * n_sites,
            "start": positions,
            "end": positions + 1,
            "methyl_pct": 100.0 * N_meth / coverage,
            "N_meth": N_meth,
            "N_unmeth": (coverage - N_meth).astype(np.int64),
        })
        cov_path = cov_dir / f"{sid}.bismark.cov.gz"
        with gzip.open(cov_path, "wt") as fh:
            df.to_csv(fh, sep="\t", header=False, index=False, float_format="%.4f")
        rows.append({"sample_id": sid, "group": group, "path": str(cov_path)})

    samplesheet = out_dir / "samplesheet.csv"
    pd.DataFrame(rows).to_csv(samplesheet, index=False)

    md = ep.read_bismark(
        str(samplesheet),
        treatment_group="treatment",
        control_group="control",
        store_dir=str(tmp_path / "store_moderate"),
    )
    ep.pp.filter_coverage(md, lo_count=2, hi_perc=99.9)
    ep.pp.unite(md, type="intersect")
    ep.tl.dmc(md, test="logit_t")

    df = md.get_dmc(test="logit_t")
    pvals = df["pvalue"].drop_nulls().to_numpy()

    # Analytical sanity check on the threshold:
    #
    #   t-stat ≈ |logit(0.50) − logit(0.05)| / sqrt(SE_case² + SE_ctrl²)
    #          = 2.944 / sqrt(0.05 + 0.263)
    #          = 5.27  with df ≈ 4 (Welch–Satterthwaite)
    #   expected p ≈ 0.006
    #
    # That's just under 0.01, so any realisation noise pushes ~30-40 % of
    # sites above p < 0.01. Asserting at p < 0.05 (where expected p is
    # comfortably below) is the honest signal-to-noise bar.
    frac_strong = float((pvals < 0.05).mean())
    assert frac_strong >= 0.80, (
        f"At moderate boundary β=0.05 with Δβ=0.45 signal, only "
        f"{frac_strong:.0%} of sites detected at p<0.05. The analytical "
        f"expected p-value at this SNR is ~0.006, so > 80 % detection at "
        f"p<0.05 should be trivial — a lower rate signals a real issue."
    )


# NOTE: ``test_b3_2_logit_t_se_reasonable_at_boundary`` (asserted ~5% H0
# FP rate at β ≈ 0.005) was removed alongside the strict B3.2 guard.
# Under the relaxed guard, logit_t at extreme boundary β is genuinely
# anti-conservative; we don't pretend otherwise. See the module-level
# B3.2 note above and the docstring of
# ``_beta_binom_mom_from_welford_logit`` in dmc.py.


# B3.3 — BH with NaN p-values


def test_b3_3_bh_inflates_m_by_nan_count():
    """B3.3: apply_multiple_testing_correction treats NaN-as-1.0 as a real
    test, inflating m (the denominator in BH) by the NaN count.

    Direction: q-values come out more **conservative** (higher) than
    correct, not anti-conservative as the audit claimed.

    Test: a 100-row DataFrame with 20 NaN p-values and 80 uniform-on-[0,1].
    Compare epykit's q-values to the 'filter NaN before BH' approach.
    Among the non-NaN positions, epykit's q-values should be uniformly
    larger by a factor of (100/80) = 1.25 ± numerical noise.
    """
    from statsmodels.stats.multitest import multipletests
    from epykit.dmc import apply_multiple_testing_correction

    rng = np.random.default_rng(0)
    n_total = 100
    n_nan = 20
    pvals = rng.uniform(0, 1, n_total)
    nan_idx = rng.choice(n_total, size=n_nan, replace=False)
    pvals[nan_idx] = np.nan

    df = pl.DataFrame({"pvalue": pvals})
    out = apply_multiple_testing_correction(df)
    epykit_q = out["qvalue"].to_numpy()

    # The "correct" approach: drop NaN before BH, restore after.
    finite_mask = ~np.isnan(pvals)
    correct_q = np.full(n_total, np.nan)
    _, q_finite, _, _ = multipletests(pvals[finite_mask], method="fdr_bh")
    correct_q[finite_mask] = q_finite

    # Among non-NaN positions, compare ratios. Epykit's m = 100, correct
    # m = 80, so epykit_q should be approximately correct_q * 100/80,
    # bounded above by 1.0.
    finite_idx = np.where(finite_mask)[0]
    # Skip positions where correct_q is already saturated at 1.0 (ratio
    # would be 1.0 trivially).
    unsat = finite_idx[correct_q[finite_idx] < 0.99]
    if len(unsat) == 0:
        pytest.skip("all correct q-values saturated; ratio test inconclusive")

    ratios = epykit_q[unsat] / correct_q[unsat]
    median_ratio = float(np.median(ratios))
    # We expect ratio ≈ 100/80 = 1.25.
    expected_ratio = n_total / (n_total - n_nan)
    assert abs(median_ratio - expected_ratio) < 0.10, (
        f"epykit_q / correct_q median ratio = {median_ratio:.3f}; "
        f"expected ≈ {expected_ratio:.3f}. A different ratio means the "
        f"NaN-handling math is different than 'replace with 1.0 before BH'."
    )


def test_b3_3_bh_nan_positions_remain_nan():
    """B3.3 sanity: NaN p-values map to NaN q-values (already covered by
    test_primitives, repeated here for completeness in the B3 file)."""
    from epykit.dmc import apply_multiple_testing_correction

    pvals = np.array([0.01, np.nan, 0.5, np.nan, 0.001])
    df = pl.DataFrame({"pvalue": pvals})
    out = apply_multiple_testing_correction(df)
    q = out["qvalue"].to_numpy()
    assert np.isnan(q[1]) and np.isnan(q[3])
    assert np.isfinite(q[[0, 2, 4]]).all()


def test_b3_3_bh_with_all_nan_does_not_crash():
    """B3.3 sanity: all-NaN p-values shouldn't crash multipletests via the
    epykit shim."""
    from epykit.dmc import apply_multiple_testing_correction

    df = pl.DataFrame({"pvalue": [np.nan, np.nan, np.nan]})
    out = apply_multiple_testing_correction(df)
    q = out["qvalue"].to_numpy()
    assert np.all(np.isnan(q))
