"""High-level orchestrators for the standard WGBS analysis flow.

Public entry points: ``qc``, ``dmc``, ``dmr``, ``annotate``. Each mutates a
``MethylData`` in place — results land in ``md.obs`` / ``md.varm`` /
``md.uns``. See the module docstrings of ``dmc.py`` and ``dmr.py`` for the
underlying engines.
"""

from __future__ import annotations

import gc
import polars as pl

from .annotate import annotate_cpg_islands, annotate_features, _GTF_CACHE
from .dmc import apply_multiple_testing_correction, process_chromosomes_dmc
from .dmr import call_dmr_sliding_window, call_dmr_tile_based, empirical_fdr_for_dmr
from .methyldata import MethylData
from .qc import bisulfite_conversion_rate, coverage_uniformity, global_methylation_report


def _auto_test(
    md: MethylData,
    design: str | None = None,
    covariates: list[str] | None = None,
    allow_n1: bool = False,
) -> str:
    """Pick a sensible test based on group size, accounting for covariates.

    When a covariate design is supplied, we MUST use the binomial GLM path
    (``"glm"``) because the closed-form ``lr`` / ``score`` paths don't admit
    covariates. The choice is therefore unconditional whenever the user
    asks for adjustment.

    ``allow_n1`` is forwarded to :func:`_auto_test_simple` and only takes
    effect when there are fewer than 2 replicates per group.
    """
    if design is not None or (covariates is not None and len(covariates) > 0):
        return "glm"
    return _auto_test_simple(md, allow_n1=allow_n1)


# One-shot warning gate: ``tl.dmc`` emits a UserWarning the first time a
# user explicitly selects ``test="fisher"`` in a session. We don't want to
# spam them across thousands of chromosomes/sites — once is enough.
_FISHER_WARNED = False


def _warn_fisher_once() -> None:
    """Emit a one-shot UserWarning when the user explicitly picks fisher."""
    global _FISHER_WARNED
    if _FISHER_WARNED:
        return
    _FISHER_WARNED = True
    import warnings
    warnings.warn(
        "test='fisher' ignores between-replicate variance; p-values are "
        "anti-conservative. Prefer test='lr' at n >= 2.",
        UserWarning, stacklevel=3,
    )


def _check_n1_and_union_footgun(
    md: MethylData,
    allow_n1: bool,
    min_samples_treatment: int,
    min_samples_control: int,
    unit: str = "sites",
) -> None:
    """Enforce n>=2 per group (unless allow_n1) and warn on union+0/0 (B6/B8)."""
    if min(len(md.treatment_ids), len(md.control_ids)) < 2 and not allow_n1:
        _auto_test_simple(md, allow_n1=False)  # raises ValueError
    unite_info = md.uns.get("unite")
    if (
        unite_info is not None
        and unite_info.get("type") == "union"
        and min_samples_treatment == 0
        and min_samples_control == 0
    ):
        import warnings
        warnings.warn(
            f"unite='union' with min_samples_treatment=min_samples_control=0 "
            f"will test {unit} covered in only one sample per group. "
            f"Recommended: both >= 2 (or unite='intersect').",
            UserWarning, stacklevel=3,
        )


def _resolve_min_samples_aliases(
    min_samples_treatment: int | None,
    min_samples_case: int | None,
    default: int = 0,
) -> int:
    """Accept the deprecated ``min_samples_case`` kwarg with a DeprecationWarning (S9).

    Returns the resolved canonical value. Either both None (use default), one
    set, or — illegally — both set (TypeError).
    """
    import warnings
    if min_samples_case is not None:
        warnings.warn(
            "min_samples_case is deprecated; use min_samples_treatment.",
            DeprecationWarning, stacklevel=3,
        )
        if min_samples_treatment is not None:
            raise TypeError(
                "Pass either min_samples_treatment or min_samples_case, not both"
            )
        return int(min_samples_case)
    if min_samples_treatment is None:
        return default
    return int(min_samples_treatment)


def _auto_test_simple(md: MethylData, allow_n1: bool = False) -> str:
    """Pick a sensible test based on group size.

    Current default at n>=2: ``"lr"`` — the quasi-binomial likelihood-ratio
    chi-square with per-site McCullagh-Nelder dispersion. This is what
    methylKit's ``calculateDiffMeth(overdispersion="MN", test="Chisq")``
    reports, computed in closed form on the same streaming
    (S0_g, S1_g, Σm²/n_g) accumulators we already keep for the score test.
    LR is closer to nominal type-I error than the score test at the small
    samples (n=6) and boundary proportions typical in WGBS.

    The default returned here MUST match the CLI ``--test`` default (lr) and
    the ``--test`` default for ``dmr`` (lr). See cli.py for the single source
    of truth.

    The score test (``test="score"``) is still available for users who want
    a marginally more powerful (but mildly anti-conservative at the
    boundaries) statistic on the same accumulators.

    At n=1 (single replicate per group) there is no between-replicate
    variability for φ̂ to estimate. By default this is treated as a hard error
    (statistical inference is not credible). Pass ``allow_n1=True`` to opt
    into the Fisher exact fallback (anti-conservative; warns at runtime).
    """
    min_group = min(len(md.treatment_ids), len(md.control_ids))
    if min_group < 2:
        if not allow_n1:
            raise ValueError(
                f"At least 2 replicates per group are required for valid "
                f"statistical inference (got treatment={len(md.treatment_ids)}, "
                f"control={len(md.control_ids)}). To proceed anyway with "
                f"Fisher exact on pooled reads (no between-replicate variance), "
                f"pass allow_n1=True to ep.tl.dmc(). Be aware p-values from "
                f"this path are anti-conservative and should not be reported "
                f"as evidence of differential methylation."
            )
        import warnings
        warnings.warn(
            "n<2 per group: falling back to Fisher exact on pooled reads. "
            "Between-replicate variance is ignored and p-values are anti-conservative.",
            UserWarning,
            stacklevel=3,
        )
        return "fisher"
    return "lr"


def qc(
    md: MethylData,
    chh_context_store: str | None = None,
    *,
    run_sex_check: bool = False,
    run_contamination: bool = False,
    run_sample_correlation: bool = False,
    correlation_method: str = "spearman",
    expected_sex_col: str | None = None,
) -> None:
    """Populate md.obs with per-sample QC metrics and cache QC tables in md.uns.

    Plan 2 §5 additions are opt-in via the ``run_*`` flags so the default
    ``tl.qc(md)`` keeps the existing fast subset.
    """
    samples = md.obs.get_column("sample_id").to_list()

    global_report = global_methylation_report(md.store, samples)
    cov_reports = [coverage_uniformity(md.store, sample) for sample in samples]
    cov_report = pl.concat(cov_reports) if cov_reports else pl.DataFrame()

    obs = md.obs
    cpG_report = global_report.filter(pl.col("context") == "CpG")
    if len(cpG_report) > 0:
        obs = obs.join(
            cpG_report.select([pl.col("sample").alias("sample_id"), "global_methylation"]),
            on="sample_id",
            how="left",
        )

    if len(cov_report) > 0:
        cov_genome = cov_report.filter(pl.col("chrom") == "genome")
        obs = obs.join(
            cov_genome.select([
                "sample",
                "mean_coverage",
                "frac_ge_1x",
                "frac_ge_5x",
                "frac_ge_10x",
                "low_coverage_flag",
            ]).rename({"sample": "sample_id"}),
            on="sample_id",
            how="left",
        )

    if chh_context_store:
        conv = []
        for sample in samples:
            try:
                rate = bisulfite_conversion_rate(md.store, sample, chh_context_store)
            except Exception:
                rate = None
            conv.append({"sample_id": sample, "bisulfite_conversion_rate": rate})
        obs = obs.join(pl.DataFrame(conv), on="sample_id", how="left")

    # --- Plan 2 §5: clinical QC additions ----------------------------------
    if run_sex_check:
        from .qc import sex_check as _sex_check
        expected = None
        if expected_sex_col and expected_sex_col in obs.columns:
            expected = {
                row["sample_id"]: row[expected_sex_col]
                for row in obs.iter_rows(named=True)
                if row.get(expected_sex_col) is not None
            }
        sex_df = _sex_check(md.store, samples, expected_sex=expected)
        md.uns["qc_sex_check"] = sex_df
        obs = obs.join(
            sex_df.select(["sample_id", "inferred_sex", "mismatch"]).rename(
                {"mismatch": "sex_mismatch"}
            ),
            on="sample_id", how="left",
        )

    if run_contamination:
        from .qc import contamination_estimate as _contam
        scores = [
            {"sample_id": s, "contamination_score": float(_contam(md.store, s))}
            for s in samples
        ]
        obs = obs.join(pl.DataFrame(scores), on="sample_id", how="left")

    if run_sample_correlation:
        from .qc import sample_correlation as _samp_corr
        corr_df = _samp_corr(md.store, samples, method=correlation_method)
        md.uns["qc_sample_correlation"] = corr_df
        if len(corr_df) > 0:
            # Per-sample min off-diagonal correlation (low → likely swap).
            off_diag = corr_df.filter(pl.col("sample_a") != pl.col("sample_b"))
            min_corr = (
                off_diag.group_by("sample_a")
                .agg(pl.min("correlation").alias("min_pairwise_corr"))
                .rename({"sample_a": "sample_id"})
            )
            obs = obs.join(min_corr, on="sample_id", how="left")

    md.obs = obs
    md.uns["qc_global_methylation"] = global_report
    md.uns["qc_coverage_uniformity"] = cov_report


def dmc(
    md: MethylData,
    test: str = "auto",
    chromosomes: list[str] | None = None,
    min_samples_treatment: int | None = None,
    min_samples_control: int = 0,
    dispersion: str = "site",
    reference: str = "methylkit",
    allow_n1: bool = False,
    # Section 1 of Plan 2: multi-group / continuous-covariate contrasts
    formula: str | None = None,
    contrast=None,
    covariates: list[str] | None = None,
    treatment_col: str = "treatment",
    *,
    min_samples_case: int | None = None,  # deprecated alias (S9)
) -> None:
    """Run DMC calling and store result in md.varm['dmc_<test>'].

    Parameters
    ----------
    md : MethylData
        Analysis object containing the methylstore path and the
        treatment/control sample lists.
    test : str
        One of ``"auto"``, ``"lr"``, ``"score"``, ``"logit_t"``,
        ``"welch_t"`` (formerly ``"beta_binomial"`` — deprecated alias),
        ``"bb_lr"`` (true quasi-binomial LRT), ``"cmh"``, ``"fisher"``,
        ``"glm"``. ``"auto"`` resolves to ``"fisher"`` at n<2 and ``"lr"``
        (methylKit parity) at n>=2.

        When ``formula`` and/or ``contrast`` are supplied, the test is
        forced to a GLM-based path regardless of ``test=``.
    formula : str, optional
        patsy formula on ``md.obs`` columns, e.g. ``"~ group"`` for a
        multi-group test or ``"~ age + sex"`` for a continuous-covariate
        primary effect. When supplied with ``contrast``, the engine fits
        the GLM once per site and runs a Wald / joint-F test against the
        contrast.
    contrast : str or np.ndarray, optional
        Contrast specification. Accepts:
        - a column name in the resolved design (``"age"`` for a continuous
          covariate primary effect; produces a single-coef Wald-z² test
          with meth-scale CIs);
        - a factor name (``"group"``); every dummy of that factor is
          included → joint F-test (multi-group);
        - a patsy linear-combination string
          (``"group[T.KO] - group[T.WT]"``); produces a single-row contrast;
        - a raw ``(k, p)`` matrix.
    covariates : list[str], optional
        Convenience list of column names to include as nuisance terms.
        Combined with ``formula`` and the resolved ``treatment_col``.
    treatment_col : str, default ``"treatment"``
        Name of the binary 0/1 column in ``md.obs`` used by the legacy
        binary path. Ignored when ``contrast`` is supplied and resolves
        without it.
    dispersion : {"site", "chrom", "shrink"}
        McCullagh-Nelder dispersion strategy used by the ``"lr"`` and
        ``"score"`` tests. Default ``"site"`` matches methylKit
        ``overdispersion="MN"``.
    chromosomes : list[str], optional
        Restrict to a subset of chromosomes. Auto-detected when None.
    min_samples_treatment, min_samples_control : int
        BIO-7: per-site minimum number of samples with non-zero coverage in
        each group. Sites that fail are NaN'd out before FDR correction.
        Primarily useful when ``ep.pp.unite(..., type="union")`` was used so
        that union-introduced zero-coverage rows aren't treated as real
        observations.

        ``min_samples_case`` is accepted as a deprecated alias for
        ``min_samples_treatment`` (S9 naming unification).
    """
    min_samples_treatment = _resolve_min_samples_aliases(
        min_samples_treatment, min_samples_case, default=0,
    )

    # --- New contrast / multi-group path -------------------------------------
    if formula is not None or contrast is not None:
        _run_dmc_contrast(
            md, test=test, formula=formula, contrast=contrast,
            covariates=covariates, treatment_col=treatment_col,
            chromosomes=chromosomes,
            min_samples_treatment=min_samples_treatment,
            min_samples_control=min_samples_control,
            dispersion=dispersion, reference=reference,
        )
        return

    # Unconditional n=1 guard: applies whether test is "auto" or explicit.
    # _auto_test_simple raises ValueError when allow_n1=False; trigger that
    # check up front so explicit test="lr"/"fisher" with n<2 also gets
    # caught instead of silently running on degenerate data.
    _check_n1_and_union_footgun(
        md, allow_n1, min_samples_treatment, min_samples_control,
    )
    selected_test = _auto_test(md, allow_n1=allow_n1) if test == "auto" else test
    if selected_test == "fisher":
        _warn_fisher_once()
    unite_info = md.uns.get("unite")
    unite = (unite_info is not None) and (unite_info.get("type") == "intersect")

    result = process_chromosomes_dmc(
        methylstore_path=md.store,
        samples_treatment=md.treatment_ids,
        samples_control=md.control_ids,
        test=selected_test,
        chromosomes=chromosomes,
        unite=unite,
        min_samples_treatment=min_samples_treatment,
        min_samples_control=min_samples_control,
        dispersion=dispersion,
        reference=reference,
    )
    result = apply_multiple_testing_correction(result, method="fdr_bh")

    # Canonicalise key name (test_used reflects the canonical name post-rename)
    from .dmc import _canonicalise_test_name
    canonical_used = _canonicalise_test_name(selected_test)
    key = f"dmc_{canonical_used}"
    md.varm[key] = result
    md.uns["dmc"] = {
        "test_requested": test,
        "test_used": canonical_used,
        "n_sites": len(result),
        "unite": unite,
        "min_samples_treatment": min_samples_treatment,
        "min_samples_control": min_samples_control,
        # Back-compat alias: legacy readers that look up "min_samples_case"
        # still find the value; new code should read "min_samples_treatment".
        "min_samples_case": min_samples_treatment,
        "dispersion": dispersion,
        "reference": reference,
        # S5: explicit pointer so MethylData.get_dmc() / .dmc resolve to the
        # table the user just wrote, regardless of which other tests have
        # been run in the same session.
        "last_key": key,
    }


def _run_dmc_contrast(
    md: MethylData,
    *,
    test: str,
    formula: str | None,
    contrast,
    covariates: list[str] | None,
    treatment_col: str,
    chromosomes: list[str] | None,
    min_samples_treatment: int,
    min_samples_control: int,
    dispersion: str,
    reference: str,
) -> None:
    """Internal: multi-group / continuous-covariate primary-effect DMC.

    Always uses test='glm_contrast' internally. Uses ALL samples in
    md.obs order (not the binary case/control split), so the design
    matrix matches md.obs row-for-row.
    """
    import numpy as np
    from .dmc import process_chromosomes_dmc
    from ._glm import build_design, resolve_contrast

    if not md.obs.height:
        raise ValueError("md.obs is empty; cannot build a design matrix.")
    samples_all = md.obs.get_column("sample_id").to_list()

    # Build design — without requiring a treatment column if we have a
    # formula that doesn't reference one. The user's `treatment_col`
    # default ("treatment") is *only* required when the existing binary
    # path would have used it; here we let the formula speak.
    need_treatment = (treatment_col in md.obs.columns) and (
        formula is None or treatment_col in formula
    )
    design_full, _design_reduced, coef_idx, term_names, formula_used, design_info = (
        build_design(
            md.obs,
            samples_ordered=samples_all,
            formula=formula,
            covariates=covariates,
            treatment_col=treatment_col,
            require_treatment_col=need_treatment,
            return_design_info=True,
        )
    )

    # Resolve the contrast against the design.
    if contrast is None:
        # Default: a single-coef contrast on `treatment_col` (this happens
        # when the user supplies a `formula=` for covariate adjustment but
        # no explicit contrast).
        contrast = treatment_col
    C, contrast_label = resolve_contrast(contrast, term_names, design_info=design_info)

    # Build per-sample level labels for the multi-group output schema:
    # take the FIRST term that is a factor of `contrast` (when contrast is
    # a factor name), otherwise no per-level breakdown.
    group_labels: list[str] | None = None
    if isinstance(contrast, str) and contrast in md.obs.columns:
        # Either a continuous column (single coef) or a categorical column
        # (joint test). For both, emit per-level labels for downstream
        # mean_beta_<level> columns when the column is categorical.
        col = md.obs.get_column(contrast)
        if col.dtype == pl.Utf8 or col.dtype == pl.Categorical:
            group_labels = col.cast(pl.Utf8).to_list()

    # Determine which samples are "case" vs "control" for the
    # backwards-compatible binary columns. If treatment_col is on obs and
    # carries a numeric 0/1 signal, use it; otherwise leave both empty so
    # mean_beta_case/control remain NaN (uninterpretable for multi-group).
    samples_case_local: list[str] = []
    samples_control_local: list[str] = []
    if treatment_col in md.obs.columns:
        try:
            mask_treat = (
                md.obs.get_column(treatment_col).cast(pl.Float64, strict=False) == 1
            ).to_list()
            samples_case_local = [s for s, m in zip(samples_all, mask_treat) if m]
            samples_control_local = [s for s, m in zip(samples_all, mask_treat) if not m]
        except Exception:
            pass

    unite_info = md.uns.get("unite")
    unite = (unite_info is not None) and (unite_info.get("type") == "intersect")

    result = process_chromosomes_dmc(
        methylstore_path=md.store,
        samples_treatment=samples_case_local,
        samples_control=samples_control_local,
        test="glm_contrast",
        chromosomes=chromosomes,
        unite=unite,
        min_samples_treatment=min_samples_treatment,
        min_samples_control=min_samples_control,
        dispersion=dispersion,
        reference=reference,
        design_full=design_full,
        contrast_matrix=C,
        contrast_label=contrast_label,
        samples_all_ordered=samples_all,
        group_labels_per_sample=group_labels,
    )
    result = apply_multiple_testing_correction(result, method="fdr_bh")

    key = "dmc_glm_contrast"
    md.varm[key] = result
    md.uns["dmc"] = {
        "test_requested": test,
        "test_used": "glm_contrast",
        "n_sites": len(result),
        "unite": unite,
        "formula": formula_used,
        "contrast": contrast_label,
        "design_terms": term_names,
        "covariates": list(covariates) if covariates else None,
        "treatment_col": treatment_col,
        "min_samples_treatment": min_samples_treatment,
        "min_samples_control": min_samples_control,
        "min_samples_case": min_samples_treatment,
        "dispersion": dispersion,
        "reference": reference,
        "last_key": key,
    }


def dmr(
    md: MethylData,
    method: str = "tile",
    # Tile-method options ---------------------------------------------------
    tile_size_bp: int = 1000,
    min_cpgs_per_tile: int = 5,
    test: str = "auto",
    chromosomes: list[str] | None = None,
    min_samples_treatment: int | None = None,
    min_samples_control: int = 0,
    dispersion: str = "site",
    reference: str = "methylkit",
    # Covariate design (tile-method only) ----------------------------------
    design: str | None = None,
    covariates: list[str] | None = None,
    treatment_col: str = "treatment",
    # Sliding-window options ------------------------------------------------
    window_bp: int = 500,
    step_bp: int = 250,
    min_cpgs: int = 5,
    min_sites_significant: int = 3,
    # Shared filters --------------------------------------------------------
    alpha: float = 0.05,
    min_abs_meth_diff: float = 0.1,
    min_mean_qvalue: float | None = 0.05,
    # Replicate-count guard --------------------------------------------------
    allow_n1: bool = False,
    # Plan 2 §3: permutation-based empirical FDR (tile method only) --------
    empirical_fdr: bool = False,
    n_perm: int = 100,
    perm_seed: int = 42,
    perm_n_jobs: int = 1,
    *,
    min_samples_case: int | None = None,  # deprecated alias (S9)
) -> None:
    """Run DMR calling and store result in ``md.uns['dmr']``.

    Two methods are supported:

    * ``method="tile"`` (default, methylKit parity, BIO-5) — aggregates
      read counts within fixed tiles and runs a single test per tile.
      Requires direct access to ``md.store`` and the per-sample methylstore;
      does not need a prior DMC table. This is the recommended path for
      whole-genome WGBS analyses.
    * ``method="sliding_window"`` — the legacy in-tree method: takes the
      DMC result on ``md`` and combines per-CpG p-values within overlapping
      windows with signed Stouffer's Z. Faster (no extra I/O) but
      substantially lower-power than tile-based aggregation at typical
      WGBS coverage.

    Parameters
    ----------
    method : {"tile", "sliding_window"}
        Which DMR algorithm to run.
    tile_size_bp, min_cpgs_per_tile : int
        Tile-method options. ``tile_size_bp=1000`` matches methylKit's default.
    test : str
        Statistical test for tile-method (ignored when ``method="sliding_window"``).
        ``"auto"`` resolves the same way as in :func:`dmc`.
    chromosomes : list[str], optional
        Restrict tile-method processing to these chromosomes.
    min_samples_treatment, min_samples_control : int
        Per-tile sample-count guard for tile-method (BIO-7).
        ``min_samples_case`` is accepted as a deprecated alias for
        ``min_samples_treatment`` (S9 naming unification).
    window_bp, step_bp, min_cpgs, min_sites_significant : int
        Sliding-window method options.
    alpha : float
        q-value threshold for "significant" at the DMC / tile level
        (used by both methods, with different downstream meanings).
    min_abs_meth_diff : float
        Minimum |meth_diff| for a DMC / tile to count.
    min_mean_qvalue : float or None
        BIO-10: post-hoc filter on the DMR-level **q-value**
        (``combined_qvalue`` for sliding-window, ``qvalue`` for tile).
        DMRs with q >= ``min_mean_qvalue`` are dropped. Set to None to keep
        all candidate DMRs. Default 0.05.

        This parameter was previously named ``min_mean_pvalue`` and applied
        to the uncorrected p-value, which was not FDR-controlled across the
        DMR set.
    """
    min_samples_treatment = _resolve_min_samples_aliases(
        min_samples_treatment, min_samples_case, default=0,
    )
    if method == "tile":
        _check_n1_and_union_footgun(
            md, allow_n1, min_samples_treatment, min_samples_control, unit="tiles",
        )
        selected_test = (
            _auto_test(md, design=design, covariates=covariates, allow_n1=allow_n1)
            if test == "auto"
            else test
        )
        if selected_test == "fisher":
            _warn_fisher_once()
        unite_info = md.uns.get("unite")
        unite = (unite_info is not None) and (unite_info.get("type") == "intersect")

        # ---- Build covariate design matrix when requested -----------------
        design_full = None
        design_reduced = None
        coef_idx = None
        term_names: list[str] = []
        formula_used: str | None = None
        if selected_test == "glm" or design is not None or covariates is not None:
            from ._glm import build_design
            samples_ordered = md.treatment_ids + md.control_ids
            design_full, design_reduced, coef_idx, term_names, formula_used = build_design(
                md.obs,
                samples_ordered=samples_ordered,
                formula=design,
                covariates=covariates,
                treatment_col=treatment_col,
            )
            # Force GLM regardless of what 'auto' resolved to: covariates only
            # work with the GLM path.
            selected_test = "glm"

        dmr_df = call_dmr_tile_based(
            methylstore_path=md.store,
            samples_treatment=md.treatment_ids,
            samples_control=md.control_ids,
            tile_size_bp=tile_size_bp,
            test=selected_test,
            chromosomes=chromosomes,
            min_cpgs_per_tile=min_cpgs_per_tile,
            alpha=alpha,
            min_abs_meth_diff=min_abs_meth_diff,
            unite=unite,
            min_samples_treatment=min_samples_treatment,
            min_samples_control=min_samples_control,
            dispersion=dispersion,
            reference=reference,
            design_full=design_full,
            design_reduced=design_reduced,
            coef_idx=coef_idx,
        )

        # Optional q-value post-filter (the tile path already filtered at
        # `alpha`, but a stricter user threshold is allowed here).
        if len(dmr_df) > 0 and min_mean_qvalue is not None and "qvalue" in dmr_df.columns:
            dmr_df = dmr_df.filter(pl.col("qvalue") < min_mean_qvalue)

        # Plan 2 §3: permutation FDR. Refuses to run when a covariate design
        # is in play (shuffling treatment labels invalidates the assumed
        # covariate structure).
        if empirical_fdr:
            if design is not None or (covariates is not None and len(covariates) > 0):
                raise ValueError(
                    "empirical_fdr=True is not supported with covariate "
                    "designs (label-shuffling invalidates stratification). "
                    "Use a stratified-permutation scheme manually if needed."
                )
            if len(dmr_df) > 0:
                dmr_df = empirical_fdr_for_dmr(
                    methylstore_path=md.store,
                    samples_treatment=md.treatment_ids,
                    samples_control=md.control_ids,
                    observed_dmr=dmr_df,
                    n_perm=n_perm,
                    seed=perm_seed,
                    n_jobs=perm_n_jobs,
                    tile_size_bp=tile_size_bp,
                    test=selected_test,
                    chromosomes=chromosomes,
                    min_cpgs_per_tile=min_cpgs_per_tile,
                    alpha=alpha,
                    min_abs_meth_diff=min_abs_meth_diff,
                    unite=unite,
                    min_samples_treatment=min_samples_treatment,
                    min_samples_control=min_samples_control,
                    dispersion=dispersion,
                    reference=reference,
                )

        md.uns["dmr"] = dmr_df
        md.uns["dmr_params"] = {
            "method": "tile",
            "tile_size_bp": tile_size_bp,
            "min_cpgs_per_tile": min_cpgs_per_tile,
            "test": selected_test,
            "alpha": alpha,
            "min_abs_meth_diff": min_abs_meth_diff,
            "min_mean_qvalue": min_mean_qvalue,
            "min_samples_treatment": min_samples_treatment,
            "min_samples_control": min_samples_control,
            # Back-compat alias for legacy readers.
            "min_samples_case": min_samples_treatment,
            "unite": unite,
            "dispersion": dispersion,
            "reference": reference,
            "design": design,
            "covariates": list(covariates) if covariates else None,
            "treatment_col": treatment_col,
            "formula_used": formula_used,
            "design_terms": term_names if term_names else None,
            "empirical_fdr": empirical_fdr,
            "n_perm": n_perm if empirical_fdr else None,
            "perm_seed": perm_seed if empirical_fdr else None,
        }
        return

    if method == "sliding_window":
        dmc_df = md.dmc
        if dmc_df is None:
            raise ValueError(
                "No DMC results available. Run ep.tl.dmc(md) first, or use "
                "method='tile' which goes directly to the methylstore."
            )

        dmr_df = call_dmr_sliding_window(
            dmc_results=dmc_df,
            window_bp=window_bp,
            step_bp=step_bp,
            min_cpgs=min_cpgs,
            min_sites_significant=min_sites_significant,
            alpha=alpha,
            min_abs_meth_diff=min_abs_meth_diff,
        )
        # BIO-10: filter on the BH-corrected DMR-level q-value, not the raw
        # combined p-value. ``call_dmr_sliding_window`` now adds
        # ``combined_qvalue`` itself.
        if len(dmr_df) > 0 and min_mean_qvalue is not None:
            q_col = "combined_qvalue" if "combined_qvalue" in dmr_df.columns else "combined_pvalue"
            dmr_df = dmr_df.filter(pl.col(q_col) < min_mean_qvalue)

        md.uns["dmr"] = dmr_df
        md.uns["dmr_params"] = {
            "method": "sliding_window",
            "window_bp": window_bp,
            "step_bp": step_bp,
            "min_cpgs": min_cpgs,
            "min_sites_significant": min_sites_significant,
            "alpha": alpha,
            "min_abs_meth_diff": min_abs_meth_diff,
            "min_mean_qvalue": min_mean_qvalue,
        }
        return

    raise ValueError(
        f"Unknown DMR method '{method}'. Expected 'tile' or 'sliding_window'."
    )


def dvc(
    md: MethylData,
    test: str = "bartlett",
    chromosomes: list[str] | None = None,
    alpha: float = 0.05,
    mean_filter_alpha: float = 0.05,
) -> None:
    """Differential-Variability CpG calling (Plan 2 §4, iEVORA-style).

    Identifies CpGs whose between-replicate variance differs significantly
    between the treatment and control groups *while* the means do not —
    the signature of an outlier-driven shift in variability that purely
    mean-based DMC analysis misses (cancer / aging methylomes).

    Result is stored at ``md.varm["dvc"]`` with columns:
        chrom, pos, strand, n_treatment, n_control,
        var_treatment, var_control, var_log_ratio,
        p_variance, q_variance, p_mean, q_mean, is_dvc

    Parameters
    ----------
    test : {"bartlett", "levene", "brown_forsythe"}
        Variance-equality test. ``"bartlett"`` is the default closed-form
        choice under the Welford streaming budget. The other two delegate
        to the same Bartlett path under streaming (see
        :func:`epykit.dvc._levene_per_site` for the design rationale).
    alpha : float
        q-value cutoff on the variance test for the ``is_dvc`` flag.
    mean_filter_alpha : float
        Sites are flagged DVC only when ``p_mean > mean_filter_alpha`` —
        i.e. variance changes that aren't accompanied by mean changes.
    """
    from .dvc import process_chromosomes_dvc
    unite_info = md.uns.get("unite")
    unite = (unite_info is not None) and (unite_info.get("type") == "intersect")
    result = process_chromosomes_dvc(
        methylstore_path=md.store,
        samples_treatment=md.treatment_ids,
        samples_control=md.control_ids,
        test=test,
        chromosomes=chromosomes,
        unite=unite,
        mean_filter_alpha=mean_filter_alpha,
        alpha=alpha,
    )
    md.varm["dvc"] = result
    md.uns["dvc"] = {
        "test": test,
        "alpha": alpha,
        "mean_filter_alpha": mean_filter_alpha,
        "n_sites": len(result),
        "n_dvc": int(result.get_column("is_dvc").sum()) if len(result) else 0,
        "unite": unite,
    }


def annotate(
    md: MethylData,
    gtf: str | None = None,
    cpg_islands: str | None = None,
    significant_only: bool = True,
    alpha: float = 0.05,
    promoter_upstream_bp: int = 2000,
    promoter_downstream_bp: int = 200,
    clear_gtf_cache: bool = True,
) -> None:
    """Annotate DMC/DMR outputs.

    By default only significant DMCs are annotated to avoid OOM. Set
    `significant_only=False` to annotate all sites (not recommended for
    whole-genome datasets).

    Parameters
    ----------
    clear_gtf_cache : bool, optional
        If True (default), clear the GTF cache and run garbage collection
        after annotation. Set to False if you plan to call annotate()
        multiple times to reuse the cached GTF.
    """
    if not gtf and not cpg_islands:
        raise ValueError("Provide at least one of gtf or cpg_islands")

    for key, df in list(md.varm.items()):
        if not key.startswith("dmc"):
            continue

        # Match old behavior: annotate only significant sites to avoid OOM
        if significant_only:
            p_col = "qvalue" if "qvalue" in df.columns else "pvalue"
            ann = df.filter(pl.col(p_col) < alpha)
        else:
            ann = df

        if len(ann) == 0:
            continue

        if gtf:
            ann = annotate_features(
                ann,
                annotation_gtf=gtf,
                promoter_upstream_bp=promoter_upstream_bp,
                promoter_downstream_bp=promoter_downstream_bp,
            )
        if cpg_islands:
            ann = annotate_cpg_islands(ann, cpg_island_bed=cpg_islands)

        # Store as separate key so full DMC results are preserved
        md.varm[f"{key}_annotated"] = ann

    if "dmr" in md.uns and isinstance(md.uns["dmr"], pl.DataFrame) and gtf:
        md.uns["dmr"] = annotate_features(
            md.uns["dmr"],
            annotation_gtf=gtf,
            promoter_upstream_bp=promoter_upstream_bp,
            promoter_downstream_bp=promoter_downstream_bp,
        )

    md.uns["annotation"] = {
        "gtf": gtf,
        "cpg_islands": cpg_islands,
        "significant_only": significant_only,
        "alpha": alpha,
        "promoter_upstream_bp": promoter_upstream_bp,
        "promoter_downstream_bp": promoter_downstream_bp,
    }

    # Clear GTF cache if requested (default: True)
    if clear_gtf_cache and gtf:
        _GTF_CACHE.clear()
        gc.collect()