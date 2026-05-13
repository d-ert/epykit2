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
from .dmr import call_dmr_sliding_window, call_dmr_tile_based
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


def qc(md: MethylData, chh_context_store: str | None = None) -> None:
    """Populate md.obs with per-sample QC metrics and cache QC tables in md.uns."""
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
        ``"beta_binomial"``, ``"cmh"``, ``"fisher"``. ``"auto"`` resolves to
        ``"fisher"`` at n<2 and ``"lr"`` (methylKit parity) at n>=2.
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

    key = f"dmc_{selected_test}"
    md.varm[key] = result
    md.uns["dmc"] = {
        "test_requested": test,
        "test_used": selected_test,
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