from __future__ import annotations

from pathlib import Path

import polars as pl

from . import filter as filter_mod
from .methyldata import MethylData


def _append_store_history(md: MethylData, step: str, path: str, n_sites: int | None) -> None:
    history = md.uns.get("_store_history")
    if not isinstance(history, list):
        history = []
    history.append({"step": step, "path": path, "n_sites": n_sites})
    md.uns["_store_history"] = history


def _count_parquet_rows(store_dir: str) -> int | None:
    try:
        import pyarrow.parquet as pq

        total = 0
        for path in Path(store_dir).rglob("part-*.parquet"):
            total += pq.read_metadata(str(path)).num_rows
        return total
    except Exception:
        return None


def filter_coverage(
    md: MethylData,
    lo_count: int = 10,
    hi_perc: float = 99.9,
    blacklist_bed: str | None = None,
    output_store: str | None = None,
) -> None:
    """Coverage filtering in-place on a MethylData object."""
    quantile = hi_perc / 100.0 if hi_perc > 1 else hi_perc
    if quantile <= 0 or quantile > 1:
        raise ValueError(f"Invalid hi_perc/quantile value: {hi_perc}")

    # Derive output store path: explicit override, or from analysis_root cache, or legacy behavior
    if output_store:
        out = output_store
    elif md._analysis_root:
        out = str(Path(md._analysis_root) / ".cache" / "filtered")
    else:
        out = f"{md.store}_filtered"
    
    filter_mod.filter_sites(
        methylstore_path=md.store,
        output_dir=out,
        min_coverage=lo_count,
        max_coverage_quantile=quantile,
        blacklist_bed=blacklist_bed,
    )
    md.store = out
    md._filtered = True
    md.uns["filter"] = {
        "lo_count": lo_count,
        "hi_perc": hi_perc,
        "blacklist_bed": blacklist_bed,
    }
    n_sites = _count_parquet_rows(out)
    if n_sites is not None:
        md.uns["n_sites_filtered"] = n_sites
        _append_store_history(md, "filtered", out, n_sites)


def unite(md: MethylData, type: str = "union") -> None:
    """Record the site-alignment strategy for downstream DMC processing.

    This does **not** materialise the full intersection/union into memory.
    ``ep.tl.dmc`` passes ``unite=True/False`` directly to
    ``process_chromosomes_dmc``, which performs the per-chromosome join
    lazily and in O(n_sites) memory — identical to the old procedural API.
    Eagerly computing the full intersection here (previously stored in
    ``md.uns["site_intersect"]``) caused an OOM on whole-genome data because
    it loaded all 338 M+ rows into RAM at once.
    """
    if type not in {"intersect", "union"}:
        raise ValueError("type must be 'intersect' or 'union'")

    md._united = True
    md.uns["unite"] = {"type": type}

def smooth(
    md: MethylData,
    bandwidth: int = 1000,
    grid_resolution_bp: int | None = None,
) -> None:
    """BSmooth-style smoothing.

    .. note::
        Not yet wired into downstream DMC/DMR calling.  The smoothed beta
        values written by this function are not currently read by
        ``ep.tl.dmc`` or ``ep.tl.dmr``, so calling this step has no effect
        on results.  Raise an error rather than silently misleading users.

        To implement properly, ``ep.tl.dmr`` (or a dedicated
        ``ep.tl.dmr_bsmooth``) must load ``md.uns["smooth_path"]`` and
        compute ``meth_diff`` from smoothed betas instead of raw counts.
        Remove this guard once that wiring is in place.

    Raises
    ------
    NotImplementedError
        Always, until downstream usage is implemented.
    ValueError
        If called before ``ep.pp.filter_coverage`` (FIX-10).
    """
    # FIX-10: smoothing on unfiltered data silently degrades results.
    if not md._filtered:
        raise ValueError(
            "Run ep.pp.filter_coverage(md) before ep.pp.smooth(md)."
        )

    # FIX-4: smoothed values are computed but never consumed downstream.
    # Raise so users are not misled into thinking smoothing affects results.
    raise NotImplementedError(
        "ep.pp.smooth() computes smoothed beta values but ep.tl.dmc / "
        "ep.tl.dmr do not yet read them, so this step currently has no "
        "effect on DMC or DMR results.  Remove this call from your pipeline "
        "until BSmooth-style downstream usage is implemented."
    )