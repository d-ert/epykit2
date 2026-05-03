from __future__ import annotations

import gc as _gc
from pathlib import Path

import polars as pl

from . import filter as filter_mod
from .dmr import smooth_methylation_bsmooth
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

    out = output_store or f"{md.store}_filtered"
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


def unite(md: MethylData, type: str = "intersect") -> None:
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
    """BSmooth-style smoothing. Write smoothed output to disk and free RAM."""
    samples = md.obs.get_column("sample_id").to_list()
    smoothed = smooth_methylation_bsmooth(
        methylstore_path=md.store,
        samples=samples,
        bandwidth=bandwidth,
        grid_resolution_bp=grid_resolution_bp,
    )

    # Write to disk immediately and free RAM — matches old scratch.py pattern
    smooth_path = f"{md.store}_smooth.parquet"
    smoothed.write_parquet(smooth_path, compression="zstd")
    del smoothed
    _gc.collect()

    md._smoothed = True
    md.uns["smooth_path"] = smooth_path
    md.uns["smooth"] = {
        "bandwidth": bandwidth,
        "grid_resolution_bp": grid_resolution_bp,
    }
    _append_store_history(
        md,
        "smoothed",
        md.store,
        md.uns.get("n_sites_filtered") or md.uns.get("n_sites_raw"),
    )
