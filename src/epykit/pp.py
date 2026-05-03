from __future__ import annotations

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
    try:
        n_sites = None
        n_sites = int(
            pl.scan_parquet(f"{out}/sample=*/chrom=*/part-*.parquet")
            .select(pl.len().alias("n"))
            .collect()["n"][0]
        )
        md.uns["n_sites_filtered"] = n_sites
        _append_store_history(md, "filtered", out, n_sites)
    except Exception:
        pass


def unite(md: MethylData, type: str = "intersect") -> None:
    """Build site-set alignment metadata for downstream DMC processing."""
    samples = md.obs.get_column("sample_id").to_list()
    if type not in {"intersect", "union"}:
        raise ValueError("type must be 'intersect' or 'union'")

    if type == "intersect":
        site_df = filter_mod.intersect_sites(md.store, samples)
        md.uns["site_intersect"] = site_df
    else:
        site_df = (
            pl.scan_parquet(f"{md.store}/sample=*/chrom=*/part-*.parquet")
            .select(["chrom", "pos", "strand"])
            .unique()
            .collect()
            .sort(["chrom", "pos"])
        )
        md.uns["site_union"] = site_df

    md._united = True
    md.uns["unite"] = {"type": type, "n_sites": len(site_df)}


def smooth(
    md: MethylData,
    bandwidth: int = 1000,
    grid_resolution_bp: int | None = None,
) -> None:
    """BSmooth-style smoothing and store results in md.uns['smoothed']."""
    samples = md.obs.get_column("sample_id").to_list()
    smoothed = smooth_methylation_bsmooth(
        methylstore_path=md.store,
        samples=samples,
        bandwidth=bandwidth,
        grid_resolution_bp=grid_resolution_bp,
    )
    md.uns["smoothed"] = smoothed
    md._smoothed = True
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
