from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from . import filter as filter_mod
from .methyldata import MethylData

logger = logging.getLogger(__name__)


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


def normalize_coverage(md: MethylData, method: str = "median") -> None:
    """methylKit-parity coverage normalisation, in-place on a MethylData.

    Computes a per-sample scaling factor so that each sample's central
    coverage statistic (median by default, or mean) matches a common
    target — the median (or mean) of the per-sample summaries. Read
    counts are scaled and re-integerised, then ``md.store`` is repointed
    at a new ``.cache/normalized`` (or ``<store>_normalized``) partition.

    This prevents deeper-sequenced samples from dominating pooled-count
    tile / region tests downstream. The per-CpG score test in
    ``ep.tl.dmc`` is much less sensitive to coverage imbalance, but
    ``ep.tl.dmr(method='tile')`` is — methylKit's reference pipeline
    inserts ``normalizeCoverage`` between ``filterByCoverage`` and
    ``tileMethylCounts`` for exactly this reason.

    Call order: ``filter_coverage`` → ``normalize_coverage`` → ``unite``.

    Parameters
    ----------
    md : MethylData
        Object whose store has been ``filter_coverage``'d.
    method : {"median", "mean"}
        Central statistic to align. ``"median"`` matches methylKit's
        default and is robust to extreme-coverage tails.

    Raises
    ------
    ValueError
        If ``filter_coverage`` has not been called yet.
    """
    if not md._filtered:
        raise ValueError(
            "Run ep.pp.filter_coverage(md) before ep.pp.normalize_coverage(md)."
        )
    if md._united:
        logger.warning(
            "normalize_coverage called after unite(); the recommended order "
            "is filter → normalize → unite. Re-running unite() is a no-op "
            "but downstream stats reflect the normalised store."
        )

    if md._analysis_root:
        out = str(Path(md._analysis_root) / ".cache" / "normalized")
    else:
        out = f"{md.store}_normalized"

    factors = filter_mod.normalize_coverage_store(
        methylstore_path=md.store,
        output_dir=out,
        method=method,
    )

    md.store = out
    md.uns["normalize"] = {
        "method": method,
        "factors": factors,
    }
    n_sites = _count_parquet_rows(out)
    if n_sites is not None:
        md.uns["n_sites_normalized"] = n_sites
        _append_store_history(md, "normalized", out, n_sites)


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
    """BSmooth-style smoothing (EXPERIMENTAL).

    Smooths per-sample beta values using a fast Gaussian kernel (bandwidth
    in base pairs). Smoothing is applied per-chromosome and per-sample.

    .. warning::
        This function is experimental. The smoothed values are computed but
        are NOT yet wired into downstream DMC/DMR calling. You can use this
        to compute smoothed beta values for manual analysis, but standard
        DMC/DMR calling will still use raw counts.

        To use smoothed values in DMR calling, a future version will add
        a ``use_smoothed=True`` parameter to ``ep.tl.dmr()``.

    Parameters
    ----------
    md : MethylData
        MethylData object (must have been filtered with ep.pp.filter_coverage)
    bandwidth : int
        Gaussian smoothing bandwidth in base pairs (default 1000)
    grid_resolution_bp : int, optional
        Internal grid resolution (default: bandwidth // 20).
        Finer resolution → higher accuracy but slower.

    Raises
    ------
    ValueError
        If called before ``ep.pp.filter_coverage`` (FIX-10).

    Notes
    -----
    Smoothed values are written to ``md.uns["smooth_path"]`` for inspection.
    Currently these are not used by ``ep.tl.dmr()``, but can be accessed
    via the Parquet store for custom downstream analysis.
    """
    # FIX-10: smoothing on unfiltered data silently degrades results.
    if not md._filtered:
        raise ValueError(
            "Run ep.pp.filter_coverage(md) before ep.pp.smooth(md)."
        )

    from .dmr import smooth_methylation_bsmooth

    samples = md.obs.get_column("sample_id").to_list()
    
    # Derive output smooth path
    if md._analysis_root:
        smooth_path = str(Path(md._analysis_root) / ".cache" / "smoothed")
    else:
        smooth_path = f"{md.store}_smoothed"
    
    logger.info(f"Running BSmooth smoothing to {smooth_path}...")
    
    smooth_methylation_bsmooth(
        methylstore_path=md.store,
        samples=samples,
        bandwidth=bandwidth,
        grid_resolution_bp=grid_resolution_bp,
        output_path=smooth_path,
    )
    
    md._smoothed = True
    md.uns["smooth_path"] = smooth_path
    md.uns["smooth_params"] = {
        "bandwidth": bandwidth,
        "grid_resolution_bp": grid_resolution_bp,
    }
    
    logger.info(
        f"✓ Smoothing complete ({len(samples)} samples, {bandwidth} bp bandwidth). "
        f"Results stored in {smooth_path}"
    )