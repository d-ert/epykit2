"""AnnData export for ecosystem interop (scanpy / muon / multi-omics).

A MethylData object is conceptually shaped exactly like an AnnData:

    obs       = md.obs                 (n_samples × covariate-columns)
    var       = (chrom, pos)           (n_sites × 2)
    X         = β  or  N_meth  or  coverage   (n_samples × n_sites)
    layers    = {"beta", "coverage", "N_meth", "N_unmeth"}

Memory strategy
---------------
A dense ``(n_samples × n_sites)`` matrix on real WGBS is enormous — 8
samples × 28 M CpGs × 4 bytes (float32) ≈ 900 MB per layer. The naive
"pivot the long-form DataFrame" approach holds the source rows, the
pivot, and the densified matrix simultaneously, which is typically 3-4×
the final array size and OOMs on real datasets.

To stay in the same ballpark as the final array, ``to_anndata`` runs
streamed, per-sample, per-chromosome:

1. Scan only ``(chrom, pos)`` lazily across the store to build the
   sorted site index and a ``(chrom, pos) → row_idx`` lookup. This step
   uses a few hundred MB at WGBS scale.
2. Allocate ``X`` as a single ``float32`` array of shape
   ``(n_samples, n_sites)``.
3. For each sample, scan only that sample's partition for the requested
   layer column, look the site index up, and fill the matching row of
   ``X``. No long-form intermediate is ever materialised.

Additional layers (``populate_layers=True``) cost one extra dense array
per layer plus one extra streaming pass; default is **only the requested
layer**, so a user who just wants β does not pay 4× memory.

A ``pp.unite`` is required before calling: with a united site set every
sample shares the same axis, the dense layout is sane, and the streaming
fill cannot leave holes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from .methyldata import MethylData

logger = logging.getLogger(__name__)


_VALID_LAYERS = {"beta", "coverage", "N_meth", "N_unmeth"}


def _site_index(store: str):
    """Build the sorted (chrom, pos) site index plus per-chromosome position
    arrays for sorted-search lookup.

    Returns
    -------
    var : pl.DataFrame
        Sorted (chrom, pos) DataFrame, length ``n_sites``.
    chrom_index : dict[str, tuple[int, np.ndarray]]
        ``{chrom: (start_idx, positions_int64)}`` — the slice
        ``var[start_idx:start_idx + len(positions_int64)]`` is the
        chromosome's contiguous block.

    Memory cost is dominated by the per-chromosome int64 position arrays
    (8 B / site). On a 338 M-CpG hg38 union store this is ~2.7 GB, vs.
    the previous Python-dict design which needed ≥ 6-10 GB for the same
    information.
    """
    import numpy as np

    var = (
        pl.scan_parquet(f"{store}/sample=*/chrom=*/part-*.parquet")
        .select(["chrom", "pos"])
        .unique()
        .sort(["chrom", "pos"])
        .collect()
    )
    chrom_index: dict[str, tuple[int, "np.ndarray"]] = {}
    # Build per-chromosome contiguous slices via a single group_by — we
    # don't materialise a dict of (chrom, pos) tuples.
    offsets = (
        var
        .with_row_index("_idx")
        .group_by("chrom", maintain_order=True)
        .agg([
            pl.col("_idx").min().alias("start"),
            pl.col("pos").alias("positions"),
        ])
    )
    for row in offsets.iter_rows(named=True):
        chrom_index[row["chrom"]] = (
            int(row["start"]),
            np.asarray(row["positions"], dtype=np.int64),
        )
    return var, chrom_index


def _value_lazyframe(lf: "pl.LazyFrame", chrom: str, layer: str) -> "pl.LazyFrame":
    """Return a lazy frame with columns (pos, value) for one chromosome."""
    lf = lf.filter(pl.col("chrom") == chrom)
    if layer == "beta":
        return (
            lf.select(["pos", "N_meth", "coverage"])
            .filter(pl.col("coverage") > 0)
            .with_columns(
                (pl.col("N_meth").cast(pl.Float32) / pl.col("coverage")).alias("value")
            )
            .select(["pos", "value"])
        )
    if layer == "coverage":
        return lf.select(["pos", "coverage"]).rename({"coverage": "value"})
    if layer == "N_meth":
        return lf.select(["pos", "N_meth"]).rename({"N_meth": "value"})
    if layer == "N_unmeth":
        return (
            lf.select(["pos", "N_meth", "coverage"])
            .with_columns((pl.col("coverage") - pl.col("N_meth")).alias("value"))
            .select(["pos", "value"])
        )
    raise ValueError(f"unknown layer {layer!r}")  # pragma: no cover


def _fill_layer(
    store: str,
    samples: list[str],
    n_sites: int,
    layer: str,
    chrom_index: dict[str, tuple[int, "object"]],
    dtype,
):
    """Stream sample × chromosome, filling a (n_samples × n_sites) array.

    For each (sample, chromosome) we read just that partition file, then
    use ``np.searchsorted`` against the chromosome's pre-built position
    array to compute the destination column indices in O(n_rows · log n_chrom).
    No Python-side dict is ever built — the only persistent index
    structures are the per-chromosome int64 position arrays.
    """
    import numpy as np

    out = np.full((len(samples), n_sites), np.nan, dtype=dtype)
    store_p = Path(store)

    for s_idx, sample in enumerate(samples):
        sample_dir = store_p / f"sample={sample}"
        if not sample_dir.exists():
            continue
        for chrom, (chrom_start, var_positions) in chrom_index.items():
            chrom_part = sample_dir / f"chrom={chrom}"
            if not chrom_part.exists():
                continue
            lf = pl.scan_parquet(f"{chrom_part}/part-*.parquet")
            df = _value_lazyframe(lf, chrom, layer).collect()
            if len(df) == 0:
                continue

            sample_positions = df["pos"].to_numpy().astype(np.int64, copy=False)
            values = df["value"].to_numpy()

            # Look up the column index of each sample CpG inside the
            # chromosome's var slice via a single sorted-search call.
            local_idx = np.searchsorted(var_positions, sample_positions)
            # Guard against positions that fall off the end of the array
            # (would happen on a *union* store where a sample contributes
            # a CpG that no other sample has — searchsorted returns
            # len(arr)). Bounds-check first, then equality-check.
            in_range = local_idx < len(var_positions)
            local_idx_safe = np.where(in_range, local_idx, 0)
            hit = in_range & (var_positions[local_idx_safe] == sample_positions)
            if not hit.any():
                continue
            global_idx = chrom_start + local_idx_safe[hit]
            out[s_idx, global_idx] = values[hit].astype(dtype, copy=False)
    return out


def to_anndata(
    md: MethylData,
    *,
    layer: str = "beta",
    populate_layers: bool = False,
    dtype: str = "float32",
    use_smoothed: bool = False,
):
    """Materialise a MethylData as an AnnData object.

    Memory-conscious by default: only the layer you asked for is dense.
    On a typical 8-sample × 28 M-CpG run this produces a ~900 MB
    ``adata.X`` and nothing else, vs. ~3.5 GB of intermediate state under
    the previous pivot-based implementation. Pass
    ``populate_layers=True`` to also fill ``adata.layers["coverage"]``,
    etc., paying one extra dense array per layer.

    Parameters
    ----------
    md : MethylData
        Methylation data. Must have been ``pp.unite``'d so every sample
        shares the same site set.
    layer : {"beta", "coverage", "N_meth", "N_unmeth"}
        Which value to put in ``adata.X``. Default "beta".
    populate_layers : bool
        If True, also fill ``adata.layers`` with every layer in
        :data:`_VALID_LAYERS`. Default **False** — the old default of
        True densified four matrices and was the most common cause of
        OOMs on real WGBS. Layer matrices share the site index so an
        extra pass per layer is cheap on disk but doubles dense RAM.
    dtype : str
        NumPy dtype for the dense matrices. Default ``"float32"``
        (4 bytes / cell). Use ``"float64"`` if you want to feed the
        result into algorithms that demand it; halves the per-sample row
        fill speed and doubles RAM.
    use_smoothed : bool
        Reserved for future use; currently raises NotImplementedError.

    Returns
    -------
    anndata.AnnData
        Shape ``(n_samples, n_sites)`` matching ``md.obs`` × the united
        site axis.
    """
    if use_smoothed:
        raise NotImplementedError(
            "to_anndata(use_smoothed=True) not implemented yet — the smoothed "
            "store has a per-sample β grid, not raw counts."
        )

    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError(
            "anndata is required for to_anndata(). "
            "Install it with: pip install 'epykit[anndata]'"
        ) from exc

    if not md._united:
        raise ValueError(
            "to_anndata() requires a united methylstore so all samples share "
            "the same site axis. Run ep.pp.unite(md, type='intersect') first "
            "(or type='union' if you want every site that appears in any "
            "sample — note the densification cost)."
        )

    if layer not in _VALID_LAYERS:
        raise ValueError(
            f"layer must be one of {sorted(_VALID_LAYERS)}; got {layer!r}"
        )

    import numpy as np
    import pandas as pd

    np_dtype = np.dtype(dtype)
    samples = md.obs.get_column("sample_id").to_list()

    logger.info("to_anndata: building site index from %s", md.store)
    var, chrom_index = _site_index(md.store)
    n_sites = len(var)
    dense_gib = len(samples) * n_sites * np_dtype.itemsize / (1024 ** 3)
    logger.info(
        "to_anndata: %d sample(s) × %d site(s) — estimated dense size %.2f GiB per layer",
        len(samples), n_sites, dense_gib,
    )
    # Loud warning if the user is about to allocate something huge. 4 GiB
    # crosses the threshold where even 32 GiB workstations start swapping
    # once anndata's own copies are factored in.
    if dense_gib > 4.0:
        logger.warning(
            "to_anndata: dense X is %.1f GiB. Consider filtering to a smaller "
            "site set first (e.g. ep.pp.filter_coverage + ep.pp.unite with "
            "type='intersect', or ep.pp.aggregate_regions to a BED of "
            "promoters/peaks) before exporting.",
            dense_gib,
        )

    X = _fill_layer(md.store, samples, n_sites, layer, chrom_index, np_dtype)

    obs_pd = md.obs.to_pandas().set_index("sample_id")

    chrom_arr = var["chrom"].to_list()
    pos_arr = var["pos"].to_list()
    var_pd = pd.DataFrame({"chrom": chrom_arr, "pos": pos_arr})
    var_pd.index = [f"{c}:{p}" for c, p in zip(chrom_arr, pos_arr)]

    adata = ad.AnnData(X=X, obs=obs_pd, var=var_pd)

    if populate_layers:
        for extra in _VALID_LAYERS:
            if extra == layer:
                continue
            logger.info("to_anndata: filling layer %r", extra)
            adata.layers[extra] = _fill_layer(
                md.store, samples, n_sites, extra, chrom_index, np_dtype,
            )

    adata.uns["epykit_assembly"] = md.assembly
    adata.uns["epykit_context"] = md.context
    adata.uns["epykit_state"] = list(md.state)

    logger.info(
        "to_anndata: built AnnData shape=%s layers=%s",
        adata.shape, list(adata.layers.keys()),
    )
    return adata


__all__ = ["to_anndata"]