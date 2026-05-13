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


def _site_index(store: str) -> tuple[pl.DataFrame, dict[tuple[str, int], int]]:
    """Build the sorted (chrom, pos) site index from the Parquet store.

    Returns the index DataFrame and a ``(chrom, pos) → row_idx`` lookup.
    The scan only touches the chrom + pos columns; ``unique()`` collapses
    any duplicates across samples (a united store should have none, but
    the call is idempotent).
    """
    var = (
        pl.scan_parquet(f"{store}/sample=*/chrom=*/part-*.parquet")
        .select(["chrom", "pos"])
        .unique()
        .sort(["chrom", "pos"])
        .collect()
    )
    chroms = var["chrom"].to_list()
    positions = var["pos"].to_list()
    lookup = {(c, int(p)): i for i, (c, p) in enumerate(zip(chroms, positions))}
    return var, lookup


def _fill_layer(
    store: str,
    samples: list[str],
    n_sites: int,
    layer: str,
    lookup: dict[tuple[str, int], int],
    dtype,
):
    """Stream sample-by-sample, filling a (n_samples × n_sites) array."""
    import numpy as np

    out = np.full((len(samples), n_sites), np.nan, dtype=dtype)
    store_p = Path(store)
    for s_idx, sample in enumerate(samples):
        sample_dir = store_p / f"sample={sample}"
        if not sample_dir.exists():
            continue
        # Scan just this sample. Lazy projection: only N_meth / coverage are
        # ever read off disk (plus chrom + pos for the index lookup).
        lf = pl.scan_parquet(f"{sample_dir}/chrom=*/part-*.parquet")
        if layer == "beta":
            lf = lf.select(["chrom", "pos", "N_meth", "coverage"]).filter(
                pl.col("coverage") > 0
            ).with_columns(
                (pl.col("N_meth").cast(pl.Float32) / pl.col("coverage")).alias("value")
            )
        elif layer == "coverage":
            lf = lf.select(["chrom", "pos", "coverage"]).rename({"coverage": "value"})
        elif layer == "N_meth":
            lf = lf.select(["chrom", "pos", "N_meth"]).rename({"N_meth": "value"})
        elif layer == "N_unmeth":
            lf = lf.select(["chrom", "pos", "N_meth", "coverage"]).with_columns(
                (pl.col("coverage") - pl.col("N_meth")).alias("value")
            )
        else:  # pragma: no cover — guarded upstream
            raise ValueError(f"unknown layer {layer!r}")

        df = lf.select(["chrom", "pos", "value"]).collect()
        if len(df) == 0:
            continue

        chroms = df["chrom"].to_list()
        positions = df["pos"].to_list()
        values = df["value"].to_numpy()

        # Build column index for this sample. dict lookup is ~250 ns per
        # entry; for 30 M sites that's ~8 s per sample, which is dwarfed
        # by the Parquet scan itself.
        col_idx = np.fromiter(
            (lookup.get((c, int(p)), -1) for c, p in zip(chroms, positions)),
            count=len(chroms), dtype=np.int64,
        )
        mask = col_idx >= 0
        if not mask.any():
            continue
        out[s_idx, col_idx[mask]] = values[mask].astype(dtype, copy=False)
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
    var, lookup = _site_index(md.store)
    n_sites = len(var)
    logger.info(
        "to_anndata: %d sample(s) × %d site(s) — estimated dense size %.2f GiB",
        len(samples), n_sites,
        len(samples) * n_sites * np_dtype.itemsize / (1024 ** 3),
    )

    X = _fill_layer(md.store, samples, n_sites, layer, lookup, np_dtype)

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
                md.store, samples, n_sites, extra, lookup, np_dtype,
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