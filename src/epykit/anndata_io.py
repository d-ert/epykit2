"""AnnData export for ecosystem interop (scanpy / muon / multi-omics).

A MethylData object is conceptually shaped exactly like an AnnData:

    obs       = md.obs                 (n_samples × covariate-columns)
    var       = (chrom, pos)           (n_sites × 2)
    X         = β  or  N_meth  or  coverage   (n_samples × n_sites)
    layers    = {"beta", "coverage", "N_meth", "N_unmeth"}

The site axis can be huge on WGBS (~30 M CpGs on hg38); densifying it
unfiltered would OOM. ``to_anndata`` therefore requires the user to have
called :func:`ep.pp.unite` first so all samples share the same site set.
If not, a clear ``ValueError`` instructs the user.
"""

from __future__ import annotations

import logging

import polars as pl

from .methyldata import MethylData

logger = logging.getLogger(__name__)


def to_anndata(
    md: MethylData,
    *,
    layer: str = "beta",
    populate_layers: bool = True,
    use_smoothed: bool = False,
):
    """Materialise a MethylData as an AnnData object.

    Parameters
    ----------
    md : MethylData
        Methylation data. Must have been ``pp.unite``'d so every sample
        shares the same site set — otherwise densifying the matrix is
        unsafe at WGBS scale.
    layer : {"beta", "coverage", "N_meth", "N_unmeth"}
        Which value to put in ``adata.X``. Default "beta".
    populate_layers : bool
        When True (default), populate ``adata.layers`` with every
        available view (beta, coverage, N_meth, N_unmeth). When False,
        only ``X`` is filled.
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

    valid_layers = {"beta", "coverage", "N_meth", "N_unmeth"}
    if layer not in valid_layers:
        raise ValueError(
            f"layer must be one of {sorted(valid_layers)}; got {layer!r}"
        )

    samples = md.obs.get_column("sample_id").to_list()

    # Materialise the sample × site matrix via one streamed pivot per layer.
    # Using a separate scan per layer keeps peak memory bounded; on a united
    # store the per-sample row count is identical.
    rows = (
        pl.scan_parquet(f"{md.store}/sample=*/chrom=*/part-*.parquet")
        .select(["chrom", "pos", "sample", "N_meth", "coverage"])
        .with_columns(
            (pl.col("coverage") - pl.col("N_meth")).alias("N_unmeth"),
            pl.when(pl.col("coverage") > 0)
              .then(pl.col("N_meth").cast(pl.Float64) / pl.col("coverage"))
              .otherwise(None)
              .alias("beta"),
        )
        .collect()
    )

    # var = unique (chrom, pos) in genomic order. We need the column order
    # because every layer pivot must align with the same site axis.
    var = (
        rows.select(["chrom", "pos"])
        .unique()
        .sort(["chrom", "pos"])
    )
    site_keys = list(zip(var["chrom"].to_list(), var["pos"].to_list()))

    def _pivot_layer(col: str):
        pivot = rows.pivot(
            values=col, index=["chrom", "pos"], on="sample",
            aggregate_function="first",
        ).join(var, on=["chrom", "pos"], how="right").sort(["chrom", "pos"])
        # Ensure every sample column exists (a missing sample on union mode
        # would otherwise drop the column entirely).
        for s in samples:
            if s not in pivot.columns:
                pivot = pivot.with_columns(pl.lit(None).alias(s))
        # AnnData is (samples × sites): transpose at the numpy step.
        return pivot.select(samples).to_numpy().T

    X = _pivot_layer(layer)

    import numpy as np
    import pandas as pd

    obs_pd = md.obs.to_pandas().set_index("sample_id")
    var_pd = pd.DataFrame({
        "chrom": [k[0] for k in site_keys],
        "pos": [k[1] for k in site_keys],
    })
    var_pd.index = [f"{c}:{p}" for c, p in site_keys]

    adata = ad.AnnData(
        X=np.asarray(X, dtype=np.float64),
        obs=obs_pd,
        var=var_pd,
    )

    if populate_layers:
        for col in ("beta", "coverage", "N_meth", "N_unmeth"):
            if col == layer:
                continue
            adata.layers[col] = np.asarray(_pivot_layer(col), dtype=np.float64)

    # Persist a few useful keys
    adata.uns["epykit_assembly"] = md.assembly
    adata.uns["epykit_context"] = md.context
    adata.uns["epykit_state"] = list(md.state)

    logger.info(
        "to_anndata: shape=%s layers=%s",
        adata.shape, list(adata.layers.keys()),
    )
    return adata


__all__ = ["to_anndata"]
