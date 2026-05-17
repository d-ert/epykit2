from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import polars as pl
from matplotlib.figure import Figure


def _get_ax(ax=None, figsize=(6, 4)) -> Tuple[Figure, object]:
    if ax is None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    else:
        return ax.figure, ax


def _save_fig(md, fig: Figure, name: str, out_dir: str | None = None) -> str:
    out = Path(out_dir or "figures")
    path = out / f"{name}.png"
    # Create the *full* parent directory, not just `out` — `name` may itself
    # contain path separators (e.g. save="subdir/figname"), and without this
    # the savefig call fails with FileNotFoundError on the intermediate dir.
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if hasattr(md, "uns"):
        if "figures" not in md.uns:
            md.uns["figures"] = {}
        md.uns["figures"][name] = str(path)
    # Close the figure to release memory
    import matplotlib.pyplot as plt

    plt.close(fig)
    return str(path)


def build_sample_site_matrix(md, n_sites: int = 10_000) -> tuple[np.ndarray, list[str]]:
    """Build a (n_samples × n_sites) β matrix for embedding-style plots.

    Shared between :func:`pca` and :func:`umap`. Picks a random subsample
    of the intersection of CpGs covered in every sample so the matrix
    has no missing values; samples_ordered is returned alongside so plot
    code can colour points by ``md.obs`` columns.
    """
    samples = md.obs.get_column("sample_id").to_list()
    common_sites = None
    for sample in samples:
        sample_sites = (
            pl.scan_parquet(f"{md.store}/sample={sample}/chrom=*/part-*.parquet")
            .select(["chrom", "pos"]).collect().unique()
        )
        common_sites = (
            sample_sites if common_sites is None
            else common_sites.join(sample_sites, on=["chrom", "pos"], how="inner")
        )
    if common_sites is None or len(common_sites) == 0:
        raise ValueError("No common sites across all samples for embedding")

    if len(common_sites) > n_sites:
        common_sites = common_sites.sample(n_sites, seed=42)

    sample_data: list[pl.DataFrame] = []
    for sample in samples:
        sample_data.append(
            pl.scan_parquet(f"{md.store}/sample={sample}/chrom=*/part-*.parquet")
            .select(["chrom", "pos", "N_meth", "coverage", "sample"])
            .join(common_sites.lazy(), on=["chrom", "pos"], how="inner")
            .collect()
        )
    all_data = pl.concat(sample_data)
    all_data = all_data.with_columns(
        pl.when(pl.col("coverage") > 0)
        .then(pl.col("N_meth") / pl.col("coverage"))
        .otherwise(None)
        .alias("beta")
    )
    pivot = all_data.pivot(
        values="beta", index=["chrom", "pos"], on="sample",
        aggregate_function="mean",
    )
    matrix = pivot.select(samples).to_numpy()
    matrix = matrix[~np.isnan(matrix).any(axis=1)]
    if matrix.shape[0] < 2:
        raise ValueError("Not enough valid sites for embedding")
    return matrix.T, samples  # (n_samples, n_sites_valid), samples


__all__ = ["_get_ax", "_save_fig", "build_sample_site_matrix"]
