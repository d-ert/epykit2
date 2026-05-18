"""MuData export -- multi-omics interop adapter .

Builds on the existing AnnData export and packages it as a MuData with
the methylation modality keyed ``"meth"``. Optional ``other_modalities``
dict lets the user bundle scRNA / scATAC AnnData objects in the same
container; obs is aligned on ``sample_id``.
"""

from __future__ import annotations

from typing import Optional


def to_mudata(
    md,
    *,
    layer: str = "beta",
    other_modalities: Optional[dict] = None,
):
    """Return a ``MuData`` with methylation as ``'meth'`` modality.

    Parameters
    ----------
    md : MethylData
        Must be ``pp.unite``-d first (same precondition as ``to_anndata``).
    layer : str
        Which methylation matrix to embed as the methylation modality's
        ``X``. Forwarded to :func:`epykit.anndata_io.to_anndata`.
    other_modalities : dict[str, AnnData], optional
        Additional modalities to bundle into the MuData. Each value is
        expected to already be an ``AnnData`` aligned on ``sample_id``.
    """
    try:
        import mudata as md_lib  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "mudata is required for to_mudata. "
            "Install with: pip install 'epykit[anndata]'"
        ) from exc

    from .anndata_io import to_anndata
    adata = to_anndata(md, layer=layer)
    modalities = {"meth": adata}
    if other_modalities:
        modalities.update(other_modalities)
    return md_lib.MuData(modalities)


__all__ = ["to_mudata"]
