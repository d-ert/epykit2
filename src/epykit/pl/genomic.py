from __future__ import annotations

from ._utils import _get_ax, _save_fig
from ..methyldata import MethylData
import polars as pl


def genomic_context_bar(md: MethylData, ax=None, figsize=(7, 4), save: str | None = None):
    dmc = md.dmc
    if dmc is None or "feature_type" not in dmc.columns:
        raise ValueError("No feature annotations found. Run ep.tl.annotate(md, gtf=...) first.")

    counts = dmc.group_by("feature_type").len().sort("len", descending=True)
    fig, ax = _get_ax(ax, figsize)
    ax.bar(counts["feature_type"].to_list(), counts["len"].to_list())
    ax.set_xlabel("Feature type")
    ax.set_ylabel("Count")
    ax.set_title("Genomic context")
    ax.tick_params(axis="x", rotation=45)

    if save:
        _save_fig(md, fig, save)
    return fig, ax


def cpg_island_pie(md: MethylData, ax=None, figsize=(5, 5), save: str | None = None):
    dmc = md.dmc
    if dmc is None or "cpg_context" not in dmc.columns:
        raise ValueError("No CpG-island annotations found. Run ep.tl.annotate(md, cpg_islands=...) first.")

    counts = dmc.group_by("cpg_context").len().sort("len", descending=True)
    fig, ax = _get_ax(ax, figsize)
    ax.pie(counts["len"].to_list(), labels=counts["cpg_context"].to_list(), autopct="%1.1f%%")
    ax.set_title("CpG island context")

    if save:
        _save_fig(md, fig, save)
    return fig, ax


__all__ = ["genomic_context_bar", "cpg_island_pie"]
