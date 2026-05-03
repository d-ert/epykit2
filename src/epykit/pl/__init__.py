from __future__ import annotations

from .._style import apply_theme as _apply_theme

_apply_theme()

from .qc import coverage_histogram, methylation_heatmap
from .differential import volcano, ma_plot, manhattan
from .genomic import genomic_context_bar, cpg_island_pie
from .clustering import pca

__all__ = [
    "coverage_histogram",
    "methylation_heatmap",
    "volcano",
    "ma_plot",
    "manhattan",
    "genomic_context_bar",
    "cpg_island_pie",
    "pca",
]
