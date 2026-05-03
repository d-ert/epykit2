from __future__ import annotations

PALETTE = {
    "hyper": "#e05263",
    "hypo": "#4a90d9",
    "neutral": "#aaaaaa",
    "island": "#2ca02c",
    "shore": "#98df8a",
    "shelf": "#dbf9db",
    "open_sea": "#d3d3d3",
    "treatment": "#e05263",
    "control": "#4a90d9",
}


def apply_theme(context: str = "paper") -> None:
    try:
        import matplotlib as mpl
    except Exception:
        return

    base = {
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "font.family": "sans-serif",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }

    mpl.rcParams.update(base)


__all__ = ["PALETTE", "apply_theme"]
