from __future__ import annotations

from pathlib import Path
from typing import Tuple

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
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if hasattr(md, "uns"):
        if "figures" not in md.uns:
            md.uns["figures"] = {}
        md.uns["figures"][name] = str(path)
    # Close the figure to release memory
    import matplotlib.pyplot as plt

    plt.close(fig)
    return str(path)


__all__ = ["_get_ax", "_save_fig"]
