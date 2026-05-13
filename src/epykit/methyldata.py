from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional

import polars as pl


@dataclass
class MethylData:
    """Central data object for WGBS methylation analysis.

    Preprocessing state (``filtered``, ``united``, ``smoothed``) is *derived*
    from ``uns`` and its ``_store_history`` log rather than stored as
    independent booleans — see the ``state`` property and the ``_filtered``
    / ``_united`` / ``_smoothed`` aliases below. This means the flags can
    never drift from reality.
    """

    obs: pl.DataFrame
    store: str
    assembly: str = "unknown"
    context: str = "CpG"

    varm: dict[str, pl.DataFrame] = field(default_factory=dict)
    uns: dict = field(default_factory=dict)

    _analysis_root: Optional[str] = field(default=None, repr=False)

    # --- State (derived from uns) -------------------------------------

    @property
    def _filtered(self) -> bool:
        """True iff filter_coverage has been run (recorded in store history)."""
        history = self.uns.get("_store_history", [])
        return any(h.get("step") == "filtered" for h in history)

    @property
    def _united(self) -> bool:
        """True iff pp.unite has been called (records md.uns['unite'])."""
        return "unite" in self.uns

    @property
    def _smoothed(self) -> bool:
        """True iff pp.smooth has been called (records md.uns['smooth_path'])."""
        return "smooth_path" in self.uns

    @property
    def state(self) -> list[str]:
        """Ordered list of preprocessing steps applied to this object.

        Reads ``uns["_store_history"]`` for store-mutating steps (filtered,
        normalized) and appends ``united`` / ``smoothed`` if recorded in
        ``uns``. The result is suitable for ``__repr__`` / ``_repr_html_``.
        """
        history = self.uns.get("_store_history", [])
        steps = [h.get("step") for h in history if h.get("step")]
        if self._united and "united" not in steps:
            steps.append("united")
        if self._smoothed and "smoothed" not in steps:
            steps.append("smoothed")
        return steps

    @property
    def treatment_ids(self) -> list[str]:
        if "treatment" not in self.obs.columns:
            return []
        return (
            self.obs
            .filter(pl.col("treatment") == 1)
            .get_column("sample_id")
            .to_list()
        )

    @property
    def control_ids(self) -> list[str]:
        if "treatment" not in self.obs.columns:
            return []
        return (
            self.obs
            .filter(pl.col("treatment") == 0)
            .get_column("sample_id")
            .to_list()
        )

    @property
    def n_samples(self) -> int:
        return len(self.obs)

    def get_dmc(
        self,
        test: Optional[str] = None,
        annotated: bool = True,
    ) -> Optional[pl.DataFrame]:
        """Look up a DMC table by test name (explicit, recommended).

        Parameters
        ----------
        test : str, optional
            Test backend name (``"lr"``, ``"score"``, ``"glm"``, ...). When
            ``None`` (default), returns the most-recently-written DMC table,
            as recorded by ``ep.tl.dmc`` in ``md.uns["dmc"]["last_key"]``.
        annotated : bool
            When True (default), prefer the ``*_annotated`` variant of the
            requested table if it exists (so plotting code that needs
            ``feature_type`` / ``cpg_context`` works out of the box).

        Returns
        -------
        pl.DataFrame or None
            The matching table, or None if no DMC has been run.
        """
        if test is None:
            key = self.uns.get("dmc", {}).get("last_key")
            if key is None:
                return None
        else:
            key = f"dmc_{test}"
        if annotated:
            ann_key = f"{key}_annotated"
            if ann_key in self.varm:
                return self.varm[ann_key]
        return self.varm.get(key)

    @property
    def dmc(self) -> Optional[pl.DataFrame]:
        """Most-recently-written DMC table (annotated if available).

        Equivalent to ``self.get_dmc(test=None, annotated=True)``. Use
        :meth:`get_dmc` with an explicit ``test=`` argument when running
        multiple tests in one session and you need a specific one. The
        legacy auto-pick-by-priority behaviour (glm > lr > score > ...) has
        been removed because it silently disagreed with the user's most
        recent call.
        """
        # Pointer-first resolution: ep.tl.dmc writes uns["dmc"]["last_key"]
        # on every run. If that's absent (older sessions), fall back to a
        # single existing key — but never auto-prioritize, to avoid the
        # surprise documented in S5.
        last = self.get_dmc()
        if last is not None:
            return last
        # Fallback: if only one dmc_* table is present, return it.
        dmc_keys = [k for k in self.varm if k.startswith("dmc") and not k.endswith("_annotated")]
        if len(dmc_keys) == 1:
            key = dmc_keys[0]
            ann_key = f"{key}_annotated"
            return self.varm.get(ann_key, self.varm.get(key))
        return None

    @property
    def significant_dmcs(self) -> Optional[pl.DataFrame]:
        df = self.dmc
        if df is None:
            return None
        if "qvalue" in df.columns:
            return df.filter(pl.col("qvalue") < 0.05)
        if "pvalue" in df.columns:
            return df.filter(pl.col("pvalue") < 0.05)
        return df

    def save(self, path: str) -> None:
        """Persist obs/varm/uns + manifest to disk.

        Path interpretation:

        * If ``path`` contains any directory components (relative or
          absolute), the data is written there verbatim. ``load(path)``
          reads from the same place — save and load are symmetric.
        * If ``path`` is a bare name (no separators) **and**
          ``_analysis_root`` is set, the data is written under
          ``<_analysis_root>/results/<path>``. This is the
          "analysis-project" convenience layout.
        * If ``path`` is a bare name with no ``_analysis_root``, it's
          treated as a relative path in the current directory.

        The previous behaviour silently re-rooted every call (even
        absolute paths) under ``<_analysis_root>/results/<basename>``,
        which broke save/load symmetry.
        """
        p = Path(path)
        has_components = p.is_absolute() or len(p.parts) > 1
        if self._analysis_root and not has_components:
            out = Path(self._analysis_root) / "results" / p.name
        else:
            out = p
        out.mkdir(parents=True, exist_ok=True)

        self.obs.write_parquet(str(out / "obs.parquet"))

        for name, df in self.varm.items():
            df.write_parquet(str(out / f"varm_{name}.parquet"))

        serialisable_uns = self.uns.copy()
        for key, value in list(serialisable_uns.items()):
            if isinstance(value, pl.DataFrame):
                parquet_name = f"uns_{key}.parquet"
                value.write_parquet(str(out / parquet_name))
                serialisable_uns[key] = {"__parquet__": parquet_name}

        meta = {
            "store": self.store,
            "assembly": self.assembly,
            "context": self.context,
            # _filtered / _united / _smoothed are derived from uns; don't
            # persist them. They are recomputed on load() from the loaded
            # uns dict (which includes _store_history, unite, smooth_path).
            "_analysis_root": self._analysis_root,
            "varm_keys": list(self.varm.keys()),
            "uns": serialisable_uns,
        }
        (out / "methyldata.json").write_text(json.dumps(meta, indent=2, default=str))

    @classmethod
    def load(cls, path: str) -> "MethylData":
        out = Path(path)
        meta = json.loads((out / "methyldata.json").read_text())
        obs = pl.read_parquet(str(out / "obs.parquet"))
        varm = {
            key: pl.read_parquet(str(out / f"varm_{key}.parquet"))
            for key in meta.get("varm_keys", [])
        }

        uns = meta.get("uns", {})
        for key, value in list(uns.items()):
            if isinstance(value, dict) and "__parquet__" in value:
                uns[key] = pl.read_parquet(str(out / value["__parquet__"]))

        md = cls(
            obs=obs,
            store=meta.get("store", ""),
            assembly=meta.get("assembly", "unknown"),
            context=meta.get("context", "CpG"),
            varm=varm,
            uns=uns,
            # _filtered / _united / _smoothed are properties derived from
            # uns — nothing to pass through the constructor. Older saves
            # that include those keys in meta are silently ignored.
        )
        md._analysis_root = meta.get("_analysis_root")
        return md

    def __repr__(self) -> str:
        n_sites = self.uns.get("n_sites_filtered") or self.uns.get("n_sites_raw", "?")
        groups = "unknown"
        if "group" in self.obs.columns:
            grouped = self.obs.group_by("group").len().sort("group")
            groups = "  ".join(
                f"{r['group']} (n={r['len']})"
                for r in grouped.iter_rows(named=True)
            )

        status_str = ", ".join(self.state) if self.state else "raw"

        varm_str = ", ".join(self.varm.keys()) if self.varm else "none yet"
        uns_keys = ", ".join(sorted(self.uns.keys())) if self.uns else "none"

        n_sites_str = f"{n_sites:,}" if isinstance(n_sites, int) else str(n_sites)
        return (
            f"MethylData [{status_str}]\n"
            f"  assembly : {self.assembly}\n"
            f"  n_samples: {self.n_samples} ({groups})\n"
            f"  n_sites  : {n_sites_str}\n"
            f"  context  : {self.context}\n"
            f"  store    : {self.store}\n"
            f"  varm     : {varm_str}\n"
            f"  uns      : {uns_keys}\n"
        )

    def _repr_html_(self) -> str:
        """Render obs as a notebook-friendly table.

        Shows every column in ``self.obs`` rather than a hardcoded subset, so
        user-supplied covariates (sex, batch, age, ...) are visible. Floats are
        rounded to 4 significant figures; the ``treatment`` column, if present,
        renders as ▶ (1) / ○ (0).
        """
        cols = list(self.obs.columns)

        def _fmt(value: object, col: str) -> str:
            if value is None:
                return "—"
            if col == "treatment":
                return "▶" if value == 1 else "○" if value == 0 else str(value)
            if isinstance(value, float):
                if value != value:  # NaN
                    return "—"
                return f"{value:.4g}"
            return str(value)

        rows_html: list[str] = []
        for row in self.obs.iter_rows(named=True):
            cells = "".join(f"<td>{_fmt(row.get(c), c)}</td>" for c in cols)
            rows_html.append(f"<tr>{cells}</tr>")

        header_html = "".join(f"<th>{c}</th>" for c in cols)
        status = ", ".join(self.state) if hasattr(self, "state") and self.state else "raw"

        return f"""
        <div style="font-family:monospace;font-size:13px">
        <b>MethylData</b> [{status}] | assembly: {self.assembly} | context: {self.context}<br>
        <table border="1" style="border-collapse:collapse;margin:8px 0">
          <tr>{header_html}</tr>
          {''.join(rows_html)}
        </table>
        Results: {', '.join(self.varm.keys()) or 'none yet'}
        </div>
        """
