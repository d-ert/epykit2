from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional

import polars as pl


@dataclass
class MethylData:
    """Central data object for WGBS methylation analysis."""

    obs: pl.DataFrame
    store: str
    assembly: str = "unknown"
    context: str = "CpG"

    varm: dict[str, pl.DataFrame] = field(default_factory=dict)
    uns: dict = field(default_factory=dict)

    _filtered: bool = field(default=False, repr=False)
    _united: bool = field(default=False, repr=False)
    _smoothed: bool = field(default=False, repr=False)
    _analysis_root: Optional[str] = field(default=None, repr=False)

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

    @property
    def dmc(self) -> Optional[pl.DataFrame]:
        # Prefer annotated versions for plotting compatibility
        preferred = [
            "dmc_beta_binomial_annotated",
            "dmc_fisher_annotated",
            "dmc_auto_annotated",
            "dmc_beta_binomial",
            "dmc_fisher",
            "dmc_auto",
        ]
        for key in preferred:
            if key in self.varm:
                return self.varm[key]
        for key in sorted(self.varm.keys()):
            if key.startswith("dmc"):
                return self.varm[key]
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
        # If analysis_root is set, save results under analysis_root/results/
        if self._analysis_root:
            out = Path(self._analysis_root) / "results" / Path(path).name
        else:
            out = Path(path)
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
            "_filtered": self._filtered,
            "_united": self._united,
            "_smoothed": self._smoothed,
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
            _filtered=meta.get("_filtered", False),
            _united=meta.get("_united", False),
            _smoothed=meta.get("_smoothed", False),
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

        status: list[str] = []
        if self._filtered:
            status.append("filtered")
        if self._united:
            status.append("united")
        if self._smoothed:
            status.append("smoothed")
        status_str = ", ".join(status) if status else "raw"

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
        rows_html = ""
        display_cols = ["sample_id", "group", "treatment"]
        extra_col = "global_methylation" if "global_methylation" in self.obs.columns else None

        for row in self.obs.iter_rows(named=True):
            treatment = row.get("treatment")
            treat_symbol = "▶" if treatment == 1 else "○"
            extra = row.get(extra_col, "—") if extra_col else "—"
            rows_html += (
                f"<tr><td>{row.get('sample_id', '—')}</td><td>{row.get('group', '—')}</td>"
                f"<td>{treat_symbol}</td><td>{extra}</td></tr>"
            )

        return f"""
        <div style="font-family:monospace;font-size:13px">
        <b>MethylData</b> | assembly: {self.assembly} | context: {self.context}<br>
        <table border="1" style="border-collapse:collapse;margin:8px 0">
          <tr><th>{display_cols[0]}</th><th>{display_cols[1]}</th><th>{display_cols[2]}</th><th>global_meth</th></tr>
          {rows_html}
        </table>
        Results: {', '.join(self.varm.keys()) or 'none yet'}
        </div>
        """
