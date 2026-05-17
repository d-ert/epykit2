"""nf-core/methylseq run-dir QC ingestion (Plan 2 §7).

Picks up Bismark alignment reports, Qualimap, and preseq outputs from
the run directory and returns a per-sample ``pl.DataFrame`` that
``MethylData.obs`` can be left-joined against.

Parsers here are small targeted regex passes (the same approach MultiQC
uses). No MultiQC dependency.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)


_BISMARK_PATTERNS = {
    "bismark_aligned_reads": re.compile(
        r"Number of alignments analysed in total:\s*(\d+)", re.MULTILINE
    ),
    "bismark_unique_alignments": re.compile(
        r"Number of paired-end alignments with a unique best hit:\s*(\d+)",
        re.MULTILINE,
    ),
    "bismark_mapping_efficiency": re.compile(
        r"Mapping efficiency:\s*([0-9.]+)%", re.MULTILINE
    ),
    "bismark_pct_meth_cpg": re.compile(
        r"C methylated in CpG context:\s*([0-9.]+)%", re.MULTILINE
    ),
    "bismark_pct_meth_chg": re.compile(
        r"C methylated in CHG context:\s*([0-9.]+)%", re.MULTILINE
    ),
    "bismark_pct_meth_chh": re.compile(
        r"C methylated in CHH context:\s*([0-9.]+)%", re.MULTILINE
    ),
}

_QUALIMAP_PATTERNS = {
    "qualimap_mean_coverage": re.compile(
        r"mean coverageData =\s*([0-9.]+)X", re.MULTILINE
    ),
    "qualimap_std_coverage": re.compile(
        r"std coverageData =\s*([0-9.]+)X", re.MULTILINE
    ),
}


def _parse_bismark_report(path: Path) -> dict:
    txt = path.read_text(errors="replace")
    out = {}
    for key, pat in _BISMARK_PATTERNS.items():
        m = pat.search(txt)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                out[key] = m.group(1)
    return out


def _parse_qualimap(path: Path) -> dict:
    txt = path.read_text(errors="replace")
    out = {}
    for key, pat in _QUALIMAP_PATTERNS.items():
        m = pat.search(txt)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                out[key] = m.group(1)
    return out


def _resolve_sample_ids(samplesheet: str) -> list[str]:
    with open(samplesheet, newline="") as fh:
        return [row["sample_id"] for row in csv.DictReader(fh) if "sample_id" in row]


def read_nfcore_methylseq_qc(
    samplesheet: Optional[str],
    run_dir: str,
    *,
    sample_ids: Optional[list[str]] = None,
) -> pl.DataFrame:
    """Walk an nf-core/methylseq run dir and pull per-sample QC metrics.

    Searches for ``<sample>*_PE_report.txt`` / ``<sample>*_SE_report.txt``
    under ``run_dir/`` (recursively) and Qualimap ``genome_results.txt``
    in a Qualimap output dir keyed by sample.

    Parameters
    ----------
    samplesheet : str, optional
        Path to the samplesheet used in the pipeline run. Required when
        ``sample_ids`` isn't provided.
    run_dir : str
        nf-core/methylseq output directory.
    sample_ids : list[str], optional
        Explicit list of sample IDs to look for. Overrides samplesheet.

    Returns
    -------
    pl.DataFrame
        One row per sample with the union of parsed metrics. Missing
        metrics are NaN. The DataFrame can be left-joined onto
        ``md.obs`` via ``obs.join(qc_df, on="sample_id", how="left")``.
    """
    run = Path(run_dir)
    if sample_ids is None:
        if samplesheet is None:
            raise ValueError("pass either samplesheet or sample_ids")
        sample_ids = _resolve_sample_ids(samplesheet)

    rows: list[dict] = []
    for sample in sample_ids:
        record: dict = {"sample_id": sample}
        # Bismark report(s)
        for pattern in (
            f"**/{sample}*_PE_report.txt", f"**/{sample}*_SE_report.txt"
        ):
            for hit in run.glob(pattern):
                try:
                    record.update(_parse_bismark_report(hit))
                    break
                except Exception as exc:
                    logger.warning(
                        "failed to parse Bismark report %s: %s", hit, exc
                    )
            if any(k.startswith("bismark_") for k in record):
                break
        # Qualimap
        for hit in run.glob(f"**/{sample}*/genome_results.txt"):
            try:
                record.update(_parse_qualimap(hit))
                break
            except Exception as exc:
                logger.warning(
                    "failed to parse Qualimap output %s: %s", hit, exc
                )
        rows.append(record)
    return pl.DataFrame(rows)


__all__ = ["read_nfcore_methylseq_qc"]
