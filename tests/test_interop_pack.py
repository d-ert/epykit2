"""Plan 2 §7: ecosystem interop pack — MuData / methylKit / MultiQC / nf-core."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

import epykit as ep


def test_mudata_round_trip_or_skip(synth_md_filtered):
    pytest.importorskip("anndata")
    pytest.importorskip("mudata")
    md = synth_md_filtered
    mu = md.to_mudata(layer="beta")
    assert "meth" in mu.mod


def test_to_methylkit_tabix_writes_files(synth_md_filtered, tmp_path):
    md = synth_md_filtered
    out = md.to_methylkit_tabix(str(tmp_path / "methylkit_out"))
    out_dir = Path(out)
    manifest_path = out_dir / "epykit_to_methylkit.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert "samples" in manifest
    # At least one .txt.gz per sample
    sample_files = list(out_dir.glob("*.methylraw.txt.gz"))
    assert len(sample_files) == len(manifest["samples"])
    assert all(f.stat().st_size > 0 for f in sample_files)


def test_multiqc_export_writes_json(synth_md_filtered, tmp_path):
    md = synth_md_filtered
    # Run a quick QC pass so there's something to write.
    ep.tl.qc(md, run_sample_correlation=True)
    out = ep.report_multiqc(md, str(tmp_path / "mqc"))
    out_dir = Path(out)
    mqc_files = list(out_dir.glob("*_mqc.json"))
    assert mqc_files, "expected at least one *_mqc.json file"
    # Schema sanity
    for f in mqc_files:
        payload = json.loads(f.read_text())
        for key in ("id", "section_name", "plot_type", "data"):
            assert key in payload, f"{f.name} missing {key}"


def test_nfcore_qc_parse_handles_empty(tmp_path):
    # Build a stub samplesheet and run dir.
    sample_ids = ["A", "B"]
    samplesheet = tmp_path / "ss.csv"
    samplesheet.write_text("sample_id,group,path\nA,t,a.cov\nB,c,b.cov\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    out = ep.read_nfcore_methylseq_qc(str(samplesheet), str(run_dir))
    assert isinstance(out, pl.DataFrame)
    assert set(out.get_column("sample_id").to_list()) == set(sample_ids)
