from __future__ import annotations

from pathlib import Path

import polars as pl

from epykit.methyldata import MethylData


def _make_methyldata(store_dir: str) -> MethylData:
    obs = pl.DataFrame(
        {
            "sample_id": ["case1", "ctrl1"],
            "group": ["cd55", "control"],
            "treatment": [1, 0],
            "path": ["case1.cov", "ctrl1.cov"],
        }
    )
    return MethylData(obs=obs, store=store_dir, assembly="hg38", context="CpG", uns={})


def test_tl_dmc_defaults_to_union_when_unite_not_called(monkeypatch, tmp_path):
    from epykit import tl

    md = _make_methyldata(str(tmp_path / "store"))

    seen = {}

    def fake_process_chromosomes_dmc(**kwargs):
        seen.update(kwargs)
        return pl.DataFrame({"chrom": ["chr1"], "pos": [1], "strand": ["+"], "pvalue": [0.1]})

    monkeypatch.setattr(tl, "process_chromosomes_dmc", fake_process_chromosomes_dmc)
    monkeypatch.setattr(
        tl,
        "apply_multiple_testing_correction",
        lambda df, method="fdr_bh": df,
    )

    tl.dmc(md, test="fisher")

    assert seen["unite"] is False
    assert md.uns["dmc"]["unite"] is False


def test_tl_dmc_uses_intersection_only_after_unite(monkeypatch, tmp_path):
    from epykit import tl

    md = _make_methyldata(str(tmp_path / "store"))
    md.uns["unite"] = {"type": "intersect"}

    seen = {}

    def fake_process_chromosomes_dmc(**kwargs):
        seen.update(kwargs)
        return pl.DataFrame({"chrom": ["chr1"], "pos": [1], "strand": ["+"], "pvalue": [0.1]})

    monkeypatch.setattr(tl, "process_chromosomes_dmc", fake_process_chromosomes_dmc)
    monkeypatch.setattr(
        tl,
        "apply_multiple_testing_correction",
        lambda df, method="fdr_bh": df,
    )

    tl.dmc(md, test="fisher")

    assert seen["unite"] is True
    assert md.uns["dmc"]["unite"] is True


def test_count_helpers_use_parquet_metadata(tmp_path):
    from epykit.io import _count_store_rows
    from epykit.pp import _count_parquet_rows

    store = tmp_path / "store"
    for sample in ["case1", "ctrl1"]:
        part_dir = store / f"sample={sample}" / "chrom=chr1"
        part_dir.mkdir(parents=True)
        pl.DataFrame({"x": [1, 2, 3]}).write_parquet(part_dir / "part-0.parquet")

    assert _count_store_rows(str(store)) == 6
    assert _count_parquet_rows(str(store)) == 6


def test_smooth_streams_to_disk(tmp_path):
    from epykit.dmr import smooth_methylation_bsmooth

    store = tmp_path / "store"
    part_dir = store / "sample=case1" / "chrom=chr1"
    part_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "pos": [10, 20, 30, 40],
            "N_meth": [1, 2, 3, 4],
            "coverage": [2, 2, 3, 4],
        }
    ).write_parquet(part_dir / "part-0.parquet")

    out = tmp_path / "smooth"
    result = smooth_methylation_bsmooth(
        str(store),
        ["case1"],
        bandwidth=10,
        grid_resolution_bp=1,
        output_path=str(out),
    )

    assert result is None
    out_part = out / "sample=case1" / "chrom=chr1" / "part-0.parquet"
    assert out_part.exists()
    written = pl.read_parquet(out_part)
    assert written.columns == ["chrom", "pos", "sample", "beta_raw", "beta_smooth"]
    assert len(written) == 4


def test_pp_smooth_uses_output_directory(monkeypatch, tmp_path):
    from epykit import pp

    obs = pl.DataFrame(
        {
            "sample_id": ["case1"],
            "group": ["cd55"],
            "treatment": [1],
            "path": ["case1.cov"],
        }
    )
    md = pp.MethylData(obs=obs, store=str(tmp_path / "store"), assembly="hg38")

    seen = {}

    def fake_smooth_methylation_bsmooth(**kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(pp, "smooth_methylation_bsmooth", fake_smooth_methylation_bsmooth)

    pp.smooth(md, bandwidth=25, grid_resolution_bp=5)

    assert seen["methylstore_path"] == md.store
    assert seen["samples"] == ["case1"]
    assert seen["output_path"] == f"{md.store}_smooth"
    assert md._smoothed is True
    assert md.uns["smooth_path"] == f"{md.store}_smooth"


def test_cache_directory_layout_created_by_read_bismark(monkeypatch, tmp_path):
    """Verify that read_bismark creates .cache/raw subdirectory structure."""
    from pathlib import Path
    
    # Create a minimal samplesheet
    samplesheet = tmp_path / "samplesheet.csv"
    samplesheet.write_text("sample_id,group,path\ntest1,control,/fake/path.cov\n")
    
    # Mock ensure_converted_sample to avoid actual file I/O
    def fake_ensure_converted(*args, **kwargs):
        return True
    
    from epykit import io
    monkeypatch.setattr(io, "ensure_converted_sample", fake_ensure_converted)
    monkeypatch.setattr(io, "_count_store_rows", lambda x: 1000)
    
    store_dir = str(tmp_path / "analysis_root")
    md = io.read_bismark(
        str(samplesheet),
        treatment_group="control",
        control_group="control",
        store_dir=store_dir,
    )
    
    # Verify the cache directory structure
    cache_root = Path(store_dir) / ".cache" / "raw"
    assert cache_root.exists()
    assert md.store == str(cache_root)
    assert md._analysis_root == str(Path(store_dir))


def test_cache_filtered_directory_created_by_filter_coverage(monkeypatch, tmp_path):
    """Verify that filter_coverage creates .cache/filtered subdirectory."""
    from pathlib import Path
    
    obs = pl.DataFrame({
        "sample_id": ["case1"],
        "group": ["cd55"],
        "treatment": [1],
        "path": ["case1.cov"],
    })
    analysis_root = tmp_path / "analysis_root"
    cache_raw = analysis_root / ".cache" / "raw"
    cache_raw.mkdir(parents=True, exist_ok=True)
    
    md = _make_methyldata(str(cache_raw))
    md._analysis_root = str(analysis_root)
    
    # Mock filter_sites to avoid actual filtering
    def fake_filter_sites(*args, **kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sample=case1" / "chrom=chr1").mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"x": [1]}).write_parquet(str(output_dir / "sample=case1" / "chrom=chr1" / "part-0.parquet"))
    
    from epykit import pp, filter as filter_mod
    monkeypatch.setattr(filter_mod, "filter_sites", fake_filter_sites)
    
    pp.filter_coverage(md, lo_count=10, hi_perc=99.9)
    
    # Verify the filtered cache directory was created
    cache_filtered = analysis_root / ".cache" / "filtered"
    assert cache_filtered.exists()
    assert md.store == str(cache_filtered)
    assert md._filtered is True


def test_cache_smoothed_directory_created_by_smooth(tmp_path):
    """Verify that smooth creates .cache/smoothed subdirectory."""
    from pathlib import Path
    
    obs = pl.DataFrame({
        "sample_id": ["case1"],
        "group": ["cd55"],
        "treatment": [1],
        "path": ["case1.cov"],
    })
    analysis_root = tmp_path / "analysis_root"
    cache_filtered = analysis_root / ".cache" / "filtered"
    cache_filtered.mkdir(parents=True, exist_ok=True)
    
    md = _make_methyldata(str(cache_filtered))
    md._analysis_root = str(analysis_root)
    
    # Verify smooth derives the correct output path
    expected_smooth_path = str(analysis_root / ".cache" / "smoothed")
    
    # Mock smooth_methylation_bsmooth to verify it gets called with correct path
    smooth_calls = []
    from epykit import pp, dmr
    original_smooth = pp.smooth_methylation_bsmooth
    
    def fake_smooth_methylation_bsmooth(**kwargs):
        smooth_calls.append(kwargs)
        return None
    
    pp.smooth_methylation_bsmooth = fake_smooth_methylation_bsmooth
    try:
        pp.smooth(md, bandwidth=1000)
        assert len(smooth_calls) == 1
        assert smooth_calls[0]["output_path"] == expected_smooth_path
        assert md.uns["smooth_path"] == expected_smooth_path
    finally:
        pp.smooth_methylation_bsmooth = original_smooth
