"""Scratch3 — exercise the new epykit features on the 6-sample CD55 dataset.

What's demoed here (everything added in the HTML-report + 4-enablers PR):

  1. ep.pp.aggregate_regions   — methylKit `regionCounts` analogue
  2. ep.pl.tss_metaplot        — matplotlib TSS metaplot
  3. md.to_bedgraph            — IGV/UCSC-friendly per-sample β tracks
  4. md.dmcs_to_bed            — significant DMCs as BED (sortable in IGV)
  5. md.dmrs_to_bed            — DMRs as BED
  6. md.to_anndata             — scanpy / muon interop
  7. md.report                 — interactive HTML report (Jinja2 + Plotly)

scratch2.py runs the analysis pipeline; this script runs the same pipeline
and then layers the new exports on top. Run end-to-end with:

    python scratch3.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename="log_output_new_feats.txt",
)

import epykit as ep
import polars as pl


HERE = Path(__file__).parent
RAW = HERE / "raw_data"
OUT = HERE / "scratch3_out"
OUT.mkdir(exist_ok=True)

import os
os.environ.setdefault("EPYKIT_CONVERT_WORKERS", "1")  # avoid OOM on 4 GB Codespaces


# Phase 1 - Standard pipeline (mirrors scratch2.py)

md = ep.read_bismark(
    str(HERE / "samplesheet.csv"),
    treatment_group="cd55",
    control_group="control",
    assembly="hg38",
    store_dir=str(HERE / "methyl_store_test"),
)
print(md)

ep.pp.filter_coverage(md, lo_count=10, hi_perc=99.9)
ep.pp.normalize_coverage(md, method="median")
ep.pp.unite(md, type="intersect")

ep.tl.qc(md)
ep.tl.dmc(md, test="auto")

total = len(md.dmc)
sig = md.dmc.filter(pl.col("qvalue") < 0.05).height
print(f"DMC: {sig:,}/{total:,} significant @ q<0.05")

ep.tl.dmr(md, tile_size_bp=500, min_cpgs_per_tile=5)
n_dmrs = len(md.uns["dmr"])
print(f"DMR: {n_dmrs:,} tiles called")

ep.tl.annotate(
    md,
    gtf=str(RAW / "gencode.v49.chr_patch_hapl_scaff.annotation.gtf.gz"),
    cpg_islands=str(RAW / "hg38_cpg_islands.bed"),
)
print("Annotation done.")


# Phase 2 - NEW: BedGraph / BED export

print("\n--- New: track exports ---")

# One BedGraph per sample (β-only, β + coverage on one sample as an example).
bg_dir = OUT / "bedgraph"
bg_dir.mkdir(exist_ok=True)
for sample in md.obs.get_column("sample_id").to_list():
    md.to_bedgraph(sample, str(bg_dir / f"{sample}_beta.bedgraph"), value="beta")
print(f"  Wrote {md.n_samples} per-sample β BedGraph(s) to {bg_dir}")

# DMC + DMR BED files for IGV.
md.dmcs_to_bed(str(OUT / "cd55_dmcs.bed"), alpha=0.05, min_abs_diff=0.1)
md.dmrs_to_bed(str(OUT / "cd55_dmrs.bed"))
print(f"  Wrote {OUT/'cd55_dmcs.bed'} and {OUT/'cd55_dmrs.bed'}")


# Phase 3 - NEW: TSS metaplot (matplotlib)

print("\n--- New: TSS metaplot ---")
try:
    ep.pl.tss_metaplot(
        md,
        gtf_path=str(RAW / "gencode.v49.chr_patch_hapl_scaff.annotation.gtf.gz"),
        window_bp=2000,
        n_bins=100,
        group_by="group",
        max_genes=5000,  # cap for runtime; remove for the full genome
        save="tss_metaplot",
    )
    print("  Wrote figures/tss_metaplot.png")
except Exception as exc:
    print(f"  TSS metaplot failed: {exc}")


# Phase 4 - NEW: AnnData export (for multi-omics workflows)

print("\n--- New: AnnData export ---")
try:
    adata = md.to_anndata(layer="beta")
    print(f"  adata.shape = {adata.shape}; layers = {list(adata.layers.keys())}")
    h5_path = OUT / "cd55.h5ad"
    adata.write_h5ad(str(h5_path))
    print(f"  Wrote {h5_path}")
except ImportError as exc:
    print(f"  anndata not installed: {exc}")
except Exception as exc:
    print(f"  to_anndata failed: {exc}")


# Phase 5 - NEW: custom-region aggregation (methylKit regionCounts analogue)
#
# Here we re-aggregate the methylstore to CpG-island regions and run DMC
# on those. This is a totally separate analysis from the per-CpG one we
# just did — the original methylstore is now gone (md.store points at
# the regions store). If you want to keep going on the per-CpG side,
# do this on a fresh md or save() first.

print("\n--- New: region aggregation (CpG islands as regions) ---")
md.save(str(OUT / "cd55_per_cpg_snapshot"))  # so we can restore later if needed

ep.pp.aggregate_regions(
    md,
    regions_bed=str(RAW / "hg38_cpg_islands.bed"),
    min_cpgs_per_region=3,
)
print(f"  Aggregated to {md.uns['regions']['n_regions']:,} CpG-island regions")
md.uns.pop("unite", None)  # re-establish unite on the region store
ep.pp.unite(md, type="intersect")
ep.tl.dmc(md, test="lr")
ri_sig = md.dmc.filter(pl.col("qvalue") < 0.05).height
print(f"  Region-level DMC: {ri_sig:,} / {len(md.dmc):,} significant")


# Phase 6 - NEW: HTML report (covers everything above)

print("\n--- New: HTML report ---")
md.report(
    str(OUT / "cd55_region_report.html"),
    title="CD55 vs Control — region-aggregated",
    gtf_path=str(RAW / "gencode.v49.chr_patch_hapl_scaff.annotation.gtf.gz"),
    alpha=0.05,
    min_abs_diff=0.1,
)
print(f"  Region-level report → {OUT/'cd55_region_report.html'}")

# Reload the per-CpG snapshot and emit a second report on that, so you
# get one file you can compare side-by-side against the region one.
md_cpg = ep.MethylData.load(str(OUT / "cd55_per_cpg_snapshot"))
md_cpg.report(
    str(OUT / "cd55_per_cpg_report.html"),
    title="CD55 vs Control — per CpG",
    gtf_path=str(RAW / "gencode.v49.chr_patch_hapl_scaff.annotation.gtf.gz"),
    alpha=0.05,
    min_abs_diff=0.1,
)
print(f"  Per-CpG report → {OUT/'cd55_per_cpg_report.html'}")

print("\nDone. All new-feature outputs are in:", OUT)