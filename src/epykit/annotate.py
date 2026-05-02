"""Genomic annotation for DMC / DMR results.

Phase 3 of the epykit pipeline:
  - annotate_features: overlaps sites/regions with gene-level features from a
    GTF/GFF3 (promoter, exon, intron, intergenic) and returns the nearest
    gene with distance-to-TSS.
  - annotate_cpg_islands: classifies each site as island / shore / shelf /
    open_sea using a UCSC CpGIsland BED track.

Performance changes vs previous version
----------------------------------------
FIX-1  _build_intron_df
    Old: double Python loop — `for gene_id, group in groupby:` with inner
         `.iterrows()`.  O(n_genes × n_exons) Python iterations; dominated
         runtime for whole-genome GTFs.
    New: fully vectorised pandas — sort → groupby shift → boolean filter.
         No Python-level loops.

FIX-2  _pick_best_overlap
    Old: grouped on ["Chromosome", "Start"] — composite key, collision-prone
         in DMR mode.
    New: group on "_row_idx" (unique per input site); simpler and correct.

FIX-3  annotate_features result columns
    Old: `for _, row in best.iterrows()` (slow) to build a dict, then
         `for i in range(n): annotation_hits.get(i)` (slow) to materialise
         columns — two O(n_sites) Python loops.
    New: single pandas merge of the overlap result back onto the full site
         index; all column materialisation is numpy/pandas C-level.

FIX-4  TSS distance
    Old: computed per-site inside the O(n) Python loop via dict lookup +
         arithmetic.
    New: pandas Series.map lookup (C-level hash map) followed by a single
         numpy subtraction across all sites.

FIX-5  annotate_cpg_islands per-hit assignment loop
    Old: `for idx in hit_idxs: cpg_context[idx] = label` — Python loop over
         potentially millions of overlap indices, called 3×.
    New: `cpg_context[hit_idxs_array] = label` — numpy fancy indexing, O(1)
         overhead per overlap call.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Hierarchical priority for overlapping features (lower index = higher priority)
_FEATURE_PRIORITY: dict[str, int] = {
    "promoter":   0,
    "exon":       1,
    "intron":     2,
    "intergenic": 3,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sites_to_pyranges(sites: pl.DataFrame) -> "pr.PyRanges":
    """Convert a DMC (chrom/pos) or DMR (chrom/start/end) DataFrame to
    a PyRanges object.
    """
    import pyranges as pr

    if "start" in sites.columns and "end" in sites.columns:
        return pr.PyRanges(
            chromosomes=sites["chrom"].to_list(),
            starts=sites["start"].to_list(),
            ends=sites["end"].to_list(),
        )
    else:
        pos = sites["pos"].to_list()
        return pr.PyRanges(
            chromosomes=sites["chrom"].to_list(),
            starts=pos,
            ends=[p + 1 for p in pos],
        )


def _build_promoter_df(
    genes_pd,
    upstream_bp: int,
    downstream_bp: int,
) -> "pd.DataFrame":
    """Construct promoter intervals from a genes pandas DataFrame (unchanged)."""
    import pandas as pd

    plus  = genes_pd[genes_pd["Strand"] == "+"].copy()
    minus = genes_pd[genes_pd["Strand"] == "-"].copy()

    plus["End"]   = plus["Start"] + downstream_bp
    plus["Start"] = (plus["Start"] - upstream_bp).clip(lower=0)

    tss_minus     = minus["End"].copy()
    minus["Start"] = (tss_minus - downstream_bp).clip(lower=0)
    minus["End"]   = tss_minus + upstream_bp

    combined = pd.concat([plus, minus], ignore_index=True)
    combined["Feature"] = "promoter"
    return combined


def _build_intron_df(exons_pd, genes_pd) -> "pd.DataFrame":
    """Derive intronic intervals as gene body minus exon intervals.

    FIX-1: fully vectorised pandas — no per-gene Python loops.

    Algorithm
    ---------
    1. Join exons to gene metadata (boundary + strand) in one merge.
    2. Clip each exon to its gene boundaries.
    3. Sort by (gene_id, Start); use groupby + shift to get prev_End per exon.
       For the first exon in each gene, prev_End = gene_Start.
    4. Rows where prev_End < Start are introns.
    """
    import pandas as pd

    _RESULT_COLS = [
        "Chromosome", "Start", "End", "Strand", "Feature", "gene_id", "gene_name"
    ]

    if len(exons_pd) == 0 or len(genes_pd) == 0:
        return pd.DataFrame(columns=_RESULT_COLS)

    # Build a gene-level lookup (one row per gene_id)
    gene_meta = (
        genes_pd
        .drop_duplicates("gene_id")
        .set_index("gene_id")[["Chromosome", "Start", "End", "Strand", "gene_name"]]
        .rename(columns={
            "Chromosome": "_g_chrom",
            "Start":      "_g_start",
            "End":        "_g_end",
        })
    )

    # Merge exon start/end with gene metadata
    ex = (
        exons_pd[["gene_id", "Start", "End"]]
        .join(gene_meta, on="gene_id", how="inner")
        .copy()
    )

    # Clip exon coords to gene span
    ex["Start"] = ex[["Start", "_g_start"]].max(axis=1).astype(np.int64)
    ex["End"]   = ex[["End",   "_g_end"]  ].min(axis=1).astype(np.int64)
    ex = ex[ex["Start"] < ex["End"]].copy()

    if len(ex) == 0:
        return pd.DataFrame(columns=_RESULT_COLS)

    # Sort so shift gives the previous exon's End within each gene
    ex = ex.sort_values(["gene_id", "Start"]).reset_index(drop=True)

    # prev_End: for first exon per gene → gene_Start; else end of prev exon
    ex["_prev_end"] = ex.groupby("gene_id", sort=False)["End"].shift(1)
    first_mask = ex["_prev_end"].isna()
    ex.loc[first_mask, "_prev_end"] = ex.loc[first_mask, "_g_start"]
    ex["_prev_end"] = ex["_prev_end"].astype(np.int64)

    # Rows where the gap before this exon is non-zero → intron
    introns = ex[ex["_prev_end"] < ex["Start"]].copy()

    if len(introns) == 0:
        return pd.DataFrame(columns=_RESULT_COLS)

    introns["_intron_end"] = introns["Start"]         # current exon start = intron end
    introns["Start"]       = introns["_prev_end"]     # previous exon end = intron start
    introns["End"]         = introns["_intron_end"]
    introns["Feature"]     = "intron"
    introns                = introns.rename(columns={"_g_chrom": "Chromosome"})

    return introns[_RESULT_COLS].reset_index(drop=True)


def _pick_best_overlap(joined_df) -> "pd.DataFrame":
    """When a site overlaps multiple features, keep only the highest-priority one.

    FIX-2: group on _row_idx (unique per input site) instead of the
    composite [Chromosome, Start] key which can collide in DMR mode.
    """
    df = joined_df.copy()
    df["_priority"] = df["Feature_b"].map(_FEATURE_PRIORITY).fillna(99)
    return (
        df.sort_values("_priority")
          .groupby("_row_idx", as_index=False)
          .first()
          .drop(columns=["_priority"])
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate_features(
    sites: pl.DataFrame,
    annotation_gtf: str,
    features: list[str] | tuple[str, ...] = ("promoter", "exon", "intron", "intergenic"),
    promoter_upstream_bp: int = 2000,
    promoter_downstream_bp: int = 200,
) -> pl.DataFrame:
    """Annotate DMC / DMR sites with gene-level genomic features.

    Parameters
    ----------
    sites : pl.DataFrame
        DMC output (columns: chrom, pos, …) or DMR output
        (columns: chrom, start, end, …).
    annotation_gtf : str
        Path to a GTF or GFF3 file (Ensembl / UCSC format).
    features : sequence of str
        Feature types to annotate.  Supported: "promoter", "exon", "intron",
        "intergenic".  Priority when overlapping: promoter > exon > intron >
        intergenic.
    promoter_upstream_bp : int
        Bases upstream of TSS included in the promoter window (default 2000).
    promoter_downstream_bp : int
        Bases downstream of TSS included in the promoter window (default 200).

    Returns
    -------
    pl.DataFrame
        Input DataFrame with four additional columns appended:
        gene_id (Utf8), gene_name (Utf8), feature_type (Utf8),
        distance_to_tss (Float32).
    """
    try:
        import pyranges as pr
    except ImportError as exc:
        raise ImportError(
            "pyranges is required for annotation. "
            "Install with: pip install pyranges"
        ) from exc

    import pandas as pd

    logger.info("annotate_features: loading %s", annotation_gtf)
    gtf = pr.read_gtf(annotation_gtf, as_df=False)

    genes_pd = gtf[gtf.Feature == "gene"].df
    exons_pd = gtf[gtf.Feature == "exon"].df

    if "gene_id" not in genes_pd.columns:
        raise ValueError("GTF is missing the 'gene_id' attribute column")

    if "gene_name" not in genes_pd.columns:
        genes_pd["gene_name"] = genes_pd["gene_id"]
    if "gene_name" not in exons_pd.columns:
        exons_pd = exons_pd.merge(
            genes_pd[["gene_id", "gene_name"]].drop_duplicates(),
            on="gene_id", how="left",
        )

    # ------------------------------------------------------------------
    # Build feature DataFrame (promoter / exon / intron)
    # ------------------------------------------------------------------
    feature_dfs: list[pd.DataFrame] = []

    if "promoter" in features:
        promoter_df = _build_promoter_df(
            genes_pd, promoter_upstream_bp, promoter_downstream_bp
        )
        feature_dfs.append(
            promoter_df[[
                "Chromosome", "Start", "End", "Strand",
                "Feature", "gene_id", "gene_name"
            ]]
        )

    if "exon" in features and len(exons_pd) > 0:
        ex = exons_pd[[
            "Chromosome", "Start", "End", "Strand", "gene_id", "gene_name"
        ]].copy()
        ex["Feature"] = "exon"
        feature_dfs.append(ex)

    if "intron" in features and len(exons_pd) > 0 and len(genes_pd) > 0:
        intron_df = _build_intron_df(exons_pd, genes_pd)
        if len(intron_df) > 0:
            feature_dfs.append(intron_df)

    # ------------------------------------------------------------------
    # Attach row indices to sites PyRanges for re-join after overlap
    # ------------------------------------------------------------------
    sites_pr_base = _sites_to_pyranges(sites)
    sites_pd      = sites_pr_base.df.copy()
    n             = len(sites)
    sites_pd["_row_idx"] = np.arange(n, dtype=np.int32)
    sites_pr2    = pr.PyRanges(sites_pd)

    # ------------------------------------------------------------------
    # Overlap annotation → FIX-3: use pandas merge instead of Python loops
    # ------------------------------------------------------------------
    # Initialise defaults (will be overwritten for annotated sites)
    annot_cols = pd.DataFrame({
        "_row_idx":    np.arange(n, dtype=np.int32),
        "gene_id":     "",
        "gene_name":   "",
        "feature_type": "intergenic",
    })

    if feature_dfs:
        all_features_df = pd.concat(feature_dfs, ignore_index=True)
        features_pr     = pr.PyRanges(all_features_df)
        joined          = sites_pr2.join(features_pr, how="left", suffix="_b")

        if len(joined.df) > 0:
            joined_df = joined.df.copy()
            best      = _pick_best_overlap(joined_df)

            # FIX-3: merge back onto the full site index in one pandas op;
            # no iterrows, no Python dict loop.
            best_slim = (
                best[["_row_idx", "gene_id", "gene_name", "Feature_b"]]
                .rename(columns={"Feature_b": "feature_type"})
            )
            # Left-merge so every row in 0..n-1 is represented
            annot_cols = (
                annot_cols[["_row_idx"]]
                .merge(best_slim, on="_row_idx", how="left")
            )
            annot_cols["gene_id"]      = annot_cols["gene_id"].fillna("").astype(str)
            annot_cols["gene_name"]    = annot_cols["gene_name"].fillna("").astype(str)
            annot_cols["feature_type"] = annot_cols["feature_type"].fillna("intergenic").astype(str)

    gene_ids      = annot_cols["gene_id"].to_numpy(dtype=object)
    gene_names    = annot_cols["gene_name"].to_numpy(dtype=object)
    feature_types = annot_cols["feature_type"].to_numpy(dtype=object)

    # ------------------------------------------------------------------
    # TSS distance — FIX-4: vectorised pandas map + single numpy subtraction
    # ------------------------------------------------------------------
    # tss_map: gene_id → tss_position (strand-aware)
    tss_map: dict[str, int] = {}
    for _, row in genes_pd.iterrows():
        strand = row.get("Strand", "+")
        tss    = int(row["Start"]) if strand != "-" else int(row["End"])
        tss_map[row["gene_id"]] = tss

    tss_series = pd.Series(tss_map, dtype="Int64")  # nullable int for missing

    if "pos" in sites.columns:
        site_mids = sites["pos"].to_numpy().astype(np.float64)
    else:
        site_mids = (
            (sites["start"].to_numpy() + sites["end"].to_numpy()) / 2.0
        ).astype(np.float64)

    # C-level hash map lookup via pandas .map
    tss_positions = (
        pd.Series(gene_ids.tolist())
        .map(tss_series)
        .to_numpy(dtype=np.float64, na_value=np.nan)
    )
    dist_to_tss = (site_mids - tss_positions).astype(np.float32)
    # Sites with no gene annotation get NaN
    dist_to_tss[gene_ids == ""] = np.nan

    return sites.with_columns([
        pl.Series("gene_id",        gene_ids.tolist(),      dtype=pl.Utf8),
        pl.Series("gene_name",      gene_names.tolist(),    dtype=pl.Utf8),
        pl.Series("feature_type",   feature_types.tolist(), dtype=pl.Utf8),
        pl.Series("distance_to_tss", dist_to_tss,           dtype=pl.Float32),
    ])


def annotate_cpg_islands(
    sites: pl.DataFrame,
    cpg_island_bed: str,
) -> pl.DataFrame:
    """Classify each CpG site by CpG-island context.

    Context definitions (UCSC convention):
      island   — site overlaps a CpG island
      shore    — site is within ±2 kb of an island boundary
      shelf    — site is within ±2–4 kb of an island boundary
      open_sea — everything else

    Parameters
    ----------
    sites : pl.DataFrame
        DMC (chrom, pos) or DMR (chrom, start, end) DataFrame.
    cpg_island_bed : str
        Path to a BED file with CpG island coordinates.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with one additional column:
        cpg_context (Utf8) — one of "island", "shore", "shelf", "open_sea".
    """
    try:
        import pyranges as pr
    except ImportError as exc:
        raise ImportError(
            "pyranges is required for CpG island annotation. "
            "Install with: pip install pyranges"
        ) from exc

    import pandas as pd

    logger.info("annotate_cpg_islands: loading %s", cpg_island_bed)
    islands = pr.read_bed(cpg_island_bed)

    if len(islands.df) == 0:
        logger.warning("CpG island BED file is empty; all sites → open_sea")
        return sites.with_columns(pl.lit("open_sea").alias("cpg_context"))

    SHORE_DIST = 2_000
    SHELF_DIST = 4_000

    islands_df = islands.df.copy()

    def _flanks(df: pd.DataFrame, inner: int, outer: int,
                label: str) -> pd.DataFrame:
        upstream = df[["Chromosome", "Start", "End"]].copy()
        upstream["End"]   = (upstream["Start"] - inner).clip(lower=0)
        upstream["Start"] = (upstream["Start"] - outer).clip(lower=0)
        upstream["_ctx"]  = label

        downstream = df[["Chromosome", "Start", "End"]].copy()
        downstream["Start"] = downstream["End"] + inner
        downstream["End"]   = downstream["End"] + outer
        downstream["_ctx"]  = label

        return pd.concat([upstream, downstream], ignore_index=True)

    shore_df = _flanks(islands_df, 0,          SHORE_DIST, "shore")
    shelf_df = _flanks(islands_df, SHORE_DIST, SHELF_DIST, "shelf")
    islands_df["_ctx"] = "island"

    # Build sites PyRanges with row index for re-join
    sites_pr_df = _sites_to_pyranges(sites).df.copy()
    n           = len(sites)
    sites_pr_df["_row_idx"] = np.arange(n, dtype=np.int32)
    sites_pr    = pr.PyRanges(sites_pr_df)

    # Assign context with priority: island > shore > shelf > open_sea
    cpg_context = np.full(n, "open_sea", dtype=object)

    for ctx_df, ctx_label in [
        (shelf_df,    "shelf"),
        (shore_df,    "shore"),
        (islands_df[["Chromosome", "Start", "End", "_ctx"]], "island"),
    ]:
        ctx_pr  = pr.PyRanges(ctx_df)
        overlap = sites_pr.overlap(ctx_pr)
        if len(overlap.df) == 0:
            continue

        # FIX-5: numpy fancy indexing instead of Python for-loop
        hit_idxs = overlap.df["_row_idx"].to_numpy(dtype=np.int32)
        cpg_context[hit_idxs] = ctx_label

    return sites.with_columns(
        pl.Series("cpg_context", list(cpg_context), dtype=pl.Utf8)
    )