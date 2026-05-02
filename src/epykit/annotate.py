"""Genomic annotation for DMC / DMR results.

Phase 3 of the epykit pipeline:
  - annotate_features: overlaps sites/regions with gene-level features from a
    GTF/GFF3 (promoter, exon, intron, intergenic) and returns the nearest
    gene with distance-to-TSS.
  - annotate_cpg_islands: classifies each site as island / shore / shelf /
    open_sea using a UCSC CpGIsland BED track.

Both functions accept either DMC output (chrom, pos) or DMR output
(chrom, start, end) and return the input DataFrame with additional annotation
columns appended.

Dependencies
------------
pyranges  — already a project dependency (used in filter.py).
            This module targets the 0.x API (PyRanges constructor, .join(),
            .overlap(), .df accessor).

GTF note
--------
Pyranges read_gtf() exposes Ensembl/UCSC attribute columns (gene_id,
gene_name, transcript_id …) when they are present in the source file.  If
your GTF omits gene_name, the column will be absent and annotate_features
will fill gene_name with gene_id instead.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Hierarchical priority for overlapping features (lower index = higher priority)
_FEATURE_PRIORITY: dict[str, int] = {
    "promoter":    0,
    "exon":        1,
    "intron":      2,
    "intergenic":  3,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sites_to_pyranges(sites: pl.DataFrame) -> "pr.PyRanges":
    """Convert a DMC (chrom/pos) or DMR (chrom/start/end) DataFrame to
    a PyRanges object with a 1-based ``site_index`` for later re-join.
    """
    import pyranges as pr

    if "start" in sites.columns and "end" in sites.columns:
        # DMR mode
        return pr.PyRanges(
            chromosomes=sites["chrom"].to_list(),
            starts=sites["start"].to_list(),
            ends=sites["end"].to_list(),
        )
    else:
        # DMC mode: single-base intervals
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
    """Construct promoter intervals from a genes pandas DataFrame.

    Promoter is defined as [TSS - upstream_bp, TSS + downstream_bp).
    Strand-aware: TSS = Start for + strand, End for − strand.
    """
    import pandas as pd

    plus  = genes_pd[genes_pd["Strand"] == "+"].copy()
    minus = genes_pd[genes_pd["Strand"] == "-"].copy()

    # + strand: TSS = Start
    plus["End"]   = plus["Start"] + downstream_bp
    plus["Start"] = (plus["Start"] - upstream_bp).clip(lower=0)

    # − strand: TSS = End
    tss_minus     = minus["End"].copy()
    minus["Start"] = (tss_minus - downstream_bp).clip(lower=0)
    minus["End"]   = tss_minus + upstream_bp

    combined = pd.concat([plus, minus], ignore_index=True)
    combined["Feature"] = "promoter"
    return combined


def _build_intron_df(exons_pd, genes_pd) -> "pd.DataFrame":
    """Derive intronic intervals as gene body minus exon intervals.

    Uses a simple per-gene subtraction: for each gene, the gene span minus
    the union of its exons gives the intron(s).  This is approximate for
    alternatively spliced genes but sufficient for annotation purposes.
    """
    import pandas as pd

    records = []
    grouped = exons_pd.groupby("gene_id", group_keys=False)

    gene_lookup = genes_pd.set_index("gene_id")

    for gene_id, exon_group in grouped:
        if gene_id not in gene_lookup.index:
            continue
        gene_row = gene_lookup.loc[gene_id]
        gene_start = int(gene_row["Start"])
        gene_end   = int(gene_row["End"])
        chrom      = gene_row["Chromosome"]
        strand     = gene_row.get("Strand", "*")
        gene_name  = gene_row.get("gene_name", gene_id)

        # Exon intervals, clipped to gene boundaries
        exon_ivs = sorted([
            (max(int(r["Start"]), gene_start),
             min(int(r["End"]),   gene_end))
            for _, r in exon_group.iterrows()
        ])

        # Intron = gaps between consecutive exon intervals within the gene
        prev_end = gene_start
        for ex_start, ex_end in exon_ivs:
            if ex_start > prev_end:
                records.append({
                    "Chromosome": chrom,
                    "Start":      prev_end,
                    "End":        ex_start,
                    "Strand":     strand,
                    "Feature":    "intron",
                    "gene_id":    gene_id,
                    "gene_name":  gene_name,
                })
            prev_end = max(prev_end, ex_end)

    if not records:
        return pd.DataFrame(
            columns=["Chromosome", "Start", "End", "Strand",
                     "Feature", "gene_id", "gene_name"]
        )
    return pd.DataFrame(records)


def _pick_best_overlap(joined_df) -> "pd.DataFrame":
    """When a site overlaps multiple features, keep only the highest-priority one."""
    import pandas as pd

    df = joined_df.copy()
    df["_priority"] = df["Feature_b"].map(_FEATURE_PRIORITY).fillna(99)
    df = (
        df.sort_values("_priority")
        .groupby(["Chromosome", "Start"], as_index=False)
        .first()
        .drop(columns=["_priority"])
    )
    return df


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
        distance_to_tss (Int32, 0 for promoter / genic, NaN for intergenic).
    """
    try:
        import pyranges as pr
    except ImportError as exc:
        raise ImportError(
            "pyranges is required for annotation. "
            "Install with: pip install pyranges"
        ) from exc

    logger.info("annotate_features: loading %s", annotation_gtf)
    gtf = pr.read_gtf(annotation_gtf, as_df=False)

    genes_pd  = gtf[gtf.Feature == "gene"].df
    exons_pd  = gtf[gtf.Feature == "exon"].df

    # Ensure gene_id column exists
    if "gene_id" not in genes_pd.columns:
        raise ValueError("GTF is missing the 'gene_id' attribute column")

    # gene_name is optional; fall back to gene_id
    if "gene_name" not in genes_pd.columns:
        genes_pd["gene_name"] = genes_pd["gene_id"]
    if "gene_name" not in exons_pd.columns:
        exons_pd = exons_pd.merge(
            genes_pd[["gene_id", "gene_name"]].drop_duplicates(),
            on="gene_id", how="left",
        )

    import pandas as pd

    feature_dfs: list[pd.DataFrame] = []

    if "promoter" in features:
        promoter_df = _build_promoter_df(
            genes_pd, promoter_upstream_bp, promoter_downstream_bp
        )
        feature_dfs.append(
            promoter_df[["Chromosome", "Start", "End", "Strand",
                          "Feature", "gene_id", "gene_name"]]
        )

    if "exon" in features and len(exons_pd) > 0:
        ex = exons_pd[["Chromosome", "Start", "End", "Strand",
                        "gene_id", "gene_name"]].copy()
        ex["Feature"] = "exon"
        feature_dfs.append(ex)

    if "intron" in features and len(exons_pd) > 0 and len(genes_pd) > 0:
        intron_df = _build_intron_df(exons_pd, genes_pd)
        if len(intron_df) > 0:
            feature_dfs.append(intron_df)

    # Build sites PyRanges
    sites_pr = _sites_to_pyranges(sites)

    # --- Overlap annotation ---
    # Temporarily attach a row index so we can re-join after overlap
    sites_pd = sites_pr.df.copy()
    sites_pd["_row_idx"] = np.arange(len(sites_pd), dtype=np.int32)
    sites_pr2 = pr.PyRanges(sites_pd)

    annotation_hits: dict[int, dict] = {}  # row_idx → best annotation

    if feature_dfs:
        all_features_df = pd.concat(feature_dfs, ignore_index=True)
        features_pr = pr.PyRanges(all_features_df)
        joined = sites_pr2.join(features_pr, how="left", suffix="_b")

        if len(joined.df) > 0:
            joined_df = joined.df.copy()
            best = _pick_best_overlap(joined_df)
            for _, row in best.iterrows():
                idx = int(row["_row_idx"])
                annotation_hits[idx] = {
                    "gene_id":      row.get("gene_id", None),
                    "gene_name":    row.get("gene_name", None),
                    "feature_type": row.get("Feature_b", "intergenic"),
                }

    # --- Compute distance to TSS ---
    # TSS lookup: gene_id → (chrom, tss_pos)
    tss_lookup: dict[str, tuple[str, int]] = {}
    for _, row in genes_pd.iterrows():
        strand = row.get("Strand", "+")
        tss    = int(row["Start"]) if strand != "-" else int(row["End"])
        tss_lookup[row["gene_id"]] = (row["Chromosome"], tss)

    # Build result columns
    n = len(sites)
    gene_ids      = [""] * n
    gene_names    = [""] * n
    feature_types = ["intergenic"] * n
    dist_to_tss   = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        hit = annotation_hits.get(i)
        if hit and hit["gene_id"]:
            gene_ids[i]      = hit["gene_id"] or ""
            gene_names[i]    = hit["gene_name"] or hit["gene_id"] or ""
            feature_types[i] = hit["feature_type"] or "intergenic"

            gid = hit["gene_id"]
            if gid in tss_lookup:
                chrom_tss, tss_pos = tss_lookup[gid]
                # Site midpoint
                if "pos" in sites.columns:
                    site_mid = int(sites["pos"][i])
                else:
                    site_mid = int((sites["start"][i] + sites["end"][i]) / 2)
                dist_to_tss[i] = float(site_mid - tss_pos)

    return sites.with_columns([
        pl.Series("gene_id",       gene_ids,      dtype=pl.Utf8),
        pl.Series("gene_name",     gene_names,    dtype=pl.Utf8),
        pl.Series("feature_type",  feature_types, dtype=pl.Utf8),
        pl.Series("distance_to_tss",
                  dist_to_tss.astype(np.float32),
                  dtype=pl.Float32),
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
        Path to a BED file with CpG island coordinates.  The UCSC
        CpGIsland track is compatible (columns: chrom, start, end, …).

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
        return sites.with_columns(
            pl.lit("open_sea").alias("cpg_context")
        )

    # Build ± extension intervals for shore and shelf
    SHORE_DIST = 2_000
    SHELF_DIST = 4_000

    islands_df = islands.df.copy()

    def _flanks(df: pd.DataFrame, inner: int, outer: int,
                label: str) -> pd.DataFrame:
        """Create up- and down-stream flanking intervals [TSS-outer, TSS-inner)."""
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

    # Build sites PyRanges with row index
    sites_pr_df = _sites_to_pyranges(sites).df.copy()
    sites_pr_df["_row_idx"] = np.arange(len(sites_pr_df), dtype=np.int32)
    sites_pr = pr.PyRanges(sites_pr_df)

    # Assign context with priority: island > shore > shelf > open_sea
    cpg_context = np.full(len(sites), "open_sea", dtype=object)

    for ctx_df, ctx_label in [
        (shelf_df,    "shelf"),
        (shore_df,    "shore"),
        (islands_df[["Chromosome", "Start", "End", "_ctx"]], "island"),
    ]:
        ctx_pr = pr.PyRanges(ctx_df)
        overlap = sites_pr.overlap(ctx_pr)
        if len(overlap.df) == 0:
            continue
        hit_idxs = overlap.df["_row_idx"].tolist()
        for idx in hit_idxs:
            cpg_context[idx] = ctx_label

    return sites.with_columns(
        pl.Series("cpg_context", list(cpg_context), dtype=pl.Utf8)
    )
