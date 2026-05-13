"""Genomic annotation for DMC / DMR results.

Public API: ``annotate_features`` (gene-feature overlap from a GTF) and
``annotate_cpg_islands`` (island / shore / shelf / open-sea context from a
UCSC CpG-island BED).

The per-chromosome join loop bounds peak memory by the largest single
chromosome rather than the whole genome. GTFs are parsed once per process
and cached in a bounded LRU (``_GTF_CACHE``, default 2 slots; override via
``EPYKIT_GTF_CACHE_SIZE`` or :func:`set_gtf_cache_size`) keyed on the
canonical file path. Interval overlaps go through bioframe, which is pure
Python (pandas + numpy) and avoids the C-extension install pain that comes
with pyranges/ncls/sorted_nearest.
"""

from __future__ import annotations

import gc
import gzip
import logging
import os
import re
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

_FEATURE_PRIORITY: dict[str, int] = {
    "promoter":   0,
    "exon":       1,
    "intron":     2,
    "intergenic": 3,
}

_FEAT_COLS = ["Chromosome", "Start", "End", "Strand", "Feature", "gene_id", "gene_name"]

# GTF cache: bounded LRU of {path -> (genes_pd, exons_pd)}.
#
# A parsed human GTF can be ~1.5 GB resident in pandas form. The default of 2
# slots is enough to keep two genomes hot (e.g. annotating DMC + DMR against
# the same GTF, or comparing mouse + human in one session) without unbounded
# growth in long-running notebooks. Override via the ``EPYKIT_GTF_CACHE_SIZE``
# env var or :func:`set_gtf_cache_size`.
_GTF_CACHE_MAX_SIZE: int = max(1, int(os.environ.get("EPYKIT_GTF_CACHE_SIZE", "2")))
_GTF_CACHE: "OrderedDict[str, tuple[Any, Any]]" = OrderedDict()


def set_gtf_cache_size(max_size: int) -> None:
    """Set the maximum number of parsed GTFs held in memory.

    The cache is keyed by canonical file path; one slot per distinct GTF.
    Decreasing the size evicts the least-recently-used entries immediately.
    """
    global _GTF_CACHE_MAX_SIZE
    if max_size < 1:
        raise ValueError(f"max_size must be >= 1, got {max_size}")
    _GTF_CACHE_MAX_SIZE = int(max_size)
    while len(_GTF_CACHE) > _GTF_CACHE_MAX_SIZE:
        evicted, _ = _GTF_CACHE.popitem(last=False)
        logger.debug("[annotate] GTF cache evicted (resize): %s", evicted)


def _gtf_cache_get(key: str) -> tuple[Any, Any] | None:
    val = _GTF_CACHE.get(key)
    if val is not None:
        _GTF_CACHE.move_to_end(key)
    return val


def _gtf_cache_put(key: str, value: tuple[Any, Any]) -> None:
    if key in _GTF_CACHE:
        _GTF_CACHE.move_to_end(key)
        _GTF_CACHE[key] = value
        return
    _GTF_CACHE[key] = value
    while len(_GTF_CACHE) > _GTF_CACHE_MAX_SIZE:
        evicted, _ = _GTF_CACHE.popitem(last=False)
        logger.debug("[annotate] GTF cache evicted (LRU): %s", evicted)


def _log(msg: str) -> None:
    """Debug-level annotation log (silent at default INFO)."""
    logger.debug("[annotate] %s", msg)


def _df_info(name: str, df) -> str:
    try:
        return f"{name}: {len(df):,} rows"
    except Exception:
        return f"{name}: (unknown shape)"


def _sites_to_df(sites: pl.DataFrame) -> "pd.DataFrame":
    """Build a pandas DataFrame of single-base (or explicit-range) site intervals
    using the Capitalized column convention (``Chromosome``/``Start``/``End``)
    shared with the GTF-derived feature DataFrames downstream.
    """
    import pandas as pd
    if "start" in sites.columns and "end" in sites.columns:
        return pd.DataFrame({
            "Chromosome": sites["chrom"].to_list(),
            "Start":      sites["start"].to_list(),
            "End":        sites["end"].to_list(),
        })
    pos = sites["pos"].to_list()
    return pd.DataFrame({
        "Chromosome": sites["chrom"].to_list(),
        "Start":      pos,
        "End":        [p + 1 for p in pos],
    })


def _build_promoter_df(genes_pd, upstream_bp: int, downstream_bp: int) -> "pd.DataFrame":
    import pandas as pd
    plus  = genes_pd[genes_pd["Strand"] == "+"].copy()
    minus = genes_pd[genes_pd["Strand"] == "-"].copy()
    plus["End"]    = plus["Start"] + downstream_bp
    plus["Start"]  = (plus["Start"] - upstream_bp).clip(lower=0)
    tss_minus      = minus["End"].copy()
    minus["Start"] = (tss_minus - downstream_bp).clip(lower=0)
    minus["End"]   = tss_minus + upstream_bp
    combined = pd.concat([plus, minus], ignore_index=True)
    combined["Feature"] = "promoter"
    return combined


def _build_intron_df(exons_pd, genes_pd) -> "pd.DataFrame":
    import pandas as pd
    if len(exons_pd) == 0 or len(genes_pd) == 0:
        return pd.DataFrame(columns=_FEAT_COLS)
    gene_meta = (
        genes_pd
        .drop_duplicates("gene_id")
        .set_index("gene_id")[["Chromosome", "Start", "End", "Strand", "gene_name"]]
        .rename(columns={"Chromosome": "_g_chrom", "Start": "_g_start", "End": "_g_end"})
    )
    ex = exons_pd[["gene_id", "Start", "End"]].join(gene_meta, on="gene_id", how="inner").copy()
    ex["Start"] = ex[["Start", "_g_start"]].max(axis=1).astype(np.int64)
    ex["End"]   = ex[["End",   "_g_end"]  ].min(axis=1).astype(np.int64)
    ex = ex[ex["Start"] < ex["End"]].copy()
    if len(ex) == 0:
        return pd.DataFrame(columns=_FEAT_COLS)
    ex = ex.sort_values(["gene_id", "Start"]).reset_index(drop=True)
    ex["_prev_end"] = ex.groupby("gene_id", sort=False)["End"].shift(1)
    first_mask = ex["_prev_end"].isna()
    ex.loc[first_mask, "_prev_end"] = ex.loc[first_mask, "_g_start"]
    ex["_prev_end"] = ex["_prev_end"].astype(np.int64)
    introns = ex[ex["_prev_end"] < ex["Start"]].copy()
    if len(introns) == 0:
        return pd.DataFrame(columns=_FEAT_COLS)
    introns["_intron_end"] = introns["Start"]
    introns["Start"]       = introns["_prev_end"]
    introns["End"]         = introns["_intron_end"]
    introns["Feature"]     = "intron"
    introns                = introns.rename(columns={"_g_chrom": "Chromosome"})
    return introns[_FEAT_COLS].reset_index(drop=True)


def _parse_gtf_streaming(gtf_path: str) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Stream-parse a GTF file, extracting only gene and exon rows.

    Results are cached in _GTF_CACHE keyed by canonical path so that
    repeated calls (e.g. annotating DMC then DMR) pay the I/O cost once.
    """
    import pandas as pd

    cache_key = str(Path(gtf_path).resolve())
    cached = _gtf_cache_get(cache_key)
    if cached is not None:
        _log(f"  GTF cache hit for {cache_key}")
        return cached

    gene_rows: list[dict] = []
    exon_rows: list[dict] = []
    attr_re = re.compile(r'(\w+)\s+"([^"]+)"')

    is_gzip = gtf_path.endswith('.gz')
    open_fn = gzip.open if is_gzip else open

    lines_read = 0
    try:
        with open_fn(gtf_path, 'rt') as f:
            for line in f:
                lines_read += 1
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 9:
                    continue
                chrom   = parts[0]
                feature = parts[2]
                if feature not in ('gene', 'exon'):
                    continue
                # GTF 1-based closed → 0-based half-open: subtract
                # 1 from start; end already correct.
                start  = int(parts[3]) - 1
                end    = int(parts[4])
                strand = parts[6]
                attrs  = {}
                for m in attr_re.finditer(parts[8]):
                    attrs[m.group(1)] = m.group(2)
                gene_id   = attrs.get('gene_id', '')
                gene_name = attrs.get('gene_name', attrs.get('gene_id', ''))
                row = {
                    'Chromosome': chrom, 'Start': start, 'End': end,
                    'Strand': strand, 'gene_id': gene_id, 'gene_name': gene_name,
                }
                if feature == 'gene':
                    gene_rows.append(row)
                else:
                    exon_rows.append(row)
    except Exception as e:
        _log(f"  ERROR parsing GTF (read {lines_read:,} lines): {e}")
        raise

    _log(f"  GTF streaming complete: {lines_read:,} lines read")
    _log(f"  Extracted {len(gene_rows):,} gene rows, {len(exon_rows):,} exon rows")

    _empty_cols = ['Chromosome', 'Start', 'End', 'Strand', 'gene_id', 'gene_name']
    genes_pd = pd.DataFrame(gene_rows) if gene_rows else pd.DataFrame(columns=_empty_cols)
    exons_pd = pd.DataFrame(exon_rows) if exon_rows else pd.DataFrame(columns=_empty_cols)

    result = (genes_pd, exons_pd)
    _gtf_cache_put(cache_key, result)
    return result


def _pick_best_overlap(joined_df) -> "pd.DataFrame":
    df = joined_df.copy()
    feat_col = "Feature_b" if "Feature_b" in df.columns else "Feature"
    df["_priority"] = df[feat_col].map(_FEATURE_PRIORITY).fillna(99)
    return (
        df.sort_values("_priority")
          .groupby("_row_idx", as_index=False)
          .first()
          .drop(columns=["_priority"])
    )


def _annotate_chromosome_chunk(
    chrom: str,
    chrom_sites: pl.DataFrame,
    chrom_features_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """Run overlap + best-pick for one chromosome. Returns pandas DataFrame."""
    import bioframe
    import pandas as pd

    COLS = ("Chromosome", "Start", "End")

    chunk_n   = len(chrom_sites)
    orig_idxs = chrom_sites["_orig_idx"].to_numpy()

    result = pd.DataFrame({
        "_orig_idx":    orig_idxs,
        "gene_id":      np.full(chunk_n, "", dtype=object),
        "gene_name":    np.full(chunk_n, "", dtype=object),
        "feature_type": np.full(chunk_n, "intergenic", dtype=object),
    })

    if chrom_features_df.empty:
        _log(f"  {chrom}: no features -> all intergenic")
        return result

    _log(f"  {chrom}: building sites DataFrame ({chunk_n:,} sites)")
    t0 = time.time()
    try:
        sites_pd = _sites_to_df(chrom_sites)
        sites_pd["_row_idx"] = np.arange(chunk_n, dtype=np.int32)
        _log(f"  {chrom}: sites DataFrame built in {time.time()-t0:.1f}s")
    except Exception:
        _log(f"  {chrom}: ERROR building sites DataFrame:\n{traceback.format_exc()}")
        return result

    _log(f"  {chrom}: features DataFrame ({len(chrom_features_df):,} features)")
    feat_df = chrom_features_df

    _log(f"  {chrom}: running overlap ...")
    t0 = time.time()
    try:
        joined = bioframe.overlap(
            sites_pd, feat_df,
            how="left",
            cols1=COLS, cols2=COLS,
            suffixes=("", "_b"),
        )
        join_rows = len(joined)
        _log(f"  {chrom}: overlap done in {time.time()-t0:.1f}s  -> {join_rows:,} rows")
    except Exception:
        _log(f"  {chrom}: ERROR during overlap:\n{traceback.format_exc()}")
        return result

    if join_rows == 0:
        del joined
        gc.collect()
        _log(f"  {chrom}: overlap returned 0 rows -> all intergenic")
        return result

    try:
        joined_df = joined
        del joined
        gc.collect()

        _log(f"  {chrom}: picking best overlaps ...")
        best = _pick_best_overlap(joined_df)
        _log(f"  {chrom}: {_df_info('best', best)}")
        del joined_df
        gc.collect()

        feat_col = "Feature_b"   if "Feature_b"   in best.columns else "Feature"
        gid_col  = "gene_id_b"   if "gene_id_b"   in best.columns else "gene_id"
        gnm_col  = "gene_name_b" if "gene_name_b" in best.columns else "gene_name"
        best_slim = (
            best[["_row_idx", gid_col, gnm_col, feat_col]]
            .rename(columns={
                gid_col:  "gene_id",
                gnm_col:  "gene_name",
                feat_col: "feature_type",
            })
        )
        del best

        local_df = (
            pd.DataFrame({"_row_idx": np.arange(chunk_n, dtype=np.int32)})
            .merge(best_slim, on="_row_idx", how="left")
        )
        result["gene_id"]      = local_df["gene_id"].fillna("").astype(str).replace("-1", "").to_numpy()
        result["gene_name"]    = local_df["gene_name"].fillna("").astype(str).replace("-1", "").to_numpy()
        result["feature_type"] = local_df["feature_type"].fillna("intergenic").astype(str).replace("-1", "intergenic").to_numpy()

        n_annotated = int((result["gene_id"] != "").sum())
        _log(f"  {chrom}: {n_annotated:,}/{chunk_n:,} sites annotated")

    except Exception:
        _log(f"  {chrom}: ERROR during post-join assembly:\n{traceback.format_exc()}")

    return result


# Public API

def annotate_features(
    sites: pl.DataFrame,
    annotation_gtf: str,
    features: list[str] | tuple[str, ...] = ("promoter", "exon", "intron", "intergenic"),
    promoter_upstream_bp: int = 2000,
    promoter_downstream_bp: int = 200,
) -> pl.DataFrame:
    """Annotate DMC / DMR sites with gene-level genomic features.

    The GTF is parsed once per process and cached; subsequent calls with the
    same file path skip the 60-90 s streaming step entirely.
    """
    try:
        import bioframe  # noqa: F401  (presence-check; used inside _annotate_chromosome_chunk)
    except ImportError as exc:
        raise ImportError("bioframe is required. pip install bioframe") from exc

    import pandas as pd

    _log("=" * 60)
    _log("annotate_features START")
    _log(f"  sites input: {_df_info('sites', sites)}")
    _log(f"  GTF: {annotation_gtf}")
    _log(f"  features requested: {list(features)}")
    _log(f"  promoter window: -{promoter_upstream_bp} / +{promoter_downstream_bp}")

    n = len(sites)
    t_total = time.time()

    # ------------------------------------------------------------------
    # Step 1: Parse GTF (uses cache after first call)
    # ------------------------------------------------------------------
    _log("Step 1/8: stream-parsing GTF (gene and exon rows only) ...")
    t0 = time.time()
    try:
        genes_pd, exons_pd = _parse_gtf_streaming(annotation_gtf)
        _log(f"  GTF parsed in {time.time()-t0:.1f}s")
        _log(f"  {_df_info('genes_pd', genes_pd)}")
        _log(f"  {_df_info('exons_pd (raw)', exons_pd)}")
        gc.collect()
        _log("  Intermediate data freed")
    except Exception:
        _log(f"FATAL: error parsing GTF:\n{traceback.format_exc()}")
        raise

    if "gene_id" not in genes_pd.columns:
        raise ValueError("GTF missing 'gene_id' attribute column")
    if "gene_name" not in genes_pd.columns:
        genes_pd["gene_name"] = genes_pd["gene_id"]
    if "gene_name" not in exons_pd.columns:
        exons_pd = exons_pd.merge(
            genes_pd[["gene_id", "gene_name"]].drop_duplicates(),
            on="gene_id", how="left",
        )

    # ------------------------------------------------------------------
    # Step 2: Deduplicate exons
    # ------------------------------------------------------------------
    _log("Step 2/8: deduplicating exons ...")
    _exon_key = ["Chromosome", "Start", "End", "Strand", "gene_id"]
    if all(c in exons_pd.columns for c in _exon_key):
        n_before = len(exons_pd)
        extra = [c for c in ["gene_name"] if c in exons_pd.columns]
        exons_pd = (
            exons_pd[_exon_key + extra]
            .drop_duplicates(subset=["Chromosome", "Start", "End", "gene_id"])
            .reset_index(drop=True)
        )
        gc.collect()
        _log(f"  exons: {n_before:,} -> {len(exons_pd):,} (removed {n_before - len(exons_pd):,} duplicates)")
    else:
        _log("  WARNING: expected exon columns not all present; skipping dedup.")

    # ------------------------------------------------------------------
    # Step 3: Build combined feature DataFrame
    # ------------------------------------------------------------------
    _log("Step 3/8: building feature intervals ...")
    feature_dfs: list[pd.DataFrame] = []

    if "promoter" in features:
        t0 = time.time()
        prom_df = _build_promoter_df(genes_pd, promoter_upstream_bp, promoter_downstream_bp)
        _log(f"  {_df_info('promoters', prom_df)}  ({time.time()-t0:.1f}s)")
        feature_dfs.append(prom_df[_FEAT_COLS])

    if "exon" in features and len(exons_pd) > 0:
        ex = exons_pd[["Chromosome", "Start", "End", "Strand", "gene_id", "gene_name"]].copy()
        ex["Feature"] = "exon"
        _log(f"  {_df_info('exons (feature)', ex)}")
        feature_dfs.append(ex[_FEAT_COLS])

    if "intron" in features and len(exons_pd) > 0 and len(genes_pd) > 0:
        t0 = time.time()
        _log("  building introns (vectorised) ...")
        try:
            intron_df = _build_intron_df(exons_pd, genes_pd)
            _log(f"  {_df_info('introns', intron_df)}  ({time.time()-t0:.1f}s)")
            if len(intron_df) > 0:
                feature_dfs.append(intron_df[_FEAT_COLS])
        except Exception:
            _log(f"  ERROR building introns:\n{traceback.format_exc()}")

    if feature_dfs:
        t0 = time.time()
        all_features_df = (
            pd.concat(feature_dfs, ignore_index=True)
            [_FEAT_COLS]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        _log(f"  {_df_info('all_features_df', all_features_df)}  ({time.time()-t0:.1f}s)")
    else:
        _log("  WARNING: no feature DataFrames built; all sites will be intergenic")
        all_features_df = pd.DataFrame(columns=_FEAT_COLS)

    del feature_dfs
    gc.collect()

    # ------------------------------------------------------------------
    # Step 4: Group features by chromosome
    # ------------------------------------------------------------------
    _log("Step 4/8: grouping features by chromosome ...")
    features_by_chrom: dict[str, pd.DataFrame] = {}
    for chrom_name, grp in all_features_df.groupby("Chromosome", sort=False):
        features_by_chrom[str(chrom_name)] = grp.reset_index(drop=True)
    n_feat_chroms = len(features_by_chrom)
    _log(f"  features grouped across {n_feat_chroms} chromosomes")
    for c, df in sorted(features_by_chrom.items()):
        _log(f"    {c}: {len(df):,} feature intervals")
    del all_features_df
    gc.collect()

    # ------------------------------------------------------------------
    # Step 5: Build TSS map
    # ------------------------------------------------------------------
    _log("Step 5/8: building TSS map ...")
    _g = genes_pd[["gene_id", "Start", "End", "Strand"]].drop_duplicates("gene_id")
    tss_values = np.where(
        _g["Strand"].to_numpy() != "-",
        _g["Start"].to_numpy(),
        _g["End"].to_numpy(),
    ).astype(np.int64)
    tss_series = pd.Series(tss_values, index=_g["gene_id"].to_numpy(), dtype="Int64")
    _log(f"  TSS map built: {len(tss_series):,} genes")
    del _g

    # ------------------------------------------------------------------
    # Step 6: Tag sites with original row index
    # ------------------------------------------------------------------
    _log("Step 6/8: tagging sites with original row index ...")
    sites_with_idx = sites.with_columns(
        pl.Series("_orig_idx", np.arange(n, dtype=np.int32))
    )
    chromosomes = sorted(sites["chrom"].unique().to_list())
    _log(f"  {n:,} sites across {len(chromosomes)} chromosomes: {chromosomes}")

    # ------------------------------------------------------------------
    # Step 7: Per-chromosome annotation loop
    # ------------------------------------------------------------------
    _log("Step 7/8: per-chromosome annotation loop ...")
    annot_parts: list[pd.DataFrame] = []

    for i, chrom in enumerate(chromosomes, 1):
        chrom_sites    = sites_with_idx.filter(pl.col("chrom") == chrom)
        chunk_n        = len(chrom_sites)
        chrom_features = features_by_chrom.get(chrom, pd.DataFrame(columns=_FEAT_COLS))

        _log(f"[{i}/{len(chromosomes)}] {chrom}: {chunk_n:,} sites, "
             f"{len(chrom_features):,} features")

        if chunk_n == 0:
            _log(f"  {chrom}: 0 sites, skipping")
            continue

        t0 = time.time()
        try:
            part = _annotate_chromosome_chunk(chrom, chrom_sites, chrom_features)
            annot_parts.append(part)
            _log(f"  {chrom}: done in {time.time()-t0:.1f}s")
        except Exception:
            _log(f"  {chrom}: UNHANDLED ERROR:\n{traceback.format_exc()}")
            part = pd.DataFrame({
                "_orig_idx":    chrom_sites["_orig_idx"].to_numpy(),
                "gene_id":      np.full(chunk_n, "", dtype=object),
                "gene_name":    np.full(chunk_n, "", dtype=object),
                "feature_type": np.full(chunk_n, "intergenic", dtype=object),
            })
            annot_parts.append(part)

        gc.collect()

    # ------------------------------------------------------------------
    # Step 8: Reassemble + TSS distance
    # ------------------------------------------------------------------
    _log("Step 8/8: reassembling results ...")
    if annot_parts:
        annot_all = (
            pd.concat(annot_parts, ignore_index=True)
            .sort_values("_orig_idx")
            .reset_index(drop=True)
        )
    else:
        _log("  WARNING: annot_parts is empty — returning all-intergenic")
        annot_all = pd.DataFrame({
            "_orig_idx":    np.arange(n, dtype=np.int32),
            "gene_id":      np.full(n, "", dtype=object),
            "gene_name":    np.full(n, "", dtype=object),
            "feature_type": np.full(n, "intergenic", dtype=object),
        })

    _log(f"  {_df_info('annot_all (reassembled)', annot_all)}")

    gene_ids      = annot_all["gene_id"].to_numpy(dtype=object)
    gene_names    = annot_all["gene_name"].to_numpy(dtype=object)
    feature_types = annot_all["feature_type"].to_numpy(dtype=object)

    n_annotated = int((gene_ids != "").sum())
    ft_counts   = {k: int((feature_types == k).sum()) for k in _FEATURE_PRIORITY}
    ft_counts["intergenic"] = int((feature_types == "intergenic").sum())
    _log(f"  annotation summary: {n_annotated:,}/{n:,} sites have a gene  | {ft_counts}")

    if "pos" in sites.columns:
        site_mids = sites["pos"].to_numpy().astype(np.float64)
    else:
        site_mids = (
            (sites["start"].to_numpy() + sites["end"].to_numpy()) / 2.0
        ).astype(np.float64)

    tss_positions = (
        pd.Series(gene_ids.tolist())
        .map(tss_series)
        .to_numpy(dtype=np.float64, na_value=np.nan)
    )

    # TSS distance: positive = downstream. On - strand, TSS sits at End and a
    # higher genomic coordinate is upstream, so flip the sign.
    _strand_lut = (
        genes_pd[["gene_id", "Strand"]]
        .drop_duplicates("gene_id")
        .set_index("gene_id")["Strand"]
    )
    strand_arr  = pd.Series(gene_ids.tolist()).map(_strand_lut).to_numpy(dtype=object)
    strand_sign = np.where(strand_arr == "-", -1.0, 1.0).astype(np.float64)
    dist_to_tss              = (strand_sign * (site_mids - tss_positions)).astype(np.float32)
    dist_to_tss[gene_ids == ""] = np.nan

    _log(f"annotate_features DONE  total elapsed {time.time()-t_total:.1f}s")
    _log("=" * 60)

    return sites.with_columns([
        pl.Series("gene_id",         gene_ids.tolist(),      dtype=pl.Utf8),
        pl.Series("gene_name",       gene_names.tolist(),    dtype=pl.Utf8),
        pl.Series("feature_type",    feature_types.tolist(), dtype=pl.Utf8),
        pl.Series("distance_to_tss", dist_to_tss,            dtype=pl.Float32),
    ])


def annotate_cpg_islands(
    sites: pl.DataFrame,
    cpg_island_bed: str,
) -> pl.DataFrame:
    """Classify each CpG site by CpG-island context."""
    _log("=" * 60)
    _log("annotate_cpg_islands START")
    _log(f"  sites: {_df_info('sites', sites)}")
    _log(f"  BED: {cpg_island_bed}")

    if len(sites) == 0:
        _log("  sites is empty — returning early with no cpg_context column")
        return sites

    try:
        import bioframe
    except ImportError as exc:
        raise ImportError("bioframe is required. pip install bioframe") from exc

    import pandas as pd

    COLS = ("Chromosome", "Start", "End")

    t_total = time.time()
    n = len(sites)

    _log("Step 1/3: loading BED ...")
    try:
        t0 = time.time()
        islands_df = bioframe.read_table(
            cpg_island_bed, schema="bed3", usecols=[0, 1, 2]
        ).rename(
            columns={"chrom": "Chromosome", "start": "Start", "end": "End"}
        )
        _log(f"  BED loaded in {time.time()-t0:.1f}s: {len(islands_df):,} islands")
    except Exception:
        _log(f"FATAL: error loading BED:\n{traceback.format_exc()}")
        raise

    if len(islands_df) == 0:
        _log("  WARNING: BED is empty -> all sites open_sea")
        return sites.with_columns(pl.lit("open_sea").alias("cpg_context"))

    SHORE_DIST = 2_000
    SHELF_DIST = 4_000

    def _flanks(df: pd.DataFrame, inner: int, outer: int, label: str) -> pd.DataFrame:
        up = df[["Chromosome", "Start", "End"]].copy()
        up["End"]   = (up["Start"] - inner).clip(lower=0)
        up["Start"] = (up["Start"] - outer).clip(lower=0)
        up["_ctx"]  = label
        dn = df[["Chromosome", "Start", "End"]].copy()
        dn["Start"] = dn["End"] + inner
        dn["End"]   = dn["End"] + outer
        dn["_ctx"]  = label
        return pd.concat([up, dn], ignore_index=True)

    shore_df           = _flanks(islands_df, 0,          SHORE_DIST, "shore")
    shelf_df           = _flanks(islands_df, SHORE_DIST, SHELF_DIST, "shelf")
    islands_df["_ctx"] = "island"
    _log(f"  flanks built: {len(shore_df):,} shore intervals, {len(shelf_df):,} shelf intervals")

    _log("Step 2/3: building sites DataFrame ...")
    try:
        sites_pr_df = _sites_to_df(sites)
        sites_pr_df["_row_idx"] = np.arange(n, dtype=np.int32)
        _log(f"  sites DataFrame: {n:,} rows")
    except Exception:
        _log(f"FATAL: error building sites DataFrame:\n{traceback.format_exc()}")
        raise

    cpg_context = np.full(n, "open_sea", dtype=object)

    _log("Step 3/3: overlapping shelf / shore / island ...")
    for ctx_df, ctx_label in [
        (shelf_df,                                           "shelf"),
        (shore_df,                                           "shore"),
        (islands_df[["Chromosome", "Start", "End", "_ctx"]], "island"),
    ]:
        t0 = time.time()
        try:
            overlap = bioframe.overlap(
                sites_pr_df, ctx_df,
                how="inner",
                cols1=COLS, cols2=COLS,
                suffixes=("", "_b"),
            )
            n_hits  = len(overlap)
            _log(f"  {ctx_label}: {n_hits:,} hits in {time.time()-t0:.1f}s")
            if n_hits == 0:
                continue
            hit_idxs = overlap["_row_idx"].drop_duplicates().to_numpy(dtype=np.int32)
            cpg_context[hit_idxs] = ctx_label
        except Exception:
            _log(f"  ERROR during {ctx_label} overlap:\n{traceback.format_exc()}")

    counts = {lbl: int((cpg_context == lbl).sum())
              for lbl in ["island", "shore", "shelf", "open_sea"]}
    _log(f"  context summary: {counts}")
    _log(f"annotate_cpg_islands DONE  total elapsed {time.time()-t_total:.1f}s")
    _log("=" * 60)

    return sites.with_columns(
        pl.Series("cpg_context", list(cpg_context), dtype=pl.Utf8)
    )