"""QC and filtering for Parquet methylation stores.

This module implements streaming, memory-efficient filtering of methylation data
stored in partitioned Parquet datasets. Key operations:
  - Sample-level summary statistics (n_CpGs, coverage, methylation level)
  - Two-pass filtering on coverage quantiles
  - Optional blacklist filtering via BED file
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import polars as pl
import numpy as np


def sample_summary(
    methylstore_path: str,
    sample: str,
    output_path: Optional[str] = None,
) -> pl.DataFrame:
    """Compute per-chromosome summary statistics for a single sample.

    Parameters
    ----------
    methylstore_path : str
        Path to the partitioned Parquet methylstore (root containing sample=*/chrom=*/)
    sample : str
        Sample identifier (exact match)
    output_path : str, optional
        If provided, write summary as Parquet; default saves to stdout

    Returns
    -------
    pl.DataFrame
        Columns: chrom, n_CpGs, mean_coverage, median_coverage, global_methylation

    Notes
    -----
    Uses lazy scanning and streaming collect to avoid loading full dataset into memory.
    """
    # Scan all Parquet files for this sample
    glob_pattern = f"{methylstore_path}/sample={sample}/**/part-*.parquet"
    lf = pl.scan_parquet(glob_pattern)

    # Group by chromosome and compute stats
    stats = (
        lf.group_by("chrom")
        .agg(
            [
                pl.count().alias("n_CpGs"),
                pl.mean("coverage").alias("mean_coverage"),
                pl.median("coverage").alias("median_coverage"),
                (pl.sum("N_meth") / pl.sum("coverage")).alias("global_methylation"),
            ]
        )
        .sort("chrom")
        .collect()
    )

    if output_path:
        stats.write_parquet(output_path)
    return stats


def get_coverage_quantile(
    methylstore_path: str,
    sample: str,
    quantile: float = 0.999,
) -> int:
    """Compute per-sample coverage quantile (streaming).

    Parameters
    ----------
    methylstore_path : str
        Path to the partitioned Parquet methylstore
    sample : str
        Sample identifier
    quantile : float
        Quantile to compute (0.0 to 1.0); default 0.999

    Returns
    -------
    int
        Coverage value at specified quantile
    """
    glob_pattern = f"{methylstore_path}/sample={sample}/**/part-*.parquet"
    lf = pl.scan_parquet(glob_pattern)
    result = (
        lf.select(pl.col("coverage").quantile(quantile))
        .collect()
        .item()
    )
    return int(result)


def filter_sites(
    methylstore_path: str,
    output_dir: str,
    min_coverage: int = 10,
    max_coverage_quantile: float = 0.999,
    blacklist_bed: Optional[str] = None,
    sample: Optional[str] = None,
) -> None:
    """Filter low-quality CpG sites from a Parquet methylstore.

    Applies two-pass filtering:
      1. Compute per-sample max_coverage from quantile
      2. Filter sites where coverage < min_coverage or > max_coverage
      3. (Optional) Remove sites in blacklisted regions from BED file

    Parameters
    ----------
    methylstore_path : str
        Path to input Parquet methylstore
    output_dir : str
        Path to write filtered Parquet store (same structure as input)
    min_coverage : int
        Minimum coverage threshold (default 10)
    max_coverage_quantile : float
        Quantile for max coverage (default 0.999, approximately 99.9th percentile)
    blacklist_bed : str, optional
        Path to BED file with regions to exclude (chrom, start, end format)
    sample : str, optional
        If provided, filter only this sample; else filter all samples in store

    Returns
    -------
    None
        Writes filtered Parquet store to output_dir
    """
    methylstore_path = Path(methylstore_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all samples in methylstore
    sample_dirs = list(methylstore_path.glob("sample=*"))
    if not sample_dirs:
        raise ValueError(f"No sample=* directories found in {methylstore_path}")

    samples_to_filter = []
    if sample:
        sample_path = methylstore_path / f"sample={sample}"
        if not sample_path.exists():
            raise ValueError(f"Sample {sample} not found in {methylstore_path}")
        samples_to_filter = [sample]
    else:
        samples_to_filter = [d.name.replace("sample=", "") for d in sample_dirs]

    # Optional: load blacklist BED file
    blacklist_ranges = None
    if blacklist_bed:
        blacklist_df = pl.read_csv(
            blacklist_bed,
            separator="\t",
            has_header=False,
            new_columns=["chrom", "start", "end"],
        )
        blacklist_ranges = {
            chrom: blacklist_df.filter(pl.col("chrom") == chrom)
            for chrom in blacklist_df["chrom"].unique()
        }

    # Process each sample
    for samp in samples_to_filter:
        print(f"Filtering sample {samp}...")

        # Step 1: Compute max coverage quantile for this sample
        max_cov = get_coverage_quantile(
            str(methylstore_path), samp, quantile=max_coverage_quantile
        )
        print(f"  Max coverage quantile ({max_coverage_quantile}): {max_cov}")

        # Step 2: Read all data for this sample, apply coverage filter
        glob_pattern = f"{methylstore_path}/sample={samp}/**/part-*.parquet"
        lf = pl.scan_parquet(glob_pattern)

        # Filter by coverage
        filtered_lf = lf.filter(
            (pl.col("coverage") >= min_coverage) & (pl.col("coverage") <= max_cov)
        )

        # Optional: apply blacklist filter
        if blacklist_ranges:
            # For each position, check if it falls within any blacklist range for its chrom
            # This is done efficiently by filtering: keep only sites NOT in blacklist
            for chrom, bl_df in blacklist_ranges.items():
                if len(bl_df) > 0:
                    # For this chromosome, create a mask of sites to exclude
                    # We do this by checking if pos falls in [start, end) for any row
                    bl_positions = []
                    for row in bl_df.iter_rows(named=True):
                        bl_positions.append((row["start"], row["end"]))
                    # Filter: keep rows where (chrom != chrom) OR (pos < min_start or pos >= max_end)
                    # Simpler: only exclude if pos in [start, end) for blacklist regions
                    for start, end in bl_positions:
                        filtered_lf = filtered_lf.filter(
                            ~(
                                (pl.col("chrom") == chrom)
                                & (pl.col("pos") >= start)
                                & (pl.col("pos") < end)
                            )
                        )

        # Step 3: Write to output directory with same partition structure
        out_sample_dir = output_dir / f"sample={samp}"
        out_sample_dir.mkdir(parents=True, exist_ok=True)

        # Collect and partition by chromosome manually
        df_filtered = filtered_lf.collect()
        for chrom in df_filtered["chrom"].unique().to_list():
            chrom_data = df_filtered.filter(pl.col("chrom") == chrom)
            chrom_dir = out_sample_dir / f"chrom={chrom}"
            chrom_dir.mkdir(parents=True, exist_ok=True)
            out_path = chrom_dir / "part-0.parquet"
            chrom_data.write_parquet(str(out_path), compression="zstd")

    print(f"Filtered Parquet store written to {output_dir}")


def intersect_sites(
    methylstore_path: str,
    samples: list[str],
    output_path: Optional[str] = None,
) -> pl.DataFrame:
    """Find CpG sites present in all specified samples.

    Parameters
    ----------
    methylstore_path : str
        Path to Parquet methylstore
    samples : list[str]
        List of sample identifiers to intersect
    output_path : str, optional
        If provided, write list of (chrom, pos, strand) to Parquet

    Returns
    -------
    pl.DataFrame
        Columns: chrom, pos, strand (sites present in all samples)
    """
    if not samples:
        raise ValueError("Must provide at least one sample")

    # For each sample, collect unique (chrom, pos, strand) tuples
    site_sets = []
    for samp in samples:
        glob_pattern = f"{methylstore_path}/sample={samp}/**/part-*.parquet"
        lf = pl.scan_parquet(glob_pattern)
        sites = (
            lf.select(["chrom", "pos", "strand"])
            .unique()
            .collect()
        )
        site_sets.append(sites)

    # Compute intersection
    intersect = site_sets[0]
    for sites in site_sets[1:]:
        intersect = intersect.join(sites, on=["chrom", "pos", "strand"], how="inner")

    intersect = intersect.unique().sort(["chrom", "pos"])

    if output_path:
        intersect.write_parquet(output_path)

    return intersect


def load_chromosome_data(
    methylstore_path: str,
    chrom: str,
    samples: list[str],
    site_intersect: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """Load all data for a specific chromosome and set of samples.

    Optionally filters to an intersection of sites (for "unite" behavior).

    Parameters
    ----------
    methylstore_path : str
        Path to Parquet methylstore
    chrom : str
        Chromosome name (e.g., "chr1")
    samples : list[str]
        Sample identifiers to load
    site_intersect : pl.DataFrame, optional
        DataFrame with columns (chrom, pos, strand) to filter sites to.
        If provided, only sites in this set are returned.

    Returns
    -------
    pl.DataFrame
        Columns: chrom, pos, strand, N_meth, N_unmeth, coverage, sample
        Collected into memory (safe because one chromosome is manageable)
    """
    glob_pattern = f"{methylstore_path}/sample=*/chrom={chrom}/part-*.parquet"
    lf = pl.scan_parquet(glob_pattern)

    # Filter to selected samples
    lf = lf.filter(pl.col("sample").is_in(samples))

    # Optional: filter to intersection sites
    if site_intersect is not None:
        site_intersect_chrom = site_intersect.filter(pl.col("chrom") == chrom)
        if len(site_intersect_chrom) == 0:
            # No sites for this chrom in intersection; return empty
            return lf.filter(pl.lit(False)).collect()
        lf = lf.join(
            site_intersect_chrom.lazy(), on=["chrom", "pos", "strand"], how="inner"
        )

    # Collect into memory (safe: one chr across samples is manageable)
    df = lf.collect()
    return df
