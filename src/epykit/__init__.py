"""epykit: MVP Parquet methylation pipeline (Bismark ingestion)

This package provides a minimal converter to normalize Bismark .cov files
into a partitioned Parquet layout: sample=<sample>/chrom=<chrom>/part-*.parquet

Modules:
  - convert: Bismark .cov → partitioned Parquet
  - filter: QC, filtering, site intersection
  - dmc: Differential methylation calling (per-chromosome)
"""

__version__ = "0.1.0"

from .convert import convert_sample
from .filter import (
    sample_summary,
    filter_sites,
    intersect_sites,
    load_chromosome_data,
    get_coverage_quantile,
)
from .dmc import (
    process_chromosomes_dmc,
    calculate_diff_meth_chromosome,
    apply_multiple_testing_correction,
    fisher_exact_vectorized,
)
