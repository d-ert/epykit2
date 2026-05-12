"""epykit: Parquet methylation pipeline (Bismark ingestion → DMR annotation)

This package provides a complete WGBS analysis pipeline:

  - convert:   Bismark .cov → partitioned Parquet methylstore
  - filter:    QC / coverage filtering / site intersection
  - dmc:       Differential methylation calling per CpG (Fisher or
               beta-binomial)
  - dmr:       DMR calling (sliding-window) and BSmooth-style smoothing
  - annotate:  Gene-feature and CpG-island context annotation
  - qc:        Bisulfite conversion rate, global methylation, coverage
               uniformity
"""

__version__ = "0.1.0"

from .methyldata import MethylData
from .io import read_bismark, read_nfcore_methylseq, load
from . import pp, tl
from . import pl as _pl_module


class _PlottingNamespace:
    def __getattr__(self, name: str):
        try:
            import matplotlib.pyplot  # noqa: F401
            import seaborn  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Plotting requires matplotlib and seaborn. Install with: "
                "pip install epykit[plotting]"
            ) from exc
        return getattr(_pl_module, name)


pl = _PlottingNamespace()

from .convert import convert_sample
from .filter import (
    sample_summary,
    filter_sites,
    intersect_sites,
    load_chromosome_data,
    get_coverage_quantile,
    normalize_coverage_store,
)
from .dmc import (
    process_chromosomes_dmc,
    calculate_diff_meth_chromosome,
    apply_multiple_testing_correction,
    fisher_exact_vectorized,
    beta_binomial_test,
)
from .dmr import (
    call_dmr_sliding_window,
    smooth_methylation_bsmooth,
)
from .annotate import (
    annotate_features,
    annotate_cpg_islands,
)
from .qc import (
    bisulfite_conversion_rate,
    global_methylation_report,
    coverage_uniformity,
)