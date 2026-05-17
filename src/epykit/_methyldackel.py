"""MethylDackel .bedGraph → partitioned Parquet converter.

MethylDackel's ``extract`` subcommand emits ``.bedGraph`` files in the same
6-column layout as Bismark's ``.cov`` files
(``chrom, start, end, methylation_percent, count_methylated,
count_unmethylated``), with one extra header line:

    track type="bedGraph" description="..."

This module is a thin wrapper around :func:`epykit.convert.convert_sample`
that passes ``format="methyldackel"`` so the polars CSV scan skips that
header. All other behaviour (caching manifest, strand inference, CpG-pair
merging) is identical to the Bismark path.

The public ``read_methyldackel`` entry point lives in :mod:`epykit.io`.
"""

from __future__ import annotations

from .convert import convert_sample, ensure_converted_sample


def convert_methyldackel_sample(
    input_path: str,
    sample_name: str,
    output_dir: str,
    row_group_size: int = 1_000_000,
    context: str = "CpG",
    reference_fasta: str | None = None,
    merge_strands: bool = True,
) -> None:
    """Convert a single MethylDackel ``.bedGraph[.gz]`` file.

    Same parameters as :func:`epykit.convert.convert_sample` minus
    ``format``; see that function for details.
    """
    convert_sample(
        input_path,
        sample_name,
        output_dir,
        row_group_size=row_group_size,
        context=context,
        reference_fasta=reference_fasta,
        merge_strands=merge_strands,
        format="methyldackel",
    )


def ensure_converted_methyldackel_sample(
    input_path: str,
    sample_name: str,
    output_dir: str,
    row_group_size: int = 1_000_000,
    context: str = "CpG",
    reference_fasta: str | None = None,
) -> bool:
    """Cache-aware MethylDackel converter. See
    :func:`epykit.convert.ensure_converted_sample`."""
    return ensure_converted_sample(
        input_path,
        sample_name,
        output_dir,
        row_group_size=row_group_size,
        context=context,
        reference_fasta=reference_fasta,
        format="methyldackel",
    )


__all__ = [
    "convert_methyldackel_sample",
    "ensure_converted_methyldackel_sample",
]
