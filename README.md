# methylstore — MVP Parquet methylation pipeline

This repository contains an initial scaffold for the MVP described in your plan: a small Python package to convert Bismark .cov files into a partitioned Parquet store and provide a CLI entrypoint.

Quickstart

1. Create a Python environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

2. Convert a sample:

```bash
epykit convert --input path/to/sample.cov.gz --sample-id S1 --output-dir methyl_store
```

Files created in this commit:

- `pyproject.toml` — project metadata and console entry
- `requirements.txt` — minimal dependencies
- `src/epykit/` — package code (converter + CLI)
- `tests/` — placeholder tests

Next steps: implement filtering, per-chromosome DMC, tests and CI.
