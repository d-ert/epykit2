"""Synthetic Bismark .cov dataset with known DMC/DMR truth.

Used by the test suite to measure power, FDR, and effect-size bias of every
DMC and DMR backend in epykit. The data-generating model is intentionally
simple so the *truth* is unambiguous:

    pi_ij  = baseline + effect[i] * is_treatment(j) + noise[i, j]
    cov_ij ~ NegativeBinomial(mean=coverage_mean, dispersion=coverage_disp)
    meth_ij ~ Binomial(cov_ij, clip(pi_ij, 0.01, 0.99))

Effect placement:

* ``n_dmrs`` contiguous regions of ``dmr_size_cpgs`` CpGs each receive the
  same signed ``dmr_effect`` (a real biological DMR).
* ``n_scattered_dmcs`` isolated CpGs outside any DMR receive ±``dmc_effect``
  with random signs (scattered DMCs, not part of a DMR).

The remainder are null (effect == 0).

Outputs in ``out_dir/``:

  cov/<sample_id>.bismark.cov.gz   — per-sample Bismark .cov files
  samplesheet.csv                  — sample_id, group, path
  truth.parquet                    — chrom, pos, is_dmc, true_meth_diff,
                                     dmr_id, in_dmr
  config.json                      — dataclass dump for reproducibility
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


@dataclass
class SimConfig:
    """Knobs for the synthetic methylation generator.

    The defaults are chosen so that a "standard" 4-vs-4 WGBS comparison
    at ~20× coverage produces detectable signal under BH-corrected
    multiple testing across ~75 k post-filter CpGs. Specifically:

    * dmc_effect = 0.40: bigger than the typical 0.20-0.30 promoter Δβ,
      so n=4 replicates can resolve it with a per-site count test (LR /
      score) at BH q<0.05. With Δβ=0.30 and n=4 the BH cutoff after
      75k tests demands an unrealistic z-statistic.
    * coverage_mean = 20 × NB(disp=5): puts most sites at ≥10× after
      filter, well within the "well-powered" regime for binomial GLMs.
    * replicate_sd = 0.03: modest between-replicate variation that still
      makes the count-vs-Welch-t comparison interesting.

    Tightening any of these gives the engines an easier time and risks
    masking real bugs; loosening them makes the power thresholds in
    ``test_accuracy.py`` too optimistic to clear at honest noise levels.
    """

    n_per_group: int = 4
    chromosomes: tuple[str, ...] = ("chr1", "chr2", "chr3", "chr4", "chr5")
    cpgs_per_chrom: int = 2_000
    baseline_meth: float = 0.30
    n_scattered_dmcs: int = 500
    dmc_effect: float = 0.40
    n_dmrs: int = 10
    dmr_size_cpgs: int = 10
    dmr_effect: float = 0.40
    coverage_mean: float = 20.0
    coverage_disp: float = 5.0  # NB shape (k); larger → less overdispersion
    replicate_sd: float = 0.03
    seed: int = 42

    @property
    def total_samples(self) -> int:
        return self.n_per_group * 2

    @property
    def n_total_sites(self) -> int:
        return len(self.chromosomes) * self.cpgs_per_chrom


def _nb_coverage(rng: np.random.Generator, mean: float, k: float, n: int) -> np.ndarray:
    """Negative-binomial coverage with mean ``mean`` and dispersion ``k``.

    numpy's parametrisation: NB(k, p) where mean = k * (1 - p) / p. We solve
    for p given the requested mean and k. Returns int64, minimum 1 to avoid
    accidental zero-coverage sites at low means.
    """
    p = k / (k + mean)
    out = rng.negative_binomial(k, p, size=n).astype(np.int64)
    return np.maximum(out, 1)


def _place_effects(cfg: SimConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decide per-site effect size + DMR membership.

    Returns (effects, dmr_id, chrom_of_site) where:
      effects[i]    = true Δβ at site i (0 for null sites)
      dmr_id[i]     = DMR index (0..n_dmrs-1) or -1 if not in a DMR
      chrom_of_site = chromosome name per site (for joining)
    """
    n = cfg.n_total_sites
    effects = np.zeros(n, dtype=np.float64)
    dmr_id = np.full(n, -1, dtype=np.int32)

    chroms_arr = np.repeat(np.array(cfg.chromosomes, dtype=object), cfg.cpgs_per_chrom)
    n_chroms = len(cfg.chromosomes)

    # 1. Place DMRs first: each DMR sits entirely on one chromosome.
    # Sign assignment is *deterministically* balanced (half hyper, half hypo)
    # so the truth table has zero mean effect over DMRs. Without this the
    # random ±1 draw produces a small fixture-level imbalance (e.g. 7/3 at
    # n_dmrs=10) that shows up as "bias" in test assertions.
    dmr_signs = np.concatenate([
        np.full(cfg.n_dmrs // 2, 1.0),
        np.full(cfg.n_dmrs - cfg.n_dmrs // 2, -1.0),
    ])
    rng.shuffle(dmr_signs)
    for d in range(cfg.n_dmrs):
        chrom_idx = rng.integers(0, n_chroms)
        within = rng.integers(0, cfg.cpgs_per_chrom - cfg.dmr_size_cpgs)
        start = chrom_idx * cfg.cpgs_per_chrom + int(within)
        end = start + cfg.dmr_size_cpgs
        effects[start:end] = float(dmr_signs[d]) * cfg.dmr_effect
        dmr_id[start:end] = d

    # 2. Scatter remaining DMCs into not-in-DMR positions. Signs balanced
    # deterministically (same reasoning as DMRs).
    pool = np.where(dmr_id == -1)[0]
    n_chosen = min(cfg.n_scattered_dmcs, len(pool))
    chosen = rng.choice(pool, size=n_chosen, replace=False)
    scatter_signs = np.concatenate([
        np.full(n_chosen // 2, 1.0),
        np.full(n_chosen - n_chosen // 2, -1.0),
    ])
    rng.shuffle(scatter_signs)
    effects[chosen] = scatter_signs * cfg.dmc_effect

    return effects, dmr_id, chroms_arr


def _positions(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """Generate sorted positions per chromosome, spaced 50–400 bp apart."""
    parts = []
    base = 1_000
    for _ in cfg.chromosomes:
        gaps = rng.integers(50, 400, cfg.cpgs_per_chrom)
        pos = base + np.cumsum(gaps).astype(np.int64)
        parts.append(pos)
    return np.concatenate(parts)


def _write_cov_gz(path: Path, chroms: np.ndarray, positions: np.ndarray,
                  N_meth: np.ndarray, N_unmeth: np.ndarray) -> None:
    """Write a Bismark .cov.gz file.

    Format (tab-separated, no header):
        chrom  start  end  methyl_percent  N_meth  N_unmeth

    epykit's converter treats ``start`` as 0-based (BED-format), matching
    nf-core/methylseq's bismark2bedGraph output. We write start = pos,
    end = pos + 1 (single-CpG interval).
    """
    coverage = (N_meth + N_unmeth).astype(np.int64)
    methyl_pct = 100.0 * N_meth / np.maximum(coverage, 1)
    df = pd.DataFrame({
        "chrom": chroms,
        "start": positions.astype(np.int64),
        "end": (positions + 1).astype(np.int64),
        "methyl_pct": methyl_pct.astype(np.float64),
        "N_meth": N_meth.astype(np.int64),
        "N_unmeth": N_unmeth.astype(np.int64),
    })
    with gzip.open(path, "wt", newline="") as fh:
        df.to_csv(fh, sep="\t", header=False, index=False, float_format="%.4f",
                  lineterminator="\n")


def generate(cfg: SimConfig, out_dir: str | Path) -> dict:
    """Generate Bismark .cov.gz files + samplesheet + truth table.

    Returns a dict with paths and summary stats so callers can hand
    ``samplesheet`` straight into ``ep.read_bismark`` and join ``truth``
    onto DMC results.
    """
    out_dir = Path(out_dir)
    cov_dir = out_dir / "cov"
    cov_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    # Per-site truth (same across all samples).
    effects, dmr_id, chroms_arr = _place_effects(cfg, rng)
    positions = _positions(cfg, rng)
    n = cfg.n_total_sites

    # Generate per-sample read counts.
    sample_records: list[dict] = []
    for sample_idx in range(cfg.total_samples):
        is_treatment = sample_idx < cfg.n_per_group
        group = "treatment" if is_treatment else "control"
        sid = f"{group}_{(sample_idx % cfg.n_per_group) + 1}"

        # True per-site β for this sample.
        per_site_effect = effects if is_treatment else np.zeros_like(effects)
        # Independent replicate noise around the group mean.
        rep_noise = rng.normal(0.0, cfg.replicate_sd, size=n)
        beta = np.clip(cfg.baseline_meth + per_site_effect + rep_noise, 0.01, 0.99)

        cov = _nb_coverage(rng, cfg.coverage_mean, cfg.coverage_disp, n)
        meth = rng.binomial(cov, beta).astype(np.int64)
        unmeth = cov - meth

        cov_path = cov_dir / f"{sid}.bismark.cov.gz"
        _write_cov_gz(cov_path, chroms_arr, positions, meth, unmeth)
        sample_records.append({
            "sample_id": sid,
            "group": group,
            "path": str(cov_path),
        })

    # Samplesheet.
    samplesheet_path = out_dir / "samplesheet.csv"
    pd.DataFrame(sample_records).to_csv(samplesheet_path, index=False)

    # Truth table.
    #
    # ``true_meth_diff`` stores the **post-clip effective Δβ**, not the
    # raw ``effects[i]``. With baseline=0.30 + effect=±0.40, the intended
    # hypo β_treatment of −0.10 gets clipped to 0.01 by the sampling
    # model, so the actual realised Δβ is 0.01 − 0.30 = −0.29 (not −0.40).
    # If we stored the unclipped intended effect, the estimator would
    # appear ~5pp positively biased on hypo sites even though it's
    # correctly recovering what was sampled.
    pi_treat = np.clip(cfg.baseline_meth + effects, 0.01, 0.99)
    pi_ctrl  = np.full_like(effects, cfg.baseline_meth)
    true_meth_diff_postclip = pi_treat - pi_ctrl
    truth_df = pl.DataFrame({
        "chrom": [str(c) for c in chroms_arr.tolist()],
        "pos": positions.astype(np.int64).tolist(),
        "is_dmc": (effects != 0).tolist(),
        "true_meth_diff": true_meth_diff_postclip.tolist(),
        "dmr_id": dmr_id.tolist(),
        "in_dmr": (dmr_id >= 0).tolist(),
    })
    truth_path = out_dir / "truth.parquet"
    truth_df.write_parquet(truth_path)

    # Config dump for reproducibility.
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    return {
        "samplesheet": str(samplesheet_path),
        "truth": truth_path,
        "out_dir": str(out_dir),
        "cov_dir": str(cov_dir),
        "n_total_sites": n,
        "n_dmcs_true": int((effects != 0).sum()),
        "n_dmrs": cfg.n_dmrs,
        "chromosomes": list(cfg.chromosomes),
        "sample_ids": [r["sample_id"] for r in sample_records],
        "treatment_ids": [r["sample_id"] for r in sample_records if r["group"] == "treatment"],
        "control_ids": [r["sample_id"] for r in sample_records if r["group"] == "control"],
        "config": cfg,
    }
