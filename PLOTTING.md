# New Plotting Functions

All new plotting functions follow a consistent API:

```python
import epykit as ep

# Load data
md = ep.read_bismark("samplesheet.csv", ...)

# Run analysis
ep.tl.dmc(md)
ep.tl.annotate(md, gtf="gencode.v49.gtf")

# Plot with new composable interface
fig, ax = ep.pl.volcano(md, alpha=0.05, min_abs_diff=0.1)
fig.savefig("volcano.pdf", dpi=300)

# Or just auto-save
ep.pl.volcano(md, save="volcano_output")
```

## Added Plots

### Differential Analysis

#### `ep.pl.ma_plot()`
Mean Average plot — methylation difference vs mean beta.
- **x-axis**: mean methylation (average across treatment & control)
- **y-axis**: methylation difference
- **color**: hypermethylated (red), hypomethylated (blue), not significant (grey)
- Detects coverage/batch biases invisible in volcano plots

```python
fig, ax = ep.pl.ma_plot(md, alpha=0.05, min_abs_diff=0.1)
```

#### `ep.pl.manhattan()`
Genome-wide significance plot — positions vs -log10(p-value).
- **x-axis**: genomic position (by chromosome)
- **y-axis**: -log10(p-value)
- **color**: alternates by chromosome for readability
- Useful for regional signal detection

```python
fig, ax = ep.pl.manhattan(md, alpha=0.05)
```

### Sample Clustering

#### `ep.pl.pca()`
PCA of per-sample methylation profiles.
- Samples ~10,000 sites (configurable) to avoid OOM
- Only uses complete (non-NaN) sites
- Colored by treatment group
- Shows sample clustering

```python
fig, ax = ep.pl.pca(md, n_sites=10000)
```

## API Conventions

All functions follow this pattern:

```python
fig, ax = ep.pl.plot_name(
    md,                    # MethylData object (first arg, required)
    *,
    param1=value1,         # plot-specific kwargs (keyword-only)
    ax=None,               # optional: draw on existing axes
    figsize=(6, 5),        # optional: figure size
    save=None,             # optional: auto-save to file/name
)
```

- **Return**: `(fig, ax)` — enables composition and customization
- **Optional save**: pass filename to save automatically
- **Theme**: applied on import from `_style.py` (spines, dpi, fonts)
- **Palette**: defined in `PALETTE` dict, easily customizable

## Example: Compose Multiple Plots

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ep.pl.volcano(md, ax=axes[0, 0])
ep.pl.ma_plot(md, ax=axes[0, 1])
ep.pl.manhattan(md, ax=axes[1, 0])
ep.pl.pca(md, ax=axes[1, 1])

fig.tight_layout()
fig.savefig("analysis_summary.pdf", dpi=300)
```

## Dependencies

- **matplotlib**: all plots
- **seaborn**: heatmaps
- **scikit-learn**: PCA only (imported on demand)

Install if missing:
```bash
pip install matplotlib seaborn scikit-learn
```
