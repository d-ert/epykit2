"""Tests for AnnData export."""

from __future__ import annotations

import pytest


def test_to_anndata_requires_unite(synth_md_filtered):
    """to_anndata() refuses to densify before unite()."""
    import epykit as ep
    pytest.importorskip("anndata")
    md = synth_md_filtered
    # The fixture has unite=intersect already; strip it to check the guard.
    md.uns.pop("unite", None)
    with pytest.raises(ValueError, match="pp.unite"):
        md.to_anndata()


def test_to_anndata_shape_and_layers(synth_md_filtered):
    import epykit as ep
    pytest.importorskip("anndata")
    adata = synth_md_filtered.to_anndata(layer="beta")
    assert adata.shape[0] == synth_md_filtered.n_samples
    assert adata.shape[1] > 0
    # Layers populated
    for layer in ("coverage", "N_meth", "N_unmeth"):
        assert layer in adata.layers
        assert adata.layers[layer].shape == adata.shape
    # Provenance keys
    assert adata.uns.get("epykit_assembly") == synth_md_filtered.assembly
    assert adata.uns.get("epykit_context") == synth_md_filtered.context


def test_to_anndata_beta_in_unit_interval(synth_md_filtered):
    import epykit as ep
    import numpy as np
    pytest.importorskip("anndata")
    adata = synth_md_filtered.to_anndata(layer="beta")
    arr = np.asarray(adata.X)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        pytest.skip("no finite β values")
    assert finite.min() >= 0.0
    assert finite.max() <= 1.0
