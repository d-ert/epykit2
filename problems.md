## Biological / statistical critique

**B1. No automated agreement test against methylKit. — Critical for credibility.**
The whole positioning is "methylKit parity" ([cli.py:1-14](epykit2/src/epykit/cli.py:1), [tl.py:1-15](epykit2/src/epykit/tl.py:1), [_glm.py:160-163](epykit2/src/epykit/_glm.py:160)). There is no fixture-based regression test that ingests a tiny .cov dataset, runs `tl.dmc` / `tl.dmr`, and compares q-values to a frozen methylKit output. Without that, every refactor is a leap of faith and "parity" is an unverified claim.

**B5. Smoothing labelled "BSmooth-style" but isn't. — Medium.**
Real BSmooth is local LOESS with coverage-weighted observations. The implementation appears to be a Gaussian convolution on a regular grid with interpolation back to CpG positions. That's a reasonable approximation, but downstream users will cite "BSmooth smoothing" and be wrong. Either rename to `gaussian_smooth` or implement coverage-weighted LOESS. The function is currently EXPERIMENTAL and not consumed by `tl.dmr` ([pp.py:160-174](epykit2/src/epykit/pp.py:160)), so this is a documentation fix today and a design decision before you wire it into DMR.
