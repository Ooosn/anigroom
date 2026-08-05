# V4 Surface-Direction Target

This directory contains the immutable clean-flow input used by the frozen
Stage 1 R036 baseline.

- file: `guide_flow3d_shell_targets_exclude_004_024_025.npz`
- SHA-256: `60a33b360bb415cb47cd38173d6e0cf4504448203ef277a5861641b40fdb3141`
- excluded source views: `004`, `024`, `025`
- guide layout: SMAL head `500`, body `4000`, candidate pool `65536`
- clean neighborhoods: head `K=24`, body `K=12`

The file is a formal model input, not a generated training output. Replacing
it changes the baseline and requires a new named R-series experiment.
