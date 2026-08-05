# R035: Hierarchical Width Profile

Status: formal from-zero run complete; hierarchical width ownership accepted
and retained by R036.

## Base And Scope

R035 keeps the accepted R034 absolute, uncapped segment allocator. Its single
method change is a coherent replacement of the active width-profile path:

- root width is positive and unbounded around a local physical reference;
- guide roots own root width, tip/root ratio, and taper as the low-frequency
  editable field;
- render roots own only zero-centered relative residual coordinates;
- all three render residuals use the existing 10k-20k geometry unlock;
- guide and render lifecycle updates reuse surface interpolation and migrate
  optimizer state.

No body region, screen-space size, absolute width endpoint, percentile gate,
or white-tiger-specific training rule is introduced.

## Parameterization

For guide root width `w_g`, tip ratio `q_g`, taper `p_g`, and render residual
coordinates `r_w`, `r_q`, `r_p`:

`w = w_g * exp(s * asinh(r_w))`

`q = sigmoid(logit(q_g) + s * asinh(r_q))`

`p = p_g * exp(s * asinh(r_p))`

where `s` is the existing shared render-geometry unlock multiplier times the
existing width residual scale. Zero residual exactly reproduces guide
interpolation. Root width and taper stay positive without physical maxima;
tip ratio keeps only its semantic `[0,1]` domain.

The generic strand field uses the same positive reference-relative coordinate
for root width. The retired `[0.00008, 0.0020]` root-width range no longer
exists in the active representation.

## Identifiability And Regularization

R034 proved that neighbor smoothness cannot detect a coherent global drift to
cylindrical strands. R035 therefore separates ownership rather than restoring
a cap:

- guide width profile is smoothed on the intrinsic mesh graph;
- render width residuals are smoothed in their zero-centered coordinates;
- the final effective width profile is smoothed in log-width, logit-ratio, and
  log-taper coordinates;
- the existing width prior penalizes render residual magnitude and softly
  anchors only the weakly identifiable guide tip/taper coordinates to their
  initialization references.

The anchor is a soft coordinate prior, not a physical clamp. Guide root width
remains free to learn coverage.

## Lifecycle Contract

Guide densification interpolates physical root width, tip ratio, and taper,
interpolates their local references, then re-encodes the child coordinates.
Render densification interpolates the zero-centered residual coordinates and
initializes new optimizer rows to zero while retaining surviving Adam rows.
The direct per-render-root absolute width parameters are not members of the
formal optimizer.

## Local Verification

- Python compilation: passed;
- complete repository suite: `56 passed`;
- zero residual equals guide interpolation for all width controls: passed;
- positive unbounded root width and taper: passed;
- semantic tip ratio: passed;
- render and guide lifecycle strict-state reload: passed;
- optimizer contains hierarchical width fields and excludes direct groom
  width endpoints: passed.

## Formal Result

The strict H100 run completed from zero at train/test composite PSNR
`33.44020 / 32.65968`; best test composite was `32.84404` at 29k. It ended
with `318942` render roots, `14278095` Gaussians, mean/max segments
`11.1918 / 19`, and `24103.67 MiB` peak live allocation. Fixed-protocol asset
QA found no width collapse or new curl-back failure. R036 retains this width
hierarchy and changes only child-spread ownership.
