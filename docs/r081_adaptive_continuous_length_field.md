# R081: Density-Adaptive Continuous Length Field

Status: local pre-training implementation complete; fixed-checkpoint H100
evidence pending; not an active Stage 1 behavior.

## Purpose

R081 tests whether primary-guide length can define one continuous field on the
animal surface without changing the learned guide values, RGB evidence, losses,
training schedule, root population, or canonical visualization protocol.

The root hierarchy has two distinct roles:

- primary guide roots carry low-frequency control values and define the local
  spatial resolution of the field;
- render roots sample that field and must not inherit piecewise cells from one
  dominant guide.

Dense guide sampling must permit faster physical variation. Sparse guide
sampling must produce a larger physical support and therefore slower variation.
This is a representation property, not a body-region rule.

## Measured R080 Failure

The canonical R080 iteration-4000 `view09_length.png` is blocky. Exact
checkpoint attribution proves that this is not a visualization artifact:

- the current field is exactly the K8 primary-guide interpolation because the
  render residual multiplier is zero;
- the maximum guide weight has median `0.648689`;
- the effective participating-guide count has median `2.26880` despite the
  nominal K8 support;
- a dominant-guide-only field is within 5% of the current value for `84.9426%`
  of render roots;
- `43.66595%` of the exact render-surface edges change K8 support;
- unchanged-support edge log-length jump P95 is `0.0260935`, while
  changed-support P95 is `0.168496`, a `6.46x` increase;
- learned guide-edge log-length jump P95 is `0.387696`, versus `0.294197` for
  the stored reference field.

The failure therefore has two measured parts: a sharply dominant truncated
weight rule and a roughened source field. R081 addresses only the first part.

Evidence:

- `D:/RTS/_tmp/panda_r080_acceptance_20260831/diagnostics/length_distribution/interpolation_weight_attribution.json`
- `D:/RTS/_tmp/panda_r080_acceptance_20260831/diagnostics/length_distribution/interpolation_edge_attribution.json`

## Single Representation Change

The inherited mode remains byte-for-byte normalized inverse-square distance.
R081 adds an explicit opt-in candidate for primary-guide length only.

For the requested active neighbor count `K`, build exact topology-safe support
for `K+1` intrinsic neighbors. Let the sorted intrinsic distances be
`d_1 <= ... <= d_(K+1)` and define the local physical support radius

```text
h(x) = d_(K+1)(x).
```

For `q_i = d_i / h`, use the compact Wendland C2 window

```text
phi(q) = (1 - q)^4 (4q + 1),  0 <= q < 1
phi(q) = 0,                    q >= 1.
```

The normalized partition-of-unity weights are

```text
w_i(x) = phi(q_i) / sum_j phi(q_j).
```

The boundary neighbor has exactly zero value and zero first derivative in the
window. Any omitted source is at least as far away as the boundary source and
also has zero support. A source may therefore enter or leave the discrete
support only through zero kernel weight instead of replacing a nonzero Kth
inverse-distance contribution.

The physical guide length combination remains arithmetic in this first gate.
This isolates the weight representation from log-domain or decoder changes and
preserves positivity and the guide-value convex hull.

## Density Semantics

`h(x)` is measured in intrinsic surface distance and is set by local guide
density:

- dense regions have a small support radius and can express faster variation;
- sparse regions have a large support radius and can express only slower
  variation.

There is no species, body-region, image coordinate, absolute length, physical
radius, tuned smoothness coefficient, or postprocess blur.

## Strict Boundaries

Phase A is a fixed-checkpoint representation diagnostic. It must not:

- train or alter the R080 checkpoint;
- change primary guide values;
- change direction, width, brush, child spread, clump, color, opacity, curl,
  phase, secondary residuals, or render-root lifecycle;
- replace intrinsic surface support with Euclidean neighbors;
- change the official attribute visualization script, view, image resolution,
  point radius, colormap, base render, or scalar convention;
- silently handle a coverage hole, duplicate padded support, non-positive
  radius, or zero denominator.

The candidate fails hard on those invalid states.

## Pre-Training Verification

### Analytic and topology tests

1. Legacy inverse-square output remains exact by default.
2. Candidate weights are finite, nonnegative, and sum to one.
3. A constant physical field is reproduced exactly.
4. The candidate output stays inside the source-value range.
5. Uniform spatial scaling changes physical support radius but not normalized
   coordinates or weights.
6. A query crossing a support swap has no value jump from replacing a nonzero
   boundary source.
7. Query-position gradients are finite on both sides of a support boundary.
8. Folded and disconnected surface fixtures contain no cross-sheet source.
9. CPU and CUDA distance/support results remain equal.
10. Insufficient connected-component support and distance ties that leave no
    positive kernel mass fail explicitly.

### Fixed checkpoint gates

Run the exact learned guide values through both weight rules on:

- Panda R080 iteration 4000;
- a matched white-tiger checkpoint that can be loaded under an exact source and
  schema contract.

Record on the same exact surface edges:

- mean/P50/P90/P95/P99/P99.9/max absolute log-length jump;
- unchanged-support versus changed-support edge jumps;
- local support radius and effective-neighbor distributions;
- guide-site self-evaluation error;
- difference from the inherited K8 field;
- build time, forward time, peak memory, and support bytes;
- constant-field, positivity, finite-gradient, and no-cross-sheet invariants.

No metric threshold is invented before observing the fixed-checkpoint evidence.
The candidate cannot enter training if it merely blurs valid variation, changes
guide semantics materially, creates coverage holes, crosses nearby surface
sheets, or costs a per-step global solve.

## Visualization Contract

Only the existing formal attribute entry point may visualize the candidate:

`tools/visualize_white_tiger_groom_attributes.py`

Comparisons use original-resolution view 09 with the same base image and all
existing rendering conventions. No new heatmap implementation is authorized.

## Efficiency Contract

The candidate is a local sparse gather and normalized weighted sum. It adds one
boundary neighbor to the requested active support and introduces no iterative
PDE solve. Candidate IDs and intrinsic vertex paths are cached under the same
topology/lifecycle contract as the inherited interpolator. Numerical weights
remain differentiable in the current query position.

## Local Implementation Status

The opt-in core is isolated in
`anigroom/surface_interpolation.py::adaptive_wendland_c2_weights`. It has no
formal caller, config field, checkpoint field, or training behavior yet. The
fixed-checkpoint numeric entry point is
`tools/diagnose_adaptive_continuous_length_field.py`; it produces JSON only and
does not render or mutate a checkpoint.

Local verification on 2026-09-01:

- focused R081 plus inherited interpolation tests: `36 passed`;
- complete repository suite: `484 passed`, `0 failed`;
- exact folded/disconnected-surface test passes;
- CPU/CUDA candidate-weight parity passes;
- legacy inverse-square weights are explicitly frozen by an exact regression;
- the historical surface-smoothing regression remains unchanged.

The implementation intentionally stops before trainer/config integration. The
next gate runs the complete Panda R080 iteration-4000 field through the numeric
diagnostic and measures guide-site semantic drift, support-boundary continuity,
time, and memory before any canonical visualization or training continuation.

## Literature Basis

The design follows established particle and scattered-data field methods, not
an AniGroom-specific smoothing heuristic:

- SPH reconstructs continuous fields with normalized compact kernels and uses
  an adaptive smoothing length to couple sampling density to spatial
  resolution.
- Reproducing-kernel particle methods formalize continuous local particle
  fields and polynomial consistency.
- Wendland functions provide compact polynomial windows with prescribed
  smoothness.
- Natural-neighbor/Sibson coordinates remain the exact nodal, Voronoi-based
  reference if the compact kernel fails the guide-site semantic gate. A formal
  surface natural-neighbor backend must use a reviewed implementation and pass
  the folded-surface regression; it must not be recreated ad hoc.

References:

- https://academic.oup.com/mnras/article/330/1/129/1018860
- https://doi.org/10.1002/fld.1650200824
- https://doi.org/10.1007/BF02123482
- https://doc.cgal.org/latest/Interpolation/index.html

## Decision Rule

R081 remains diagnostic until analytic tests, Panda and white-tiger fixed-field
evidence, performance checks, and canonical visual review all pass. Only then
may a separately reviewed config/checkpoint migration expose the backend to a
bounded training continuation.
