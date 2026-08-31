# R081: Density-Adaptive Continuous Length Field

Status: fixed-checkpoint Panda gate completed and rejected; never integrated
into Stage 1 training or visualization.

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

Initial local verification on 2026-09-01:

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

## H100 Attempt Ledger

The first fixed-checkpoint invocation at commit `32f654a` passed the exact
source/checkpoint preflight, focused tests, and complete HGC test suite, then
stopped before candidate evaluation. The diagnostic validator incorrectly
required all three cached query-face vertex paths for every support source to
be finite. Formal `SurfaceSupport` stores `+inf` for unused face vertices and
defines the source distance as the minimum over the three paths; one finite
nonnegative path per support slot is the correct coverage invariant.

The failed invocation produced no diagnostic JSON and did not train, render,
mutate the checkpoint, or release the H100 allocation. Its preserved remote log
is:

`/home/wangyy/panda-r081-adaptive-continuous-field-runtime-20260901/logs/panda_r080_iter4000_r081_fixed_field.log`

SHA-256:
`7593af5f54a17f972a1ddfe9cde7a69f8e7c0ee079e3b77ff857a97d54db4518`.

The corrected validator now accepts unused `+inf`, requires at least one finite
nonnegative path in every `[query, source]` slot, and rejects NaN, `-inf`,
negative finite paths, or all-three-`+inf` holes. This is a diagnostic-only
correction; candidate and inherited interpolation values are unchanged.
After the correction, the complete local suite reports `489 passed`, with
focused mixed-path and coverage-hole regressions included. The exact H100 gate
must be rerun under a new output/log identity rather than overwriting the failed
attempt.

The corrected retry at commit
`a36e0de01c2fbd6d1ac5bd932f4e3f52028df019` completed on the full Panda R080
iteration-4000 population:

- render roots / guide roots / exact render edges:
  `496632 / 4500 / 15892224`;
- legacy/candidate support width: `8 / 9`;
- candidate support build and distance/weight time: `0.07578 / 0.14029 s`;
- candidate support bytes: `89,393,760`, versus legacy `79,461,120`;
- peak allocated/reserved CUDA memory: `1.397 / 1.548 GB`;
- candidate effective-guide-count median: `2.03082`, below legacy `2.26880`;
- candidate guide-site relative self-error P50/P95/max:
  `0.01072 / 0.06188 / 0.29651`;
- candidate-versus-legacy render-length relative difference P50/P95/max:
  `0.00661 / 0.04001 / 0.27416`.

The candidate smooths only inside an unchanged K+1 support cell:

| Absolute log-length edge jump | Legacy | R081 candidate |
| --- | ---: | ---: |
| all-edge mean | 0.026418 | 0.025634 |
| all-edge P95 | 0.114162 | 0.122116 |
| all-edge P99 | 0.226566 | 0.247155 |
| unchanged-support P95 | 0.027774 | 0.010192 |
| changed-support P95 | 0.169238 | 0.183033 |
| changed/unchanged P95 ratio | 6.093 | 17.959 |

Although the boundary source itself has zero Wendland mass, the query-adaptive
K+1 radius rescales every surviving weight when the discrete support changes.
With only eight active neighbors, the window is more concentrated than the
legacy field and produces a worse support seam. R081 therefore fails its core
continuity objective. No white-tiger run or canonical visualization is
justified, because the Panda numeric gate already rejects the representation.

Formal retry JSON:

`D:/RTS/_tmp/panda_r081_continuous_field_acceptance_20260901/retry1/panda_r080_iter4000_r081_fixed_field.json`

SHA-256:
`0a94e109828371a2c628c28a964e8e329493f5464b08d95cb4491abed42f9778`.

The next attempt may retain the literature-backed compact C2 window only if it
separately tests the standard particle-method premise of a larger fixed
neighbor mass. It must not tune a Panda length, region, or smoothing loss, and
it must continue to stop before training or visualization unless the exact
support-change edge gate improves.

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

R081 is rejected. Its core helper and diagnostic remain isolated evidence; no
config/checkpoint migration or bounded training continuation is authorized.
