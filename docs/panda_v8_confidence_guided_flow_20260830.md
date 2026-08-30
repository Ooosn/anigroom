# Panda V8 Confidence-Guided Directed Flow

Status date: 2026-08-30.

This note records the causal diagnosis, rejected alternatives, accepted local
prototype, formal implementation contract, and remaining execution gate for
the Panda upper-back flow conflict reported from the R073 3k asset.

## Scope And Frozen Training State

The visible defect is the opposing strand flow in the upper-back crop of the
Panda pure-fur asset. Training is not the source of this direction conflict:

- R073 is frozen at 3k;
- its guide-direction residual multiplier is exactly zero at 3k;
- guide-direction residual unlock starts at 10k and finishes at 20k;
- the inspected checkpoint is therefore still showing the initialized guide
  field for this direction question.

The remote R073 checkpoint remains:

`/home/wangyy/panda-r073-budget-runtime-20260830/outputs/panda_r073_budget_0_3k_h100_20260830/checkpoint_003000.pt`

No later R073 training was authorized or launched during this investigation.

## Corrected Causal Attribution

The first numeric attribution accidentally paired the global graph endpoint
arrays with postratio edge dots whose helper owns a different canonical edge
ordering. Those root-pair labels were invalid. The diagnostic contract is now
repaired: postratio edge dots are serialized with their own `edge_u/edge_v`.

Recomputing every dot directly from `cleaned_directed_flow3d` on the global
canonical graph gives the valid result for the user-marked upper-back region:

- 952 local graph edges;
- 87 negative directed edges (`9.14%`);
- 41 / 453 negative front-facing edges (`9.05%`);
- 37 front-facing edges above 120 degrees;
- 35 / 457 negative back-facing edges;
- all 87 negative edges lie inside one V7 global-sign supernode;
- none is a cross-supernode sign constraint.

The worst valid front-facing pair reaches `161.14 deg`. Its endpoints often
pair a strong anchor with a root having near-zero unary margin/coherence, or
two roots with little shared view support. The source view-27 drawing and its
aligned 2D orientation are smooth, while the conflict already exists in
`raw_flow3d/flow3d`. Therefore:

1. curl and training are excluded;
2. a simple final arrow flip is insufficient;
3. V7's equality union can correctly protect a sign block while still hiding
   a continuous 3D-axis seam inside that block;
4. low-reliability fused axes must be reconstructed from stronger surface
   neighbors, not treated as equal observations.

The render crop contains both sides of the silhouette: 1,448 sampled strands
intersect it, split into 738 front-facing and 710 back-facing strands. That
overlap amplifies the appearance, but does not explain away the true guide
seam: the guide field itself contains the 87 valid negative edges above.

## Rejected Repairs

### Pure sign reflection

An axis-preserving tangent-sign pass was tested first. It retained each root's
local axis and outward-normal ratio, changing only the tangent sign selected
from stronger neighbors. Panda accepted zero basins and retained
`594 / 233` observed negative/severe graph edges. This proves that the defect
is not only a binary arrow-head error; the local 3D axis is rotated.

### Unnormalized whole-basin averaging

A max-confidence watershed followed by summing every upstream-neighbor weight
removed the marked conflict, but correlated neighbor count inflated the
propagation force. On changed Panda roots the unweighted axial evidence
residual moved `0.3900 -> 0.6394`; White moved `0.4447 -> 0.6850`. That arm was
rejected even though its continuity numbers were strong.

### Full-graph negative seeding

Starting propagation from edges with unobserved endpoints removed only 24
additional Panda and 10 additional White all-graph negative edges. It also
rotated a small set of roots with no direct evidence by as much as 125 degrees.
Training interpolation already selects only `observed` clean-flow roots as
sources, so this broader trigger has no justified benefit. V8 starts from
observed conflicts and may propagate through uncertain/unobserved neighbors;
it does not let an unobserved-only edge initiate a repair.

## Accepted Algorithm

The formal function is
`anigroom.flow.confidence_guided_direction.refine_confidence_guided_directed_flow`.
It is enabled only by:

`--directed-flow-propagation-mode confidence-guided`

The default is `none`, preserving exact V7 behavior unless the V8 route is
explicitly selected.

### Joint reliability

Each root receives one continuous reliability value:

`trusted axial confidence * global unary normalized margin * global vote coherence`

No species, body part, image coordinate, view index, or hand-authored arrow is
consulted. No confidence cutoff is introduced.

### Stage 1: canonical confidence watershed

All roots compete as sources in a canonical max-product surface watershed.
Confidence decays by the existing trusted-view-cluster factor `0.85` per graph
step. A root joins another source only when the propagated reliability is
strictly stronger than its own; canonical rank resolves exact ties.

Within one watershed, stronger upstream directions are parallel transported
to each root. Neighbor vectors are averaged first, then contribute only the
single propagated confidence at that root. This prevents graph degree from
inflating evidence. The root's original direction remains weighted by its own
joint reliability.

A watershed is accepted only when all of the following hold:

- severe incident edge count strictly decreases;
- total negative incident edge count strictly decreases;
- negative-dot hinge strictly decreases;
- no previously non-severe edge becomes severe;
- severe edges removed per changed root is strictly above the input sample's
  own graph-wide severe-edge count per root.

The last condition is a sample-adaptive benefit/cost gate, not a Panda-tuned
threshold. Accepted source roots become protected anchors.

### Stage 2: monotone local cleanup

Residual observed negative edges enter a priority queue. A root can update
only from neighbors whose confidence remains stronger after the same `0.85`
decay. Neighbor directions are parallel transported and confidence-normalized.
An update is accepted only when its incident negative count and hinge both
strictly decrease and it creates no new severe edge. Protected watershed
owners cannot move.

Because every accepted local update strictly reduces the global negative-edge
count, the pass terminates without an iteration cap. Root and edge canonical
ordering make the result invariant to input root and edge permutations.

## Accepted Local Regression

The accepted observed-edge counts are:

| Sample | V7 negative | Watershed | Final | V7 severe | Watershed | Final | New severe |
|---|---:|---:|---:|---:|---:|---:|---:|
| Panda | 594 | 280 | 92 | 233 | 24 | 5 | 0 |
| White tiger | 1104 | 793 | 202 | 246 | 110 | 40 | 0 |

Panda accepts 4 watersheds and 110 local updates; White accepts 8 watersheds
and 207 local updates. Formal-module output reproduces the accepted prototype
to a maximum component error of `1.7881393e-7`. The maximum reported angular
difference is only a float32 dot/`acos` sensitivity at near-identical vectors.

The final candidate changes 843 Panda roots and 812 White roots. All accepted
watershed owners remain unchanged. Both outputs are finite, normalized,
outward-facing, and contain zero newly severe graph edges.

The evidence tradeoff is intentionally measured against original V7 rather
than hidden:

| Sample | Joint-trust axial residual | Joint-trust directed residual |
|---|---:|---:|
| Panda | `0.40362 -> 0.42506` | `0.37232 -> 0.37778` |
| White tiger | `0.41852 -> 0.43730` | `0.39794 -> 0.40602` |

Panda's raw all-view directed residual improves slightly
(`0.45477 -> 0.45310`). White's raw directed residual changes
`0.43794 -> 0.44891`. These small global evidence movements accompany the
large monotone continuity reduction and occur primarily on roots whose fused
axis/sign evidence is unreliable.

## User-Region Acceptance

The user crop is matched to the exact R073 side-positive-Y asset at score
`0.9907600`, image bbox `[1285, 5, 1519, 64]`. On the fixed 952-edge guide
analysis box:

| Field | All negative | Front negative | Back negative | Above 120 deg |
|---|---:|---:|---:|---:|
| V7 | 87 | 41 | 35 | 83 all / 37 front |
| V8 final candidate | 0 | 0 | 0 | 0 |

The final arrow overlay contains no red 3D-conflict arrows in that box.

Local candidate target:

`D:\RTS\_tmp\panda_r073_budget_acceptance_20260830\v8_confidence_normalized_local_cleanup_20260830\panda_v8_final_candidate.npz`

SHA-256:

`26c5e1dd7ab4c5bddb0c6227ca62018d785e9e61d4231d741a0ba9d7aa04a123`

Final user-region overlay:

`D:\RTS\_tmp\panda_r073_budget_acceptance_20260830\v8_final_candidate_user_region_20260830\v7_guide_overlay.png`

SHA-256:

`ff041c0c001f1ac3e6c79e18d834f8dde120108214e78104d3d6af745cbe44a9`

Formal local validation report:

`D:\RTS\_tmp\panda_r073_budget_acceptance_20260830\formal_v8_module_validation_20260830\report.json`

SHA-256:

`1f582294fde1cc7a4bd47b477e56363ed46f6eaedd6650d8f1c25ca6d587b726`

## Formal Generation Gate

The local candidate is diagnostic evidence, not yet an accepted immutable
training target. The next step is to commit and review the source, then run:

`scripts/server/run_panda_white_confidence_guided_v8.sh`

The launcher preserves V7 data and fusion arguments, selects the V8 mode
explicitly, runs the full test suite, refuses a dirty or mismatched source
commit, refuses output overwrite, verifies every NPZ diagnostic contract, and
fails if any confidence-guided edge becomes newly severe.

Only after formal Panda and White targets pass fixed-view arrows and canonical
asset inspection may a new from-zero H100 training run begin. The frozen R073
3k experiment must not be resumed as a substitute for that from-zero gate.
