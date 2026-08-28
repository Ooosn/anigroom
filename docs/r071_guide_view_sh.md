# R071 Primary-Guide View-Dependent SH Gate

Status date: 2026-08-28.

Status: implementation and local validation in progress. No H100 result is
accepted by this document.

## Question

The matched Panda experiment proves that the same R068+V7 representation can
produce a coherent view-09 coat when trained from zero on view 09 alone, while
the 30-view run is already granular before 9k and becomes a noisy bald patch
after guide unlock. Can a bounded view-dependent appearance field absorb the
trusted angular RGB variation before it pressures shared roots, opacity, and
later guide geometry?

## Single Method Change

R071 inherits the exact R068 behavior with the rejected R069 support gauge
explicitly disabled. It adds one appearance field:

- every primary guide owns a zero-initialized degree-one RGB spherical-
  harmonic residual;
- the field has no DC term, so root/tip color retains intrinsic coat color;
- all Gaussians belonging to one render strand receive the same effective SH
  residual after primary-guide surface interpolation;
- the basis is evaluated in a detached hair-local frame built from the guide
  direction and surface normal;
- the field cannot send gradients to root placement, direction, length,
  width, mesh calibration, opacity, or lifecycle state;
- RGB-flow treats the field as color and does not update it.

The SH coefficients are bounded as `tanh(raw) * 0.20`. The field is active
from iteration zero and uses its own `0.020` learning rate. It is not frozen by
the 9k geometry-guide freeze.

## Trusted View Ownership

The accepted V7 target stores
`axis_view_cluster_selected_direct_weight[V,G]`. Its sibling `summary.json`
stores the matching `views_used` IDs. R071 requires both, validates exact guide
face/barycentric identity, normalizes positive evidence by its global q95, and
uses the resulting confidence only to scale the current view's SH gradient.

The confidence does not alter the forward SH value. A training view absent
from the trusted V7 set receives exactly zero SH gradient. No image-space box,
species rule, body label, or hard-coded view ownership is introduced.

## Why Guide-Owned

Primary-guide ownership keeps only `4500 * 3 * 3` coefficients. Render-root
lifecycle changes automatically inherit the field through the existing K8
surface support; no per-render-root SH parameter or Adam-state migration is
required. K8 interpolation also prevents hundreds of thousands of independent
view-dependent color outlets.

## 0–9k Gate

The first run stops at 9k. The original Panda defect is visible by 3k/6k, and
guide length, width, and direction are still frozen through 9k. Checkpoints and
full-resolution view-09 renders are required at 3k, 6k, and 9k using bbox
`[835,149]-[1199,304]` for the fixed user-reported region.

Acceptance requires:

1. support-off forward/state behavior remains exact;
2. the trusted confidence identity and gradient-ownership tests pass;
3. the 30-view Panda 3k/6k/9k crop is materially closer to the clean
   single-view control and contains no new blotch, stripe drag, or opacity hole;
4. root count, lifecycle events, memory, and checkpoint reload remain valid;
5. SH coefficients remain bounded and nonsaturated;
6. no 30k run is authorized from PSNR alone;
7. a matched white-tiger regression is required before promotion.

## Configuration

`configs/r071_guide_view_sh_0_9k_gate.env`

The candidate changes only guide-view SH support/scale/LR and the diagnostic
iteration/checkpoint horizon. `GUIDE_SUPPORT_GAUGE_WEIGHT=0` preserves the R068
parent rather than inheriting the rejected R069 loss.
