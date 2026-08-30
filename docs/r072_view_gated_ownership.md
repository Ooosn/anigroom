# R072 Per-View Trusted Ownership

Status date: 2026-08-30.

Status: rejected after a completed Panda 0-9k H100 gate. The implementation,
preflight, checkpoints, and renders are valid; the ownership magnitude is not.

## Formal Result

The source at `29e7eab656b62aafb99dde65a04405c7d9feb372` passes all
`390` local tests. The full-resolution H100 preflight and uninterrupted 0-9k
run complete without OOM, CUDA error, nonfinite state, or fallback.

| iteration | R068+V7 view09 composite | R072 | delta |
|---:|---:|---:|---:|
| 3k | `17.81787` | `17.29291` | `-0.52496 dB` |
| 6k | `19.70916` | `18.92481` | `-0.78435 dB` |
| 9k | `21.00030` | `20.08535` | `-0.91494 dB` |

At 9k R072 has `608,775` roots versus the parent's `669,143`; the final
lifecycle event selects `2,260` parents versus `3,017`. The root-move loss is
about half the parent value, fixed-RGB L1 is worse (`0.09635` versus
`0.08998`), and mask L1 is worse (`0.04041` versus `0.03609`). The fixed crop
remains granular and becomes more weakly supported rather than approaching the
clean single-view control.

The decisive measurement is gradient mass. Only `21.20%` of training
view/guide entries are owned and the mean gate is `0.06993`. R072 therefore
does not merely select trusted views; it reduces the expected placement and
opacity gradient by about fourteen-fold. It starves trusted owners along with
rejecting untrusted ones. This result rejects the raw confidence magnitude as
the optimizer gate, but it does not reject the trusted-view support set.

The next isolated candidate must separate selection from strength: preserve
the same nonzero trusted owner set while normalizing each guide's owner weights
to conserve its expected gradient budget under uniform 30-view sampling. No
appearance field, floor, species rule, or new confidence source is mixed into
that experiment.

Formal artifacts:

- HGC runtime: `/home/wangyy/panda-r072-view-gate-runtime-20260830`;
- 9k checkpoint SHA-256:
  `aa060a1fe7dc0022ba4c7c2d105801ee1f4dcc2f52ca2d22c6f69e3d9581f1b6`;
- training log SHA-256:
  `8976bffb9020828975b0eb163944a2755413dbe2d848e50510502edf9698ffa8`;
- local crops:
  `D:/RTS/_tmp/panda_r072_view_gate_acceptance_20260830`.

The retained first preflight attempt completed its real one-step model work
and failed only a wrapper post-assertion (`AssertionError: 1`); the corrected
preflight then passed before the formal run. No 30k continuation is authorized.

## Question

Three candidates have now failed to repair the Panda coat, and each failure
narrowed the cause rather than the symptom:

- R069 fixed guide length and width at 12k but was rejected because its shared
  slenderness gradient created a long-hair escape;
- R070 was cancelled before training when exact-crop review showed the visible
  failure is salt-and-pepper raw-fur RGB noise with continuous alpha, not the
  scalar length field;
- R071 added a bounded guide-owned view-dependent SH appearance field. It
  gained only `+0.145/+0.210/+0.288 dB` at 3k/6k/9k while its own coefficients
  saturated from `50.2%` to `71.7%`, and the render-root population barely
  moved (`668,060` against the parent's `669,143`). The appearance outlet
  absorbed error without repairing the root distribution.

Meanwhile the matched single-view control established the positive result:
trained from zero on view 09 alone, with V7, root count, resolution, losses,
and unlock schedule unchanged, the same representation produces a materially
more continuous coat by 6k.

That leaves one untested hypothesis. Before 9k the only parameters that move
the image are render-root placement, root/tip opacity, and the 600-9000 root
lifecycle. Thirty views all claim those degrees of freedom everywhere, so a
view with no reliable evidence for a region still moves roots there. R072 asks
whether restricting that claim to trusted evidence repairs the distribution.

## Single Method Change

R072 inherits exact R068 behavior with both rejected candidates explicitly
disabled (`GUIDE_SUPPORT_GAUGE_WEIGHT=0`, `GUIDE_VIEW_SH_SUPPORT=0`). It adds
one gradient-ownership rule and no new loss, schedule, or parameter.

For each training view the accepted V7 trusted-view evidence is interpolated
onto render roots through the existing K8 primary-guide surface support, and
the resulting share in `[0, 1]` gates the gradient of:

- the strand roots that enter Gaussian construction;
- the copy of the root position that drives guide surface-support weights;
- decoded root and tip opacity;
- the per-view `pixel_to_root` densification residual.

Every gate is straight-through, `x.detach() + c * (x - x.detach())`, so the
forward value is bit-identical and a unit gate reproduces the parent run
exactly. Any measured difference is therefore attributable to ownership alone.

## What Is Deliberately Not Gated

- **Appearance.** Root/tip color and every RGB path keep full ownership.
  R071 already showed appearance is not the lever, and gating it would confound
  this experiment with that one.
- **Visibility and opacity history.** `visible_count`, `gaussian_sample_count`,
  and `opacity_sum` stay ungated. A root genuinely is visible in a view
  regardless of whether that view's flow direction is trusted, and gating those
  counters would corrupt the lifecycle's visibility-history criterion.
- **The mesh no-penetration constraint** and the returned `roots_local`. These
  are view-independent geometry. Leaving them ungated is what lets the surface
  regularizers propagate corrections into roots a given view may not move,
  which is the interpolation half of the design.

The gradient-derived lifecycle evidence (`root_grad_abs_sum` and the Gaussian
gradient sums) needs no separate rule: those gradients flow through the same
gate, so ownership propagates into them automatically.

## Trusted View Ownership

`axis_view_cluster_selected_direct_weight[V, G]` in the accepted V7 target is
the only source of ownership, with view identity taken from the sibling
`summary.json` and guide face/barycentric identity validated exactly. No image
box, species rule, body label, or hard-coded view list is introduced.

Measured coverage of the Panda target, normalized by its positive q95:

| quantity | Panda | White |
|---|---:|---:|
| positive entries | `22.48%` | `25.02%` |
| guides with at least one view | `89.18%` | `93.76%` |
| mean owner views per guide | `7.418` | `8.258` |
| guides with an owner above `0.5` | `68.11%` | `62.36%` |

The 36 cameras split by fixed stride 6 into 30 train and 6 test views, and the
V7 protocol excludes views `4`, `24`, and `25`. Views `4` and `25` are
therefore training views with no trusted evidence; view `24` is a test view and
does not matter. Under the single-variable floor of `0.0` those two views still
render forward and still supervise appearance, but claim no placement,
opacity, or lifecycle ownership. This is a consequence of the accepted flow
protocol, not a new exclusion, and the run records it in
`view_gate_report.json`.

## Known Limitation

V7 confidence measures direction observability: GPT flow agreement, projection
conditioning, and visibility at annotation time. It does not measure training
time occlusion, silhouette blending, view-dependent shading, or Gaussian
coverage. The complete ownership term would be a product of several
confidences. R072 deliberately uses the derived V7 term alone, because it is
already accepted, identity-validated, and species-independent, and because
adding estimated terms in the same run would make the result unattributable.
If R072 succeeds, the remaining factors are the natural follow-up; if it fails
for coverage reasons, `view_gate_report.json` and the floor are the evidence
that says so.

## 0-9k Gate

`configs/r072_view_gated_ownership_0_9k_gate.env`

Checkpoints and full-resolution view-09 renders are required at 3k, 6k, and 9k
using bbox `[835,149]-[1199,304]`, the same fixed user-reported region used for
R071 and the single-view control.

Acceptance requires:

1. support-off forward and gradient behavior remains exact;
2. a unit gate reproduces the ungated gradients exactly;
3. the 30-view Panda 3k/6k/9k crop moves materially toward the clean
   single-view control, with no new blotch, stripe drag, or opacity hole;
4. root count, lifecycle events, memory, and checkpoint reload remain valid;
5. the trusted-view report matches the measured coverage above;
6. no 30k run is authorized from PSNR alone;
7. a matched white-tiger regression is required before promotion.

Rejection is expected to be reported as plainly as R069, R070, and R071 were.

## Local Validation

- module contract: `tests/test_view_gated_ownership.py`, 12 passed
- Stage-1 integration: `tests/test_view_gated_ownership_stage1.py`, 12 passed
- full suite: `390 passed`
