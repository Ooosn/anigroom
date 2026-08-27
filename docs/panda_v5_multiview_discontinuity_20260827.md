# Panda V5 Multiview Discontinuity Investigation

Status: the unsigned tangent-axis defect was resolved by V6. A later directed
audit reopened arrow-head continuity and is resolved by the V7 canonical
global-direction target. The completed R068+V5 checkpoint remains the trained
rollback; V6 is the fixed-axis parent and V7 is the accepted initialization
input for the next Panda cross-sample run.

## Reported Defect

Panda view 27 contains two serious local initialization discontinuities:

- the black shoulder band;
- the upper-back color boundary.

The source view-27 line drawing and sampled 2D orientation are smooth at both
locations. Stage tracing shows that the conflicts already exist in
`raw_flow3d` and persist in `flow3d`, before directed sign cleaning, continuous
lift fitting, consensus, training, or curl. The defect is therefore introduced
by initial multiview 2D-to-3D axis fusion.

## Exact Attribution

The exact leave-one-view-out replay identifies real view competition rather
than a silhouette-only or training defect. In the shoulder region, views 23,
21, 22, 20, and 19 rotate the fused axis away from smooth view-27 evidence.
The leading effects are:

| View | Effective share | View-27 mismatch delta | Exact LOO rotation |
| ---: | ---: | ---: | ---: |
| 23 | 17.43% | +6.50 deg | 11.71 deg |
| 21 | 11.01% | +4.74 deg | 8.67 deg |
| 22 | 11.65% | +3.99 deg | 6.61 deg |

Direct evidence supplies `99.37%` of the shoulder weight; silhouette-band
evidence supplies only `0.63%`. The existing V5 directional-observability term
retains almost all of the harmful evidence: shoulder view 23 has an
effective/raw weight ratio of `0.961`. Directional conditioning alone therefore
does not identify this cross-view conflict.

The full Panda 3DGS checkpoint export confirms a downstream orientation
competition rather than simple depth overlap. Major-axis mismatch to nearest
raw flow has median `21.87 deg` in the shoulder versus `6.64 deg` in a body
control. Conflicted screen-bin fraction is `3.20%` versus `0.60%`, while median
near/far depth gap is effectively unchanged (`0.509` versus `0.504`).

## Experiments

All experiments used the same parameters for Panda and white tiger. No species,
body-region, root-index, image-coordinate, or view-index rule was allowed.

Broad confidence-weighted propagation repairs Panda but over-smooths white
tiger. Its Panda shoulder mismatch changes `52.00 -> 27.59 deg`, but white
view-27 mismatch changes `23.98 -> 25.86 deg`. Local-max-only propagation is
more aggressive and degrades white further.

Relative-confidence competition reveals the tradeoff:

- a permissive relative-confidence arm repairs Panda shoulder to
  `32.58/29.67 deg` mismatch/local jump, but degrades white view 27 by
  `+0.99 deg`;
- a strict neighbor-dominance arm preserves white (`+0.09 deg`) but leaves the
  Panda shoulder at `42.14/53.86 deg`.

The successful tangent-axis diagnostic uses a robust local view cluster, keeps
only the top `5%` hard cluster margins, then repairs residual q95 outliers only
when at least two thirds of direct view weight supports the propagated axis
within `30 deg`. It produces:

| Metric | Panda raw | Diagnostic candidate |
| --- | ---: | ---: |
| Shoulder view-27 mismatch | 52.00 deg | 25.92 deg |
| Shoulder local jump | 72.62 deg | 29.78 deg |
| Upper-back view-27 mismatch | 53.27 deg | 47.91 deg |
| Upper-back local jump | 37.52 deg | 22.28 deg |
| Global view-27 mismatch | 23.95 deg | 23.40 deg |
| All-view direct mismatch | 16.26 deg | 15.78 deg |

This tangent-axis result is necessary but not sufficient: the first formal
replay showed that legacy downstream whole-vector consensus could overwrite
it.

## Formal End-to-End Finding

The trusted tangent axis itself transfers, but the legacy downstream direction
pipeline is not neutral to it. In the first formal HGC replay:

- Panda `flow3d` shoulder mismatch/local jump improves
  `44.16/52.00 -> 25.92/29.78 deg`;
- Panda `flow3d` upper-back mismatch/local jump improves
  `50.65/28.76 -> 47.91/22.28 deg`;
- white-tiger `flow3d` changes only `+0.14 deg` on view 27 and `+0.15 deg` on
  all-view direct mean.

After the existing continuous-ratio and 16-iteration direction-consensus
stages, however, white-tiger all-view direct mismatch changes
`15.38 -> 19.45 deg`, and Panda upper-back local jump changes
`46.43 -> 47.76 deg`. One upper-back root's final normal/tangent ratio is
amplified from about `0.37` before consensus to about `2.20` after consensus.
Thus the complete target does not pass cross-species acceptance even though the
new tangent axis does.

Pure shell-normalized axes, direct-first shell axes, final selected-shell direct
axes, confidence-gated consensus rollback, and one/two rounds of trusted-neighbor
max-edge propagation were also tested. None repaired the complete Panda
upper-back field while preserving white-tiger multiview evidence. These arms
are rejected rather than hidden behind Panda-specific thresholds.

## Fixed-axis multiview-ratio resolution

The final method separates ownership. The trusted view cluster owns the
unsigned tangent axis. The continuous direction stage supplies only an initial
tangent sign and outward normal/tangent ratio. At the final shell point, each
direct view supplies a one-dimensional linear constraint on that ratio:

`cross(o, p_axis + rho * p_normal) = A * rho + b = 0`.

Weighted least squares solves nonnegative `rho` with no upper cap. A root keeps
the LS update only when it improves both its direct multiview residual and its
maximum transported graph-edge jump; otherwise it retains the pre-consensus
ratio. The old whole-vector consensus is superseded in this mode and cannot
rotate the trusted tangent axis.

The formal Panda result keeps all `4194` observed roots, accepts `434` guarded
ratio updates, and produces:

- shoulder final mismatch/local jump: `26.12 / 24.77 deg`;
- upper-back final mismatch/local jump: `49.18 / 29.22 deg`;
- global view-27 mismatch: `23.80 -> 20.55 deg`;
- corrected direct all-view mean: `16.11 -> 12.78 deg`.

The matched white-tiger run keeps all `4407` observed roots, accepts `513`
ratio updates, improves view-27 mismatch `24.01 -> 22.97 deg`, and improves
corrected direct all-view mean `14.32 -> 13.55 deg`. Panda and white-tiger
canonical views `00`, `09`, `18`, and `27` pass visual inspection without a new
visible discontinuity. All numeric arrays are finite.

## Decision

- Accept V7 as the initialization target for the next Panda cross-sample run.
- Keep V6 as the exact fixed-axis parent and V5 plus its completed R068
  checkpoint as the exact trained rollback.
- Keep `trusted-view-cluster` as the formal tangent-axis owner, then apply the
  canonical global sign solver and fixed-sign directed ratio refit.
- Do not claim a training improvement until a new from-zero run completes.

V6's prior acceptance is superseded because axial `abs(dot)` metrics hid
directed reversals. V7 formal Panda/white targets introduce zero severe edges,
pass fixed views `00/09/18/27`, and pass the matched 64-step graph-streamline
audit. See `docs/v7_global_directed_flow.md`.

## Evidence

- Exact attribution:
  `D:/RTS/_tmp/panda_v5_view27_attribution_20260827`
- Standalone candidate sweep:
  `D:/RTS/_tmp/flow_trust_view_cluster_20260827`
- Formal contribution-cluster replay:
  `D:/RTS/_tmp/formal_trusted_flow_results_20260827`
- Formal pure/direct/selected-point controls:
  `D:/RTS/_tmp/formal_trusted_flow_pure_axis_results_20260827`,
  `D:/RTS/_tmp/formal_trusted_flow_direct_first_results_20260827`, and
  `D:/RTS/_tmp/formal_trusted_flow_selected_point_results_20260827`
- Formal HGC runtime:
  `/home/wangyy/anigroom-flow-trusted-20260827`
- Accepted V6 Panda target SHA-256:
  `b3f49317dbf9d09a2d3981dc02b48cf4dff5e67b19f900efbf0268ac270d8e29`
- Matched white-tiger target SHA-256:
  `29d07139d6214cf9540e814a8f872128ad29999890221e8afe0b2c5599586dd1`
- Accepted V7 Panda target SHA-256:
  `6a220f52b15ca996c88e71802d3309f9499ade442f79dc72300f1af12b5fa56f`
- Accepted V7 white-tiger target SHA-256:
  `f009af820560adf19b6eedbb8bf2c5d29df00cca576be13161b4ee2ebaed6510`
- V7 formal runtime:
  `/home/wangyy/anigroom-global-directed-v7-20260828`
- V7 local acceptance:
  `D:/RTS/_tmp/anigroom_v7_formal_results_20260828`
- V7 streamline audit:
  `D:/RTS/_tmp/anigroom_v7_streamline_audit_20260828`
- Accepted formal analysis:
  `D:/RTS/_tmp/formal_trusted_flow_fixed_ratio_results_20260827`
- Rejected formal contribution-cluster target SHA-256:
  `25f6c23b081c631354890c06c4e7e801fae3702ee456e9788d04a9849f507945`
- Rejected matched white target SHA-256:
  `b0a05fe66b31322fc8db6ebb262b1935c0de58e281f1e5e94ea9541160550b63`

Both held HGC qlogin allocations were preserved. No training process was
started and no prior target or checkpoint was overwritten.
