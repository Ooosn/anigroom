# Panda V5 Multiview Discontinuity Investigation

Status: causal attribution and a bounded experimental implementation are
complete. The accepted V5 target is unchanged. The experimental
`trusted-view-cluster` axis mode is opt-in; `anchor-propagated` remains the
formal default.

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
| All-view direct mismatch | 15.11 deg | 15.16 deg |

The matched white-tiger diagnostic changes global view-27 mismatch by
`+0.139 deg`, all-view direct mean by `+0.146 deg`, and highest-trust-quintile
median direction by `0.371 deg`. The final residual stage accepts only two
white-tiger roots.

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

## Decision

- Keep the accepted V5 target and `anchor-propagated` default unchanged.
- Keep `trusted-view-cluster` as an explicit experimental mode with synthetic,
  permutation, sign-invariance, zero-evidence, and integration tests.
- Do not train from any target produced in this investigation.
- The next method question is downstream ownership: continuous lift and final
  direction consensus must be made conditional on trusted tangent evidence
  without globally flattening white-tiger boundaries.

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
- Rejected formal contribution-cluster target SHA-256:
  `25f6c23b081c631354890c06c4e7e801fae3702ee456e9788d04a9849f507945`
- Rejected matched white target SHA-256:
  `b0a05fe66b31322fc8db6ebb262b1935c0de58e281f1e5e94ea9541160550b63`

Both held HGC qlogin allocations were preserved. No training process was
started and no accepted target or checkpoint was overwritten.
