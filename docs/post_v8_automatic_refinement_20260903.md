# Post-V8 Automatic Multiview Refinement

Status date: 2026-09-03.

Status: fixed-input Panda and white-tiger diagnostics pass. The candidate is
not yet a training baseline; the formal V8 targets remain immutable parents.

## Purpose

V8 ends after propagating a jointly reliable complete 3D direction field. It
does not revisit the original per-view evidence afterward. This experiment
alternates two existing owners until an automatic fixed-point gate stops:

1. refit only the tangent-plane angle against the exact formal per-view V7
   evidence matrix;
2. run the accepted V8 joint-confidence directed propagation on that field.

No V6 axis cluster, V7 global sign solve, or V7 ratio fit is recomputed.

## Fixed Confidence Contract

The BA input is the formal `[V,N]` evidence saved in the V8 target:

- vectors: `axis_view_cluster_selected_direct_vectors`;
- weights: `axis_view_cluster_selected_direct_weight`.

For both samples the shapes are `[33,4500,3]` and `[33,4500]`. The weights
already include image confidence, visibility, view-angle weighting,
directional observability, the observed mask, and the existing confidence
floor. Panda/white contain `33382/37162` positive view-root pairs.

Propagation reuses the formal V8 `[N]` root reliability factors without
recomputing them:

`axis_view_cluster_final_confidence`
`* axis_view_cluster_global_unary_normalized_margin`
`* axis_view_cluster_global_unary_vote_coherence`.

The reconstructed product matches the stored
`axis_view_cluster_confidence_flow_joint_confidence` within
`1.49e-8/2.98e-8` for Panda/white.

## Automatic Algorithm

Each BA solve starts from the current complete direction, keeps its current
normal component fixed, and optimizes one gnomonic tangent-angle coordinate
per guide. Zero-weight roots receive zero data gradient. The loss is the
formal confidence-weighted axial reprojection residual over every contributing
view.

The unchanged formal
`refine_confidence_guided_directed_flow` pass then propagates the complete
candidate direction using the frozen V8 reliability factors and exact stored
postratio graph.

One outer cycle is accepted only when both quantities are nonincreasing and at
least one improves strictly:

- confidence-weighted multiview residual;
- parallel-transported surface connection energy `mean(1-dot)`.

The BA angle step is backtracked by powers of two until the complete
BA-plus-propagation cycle satisfies that Pareto gate. If no scale passes, the
algorithm stops and retains the last accepted field. A second fixed-point
stop also checks relative energy improvement and P95 direction change. There
is no species, region, selected-view, or image-coordinate rule.

## Fixed-Input Results

Both formal V8 inputs automatically accept exactly one cycle and reject cycle
two because every second-cycle candidate improves image fit only by worsening
the accepted surface energy.

| sample | accepted BA scale | multiview energy | relative gain | surface energy gain | V8-propagation changed roots | cycle-2 result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Panda | `0.25` | `0.151729 -> 0.135994` | `10.370%` | `5.727%` | `314` | rejected |
| white tiger | `0.0625` | `0.210385 -> 0.205110` | `2.507%` | `0.531%` | `39` | rejected |

Panda transported graph angle P95/P99 changes from `55.30/94.78` to
`52.17/87.26` degrees; edges above 90/135 degrees change `365/10 -> 254/4`.
The largest root rotations belong to zero-joint-confidence, usually
zero-view-evidence roots; they are filled by propagation rather than by BA.
The V8 propagation reports zero new severe edges for both samples.

Independent reruns reproduce the final direction NPY hashes exactly:

- Panda: `81a59b6cb56897841d71b9514d1f51d426b5fbba5286b0f5ec9802938b8c9bf4`;
- white tiger: `3512bfa7c01230511351f8868405a519e023fc2aab2ecb20a2f97785347f471a`.

## Implementation And Tests

- algorithm: `anigroom/flow/post_v8_refinement.py`;
- actual-input runner: `tools/diagnose_post_v8_refinement.py`;
- focused tests: `tests/test_post_v8_refinement.py`;
- existing canonical surface-map entry point extended only to accept one
  identity-locked external direction array:
  `tools/visualize_r085_direction_surface_maps.py`.

Focused V7/V8 and new tests pass `24/24`. The complete repository passes
`708` tests with the existing `14` Matplotlib dependency warnings.

## Artifacts

Panda:

- report:
  `D:/RTS/_tmp/panda_post_v8_refinement_20260903_attempt4_final/post_v8_refinement.json`,
  SHA-256 `79ef8107ff235f6c0da7cc1576abab38ddb5d63cf0851f57e1db3a1903a025f0`;
- candidate target:
  `D:/RTS/_tmp/panda_post_v8_refinement_20260903_attempt4_final/candidate_target.npz`,
  SHA-256 `d1917daabc538364ed6584b8c9e088ef71845adbd5afb4e3f0bbdd8e76eed69d`;
- manifest / reliable-runner result SHA-256:
  `c143911d966cd93e8677fab784860dd5a634615c03180c3792c02a32d6e74a14`
  / `76a68abaec6b1211396ad571fdcef10f1461b8aabf1d6c80e25b58e04bba36b4`.

White tiger:

- report:
  `D:/RTS/_tmp/white_post_v8_refinement_20260903_attempt3_final/post_v8_refinement.json`,
  SHA-256 `4db700848c86fc410559d084c84ded2c45038b7d64d96f8d79bf606c98762067`;
- candidate target:
  `D:/RTS/_tmp/white_post_v8_refinement_20260903_attempt3_final/candidate_target.npz`,
  SHA-256 `2c42886e15c988512521078e0bd837fe248944b293b69eacac2be3e98b865776`;
- manifest / reliable-runner result SHA-256:
  `16aac59290b56a49fa995f61b6748976b346b498e9f78a6d3ccb0d612bdf58a0`
  / `224ce42fdda8c35e6894b1627185ecb1920ff452bc2565cbcabb5ea85dc9eef6`.

Canonical Panda view09:

- image:
  `D:/RTS/_tmp/panda_post_v8_refinement_canonical_view09_20260903_final/view09_direction_surface_external.png`,
  SHA-256 `6053adcc7852b7beb89a45e6fda32312ff23209b503cb545dcc818730036698a`;
- report / manifest SHA-256:
  `3170237425af0c63440c4b22e1a798014833c82f4baade8e36254e4803811ead`
  / `f0368bffe9b74e4008f198ee3e6d9d91766e646eb4cfce619031b0dc4fddcfc6`;
- canonical view27 arrow overlay:
  `D:/RTS/_tmp/panda_post_v8_refinement_arrows_view27_20260903_final/direction/view27_shell_cleaned_3d_arrows_overlay.png`,
  SHA-256 `6102fa6d8efb06428fff94539edd269873d68d904995ed2ce26e42dcdb805a33`.

## Decision

The automatic BA-plus-V8-propagation idea is numerically viable on both
samples and self-stops after one accepted cycle. It may advance as a target
candidate after user visual review. It has not been integrated into formal
target generation and has not been used for training.
