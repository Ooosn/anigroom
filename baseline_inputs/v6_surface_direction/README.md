# V6 trusted tangent + guarded multiview ratio target

Status: accepted white-tiger initialization baseline. A matched Panda V6 target
is accepted for the next cross-sample run. This does not replace the completed
R068+V5 training checkpoint.

## Artifact

- file: `guide_flow3d_shell_targets_exclude_004_024_025.npz`
- SHA-256:
  `29d07139d6214cf9540e814a8f872128ad29999890221e8afe0b2c5599586dd1`
- roots / observed roots: `4500 / 4407`
- views: `33`, excluding `4`, `24`, `25`
- mesh alignment: `scale=1.28`, `translation=[0, 0.32, 0.02]`
- neighborhood: mesh-geodesic, body K12 / head K24

## Method

V6 keeps V5 directional observability, then:

1. selects a trusted tangent axis from robust multiview contribution clusters;
2. keeps only q95 cluster-margin switches and direct-supermajority residual
   propagation;
3. freezes that tangent axis;
4. solves the nonnegative normal/tangent ratio by weighted multiview linear
   least squares at the final shell point;
5. accepts a per-root ratio update only when it improves both direct 2D
   residual and mesh-neighborhood maximum jump;
6. supersedes the old whole-vector final consensus for this mode.

The method has no species, semantic region, root index, image coordinate, or
view-index rule.

## Acceptance

- Panda view-27 shoulder local jump: `29.70 -> 24.77 deg`
- Panda view-27 upper-back local jump: `46.43 -> 29.22 deg`
- Panda global view-27 mismatch: `23.80 -> 20.55 deg`
- Panda direct all-view mean: `16.11 -> 12.78 deg`
- White-tiger global view-27 mismatch: `24.01 -> 22.97 deg`
- White-tiger direct all-view mean: `14.32 -> 13.55 deg`
- guarded LS ratio updates: Panda `434`, white tiger `513`
- complete numeric arrays: finite
- canonical visual gate: Panda and white tiger views `00`, `09`, `18`, `27`
  passed without a new visible discontinuity
- full local test suite: `313 passed`

The matched formal Panda target SHA-256 is
`b3f49317dbf9d09a2d3981dc02b48cf4dff5e67b19f900efbf0268ac270d8e29`.
Formal HGC evidence is under
`/home/wangyy/anigroom-flow-trusted-20260827` and local analysis under
`D:/RTS/_tmp/formal_trusted_flow_fixed_ratio_results_20260827`.
