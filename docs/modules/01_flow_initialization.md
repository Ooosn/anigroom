# Module 1: Flow / Initialization

## Purpose

This module turns image-space fur direction evidence into stable mesh-rooted
3D initialization targets. It owns the initial flow field and guide-root target
quality, not the training schedule.

## Current Formal Target

Current accepted line:

`D:\petsgaussianhair\_downloads\tiger_hair_flow_36\shell_fused_smal_head500_body4000_candidate65536_headk24_bodyk12_v4_surface_direction`

Current generation settings:

- Mesh alignment: `scale=1.28`, `translation=[0, 0.32, 0.02]`
- Excluded views: `4`, `24`, `25`
- Dense candidates: `65536`
- Guide roots: `500` head roots, `4000` body roots
- Clean K: `24` for head, `12` for body
- Observed roots: `4407 / 4500`
- Direction consensus anchors: `552`
- Direction-only change from `v3_height_smooth`:
  - guide-root direction cleaning uses a mesh-geodesic neighborhood;
  - neighboring 3D directions are parallel-transported before comparison or
    averaging;
  - guide-to-render direction interpolation uses source/query normals and the
    same parallel transport instead of unsigned sign alignment.
- Root positions, normals, barycentrics, shell height, local spacing,
  confidence, visibility and region labels are bitwise identical to
  `v3_height_smooth`.
- Confidence split:
  - field confidence answers whether a root has enough evidence for dense
    initialization/interpolation;
  - anchor confidence answers whether a root direction is reliable enough for
    strong anchor supervision.

Retained baselines:

`D:\petsgaussianhair\_downloads\tiger_hair_flow_36\shell_fused_smal_head500_body4000_candidate65536_headk24_bodyk12_v3_height_smooth`

`D:\petsgaussianhair\_downloads\tiger_hair_flow_36\shell_fused_smal_head500_body4000_candidate65536_headk24_bodyk12_v2_consensus`

Keep `v3_height_smooth` and `v2_consensus` for controlled experiments and
ablations. Neither is the current training target.

Rejected diagnostic: sign-aware direction consensus improved unsigned local
axis smoothness but introduced directed sign flips in the tail/body field, so it
must not be used as a formal target-generation path.

## Formal Code

- `anigroom/flow/clean_flow.py`
  - Loads clean-flow targets.
  - Samples clean-flow directions to arbitrary roots.
  - Converts 3D directions to local groom controls.
  - Provides 3D anchor and smoothness losses.
  - Keeps field confidence and anchor confidence separate: initialization and
    length use the dense cleaned field confidence, while anchor losses use
    `direction_anchor_confidence`.
  - Keeps scalar interpolation unchanged from v3.
  - Interpolates directed 3D flow with normal-compatible neighbors and
    parallel transport into each query root's surface frame.
- `anigroom/flow/surface_graph.py`
  - Builds intrinsic guide-root neighborhoods through the triangle mesh.
  - Does not connect roots merely because they are close in Euclidean space.
- `anigroom/flow/direction_geometry.py`
  - Implements the minimal-rotation parallel transport shared by target
    generation and runtime sampling.
- `anigroom/projection/mesh_visibility.py`
  - Mesh depth rendering and visibility-aware projection utilities.
- `tools/build_white_tiger_smal_head_guides.py`
  - Builds SMAL/anatomy-aware head/body guide roots.
- `tools/fuse_gpt_flow_shell_multiview.py`
  - Fuses multi-view GPT/Gabor flow evidence with normal-shell lift and
    direction consensus.
- `tools/visualize_flow_targets_as_strands.py`
  - Canonical strand-like visualization for flow targets.

## Diagnostic Or Historical Code

These files can be useful for analysis but must not silently replace the formal
target:

- `tools/fuse_gpt_flow_multiview.py`
- `tools/postprocess_directional_white_tiger.py`
- `tools/directional_field/`
- older `_downloads/tiger_hair_flow_36/shell_fused_*` targets

## Acceptance Evidence

Before using a new target in training, produce:

- Full-resolution direction arrows for the target view.
- Full-resolution strand target visualization.
- At least one head/front view and one side/body view.
- A short note explaining what changed from the previous accepted target.

Current `v4_surface_direction` acceptance notes:

- The reported visual defect is a real local crossing problem, not primarily a
  root-to-tip sign error. On v3 guides, local neighbor axes differ by at least
  45 degrees on `4.33%` of edges, while true sign reversal is only `0.11%`.
- V3 local crossing correlates with Euclidean neighbors from incompatible
  surface regions. It is stronger near silhouettes/high-curvature regions and
  in the head, but is not exclusive to the silhouette.
- Mesh-geodesic cleaning reduces mean guide crossing from `0.03838` to
  `0.02062` and 45-degree edge conflicts from `4.33%` to `2.28%`.
- Replacing only the guide target is insufficient: v4 with the old Euclidean
  guide-to-render interpolation is worse than v3. The target and runtime
  interpolation therefore form one atomic method change.
- With normal-compatible, parallel-transported interpolation, the 100k-root
  crossing score falls from v3's `0.00261` to `0.00183`, 45-degree conflicts
  fall from `0.112%` to `0.049%`, and sign reversals are zero in the probe.
- The formal PyTorch sampler matches the independent NumPy probe within
  `0.0023` degrees mean axis angle and uses about `127 MB` peak allocated CUDA
  memory for the full 100k-root regression.
- Evidence-weighted agreement with the original multi-view flow changes by only
  `-0.35` percentage points, so the improvement is not obtained by flattening
  the field away from image evidence.
- Regression evidence is stored under
  `_diagnostics/flow_direction_continuity_v3_v11_20260715`.

## Boundary

This module may output target points, 3D directions, confidence, observed masks,
anchor masks, and lift/lambda proxies. It must not decide densification timing,
residual unlock, PSNR reporting, or training-stage schedule.
