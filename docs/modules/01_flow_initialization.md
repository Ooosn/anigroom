# Module 1: Flow / Initialization

## Purpose

This module turns image-space fur direction evidence into stable mesh-rooted
3D initialization targets. It owns the initial flow field and guide-root target
quality, not the training schedule.

## Current Formal Target

Current accepted line:

`baseline_inputs/v5_surface_direction/guide_flow3d_shell_targets_exclude_004_024_025.npz`

Current generation settings:

- Mesh alignment: `scale=1.28`, `translation=[0, 0.32, 0.02]`
- Excluded views: `4`, `24`, `25`
- Dense candidates: `65536`
- Guide roots: `500` head roots, `4000` body roots
- Clean K: `24` for head, `12` for body
- Observed roots: `4407 / 4500`
- Direction consensus anchors: `563`
- V5 inherits the V4 direction changes from `v3_height_smooth`:
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
- V5 correction:
  - initial tangent-axis fusion uses parameter-free directional observability;
  - unsupported edge-on projections no longer contribute false tangent axes;
  - later normal/lift fitting, sign, consensus, and shell behavior are unchanged.

Retained baselines:

`baseline_inputs/v4_surface_direction/guide_flow3d_shell_targets_exclude_004_024_025.npz`

`D:\petsgaussianhair\_downloads\tiger_hair_flow_36\shell_fused_smal_head500_body4000_candidate65536_headk24_bodyk12_v3_height_smooth`

`D:\petsgaussianhair\_downloads\tiger_hair_flow_36\shell_fused_smal_head500_body4000_candidate65536_headk24_bodyk12_v2_consensus`

Keep V4, `v3_height_smooth`, and `v2_consensus` for controlled experiments and
ablations. None is the current training target.

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
  - Fuses multi-view GPT/Gabor flow evidence with normal-shell outward
    evidence and direction consensus.
- `tools/fuse_gpt_flow_multiview.py`
  - Provides the shared image-flow parsing, graph cleaning, and canonical
    overlay primitives used by the shell fusion entry point.
  - Its older direct-fusion CLI is not the accepted target generator, but the
    module itself is a required dependency of the accepted shell route.
- `tools/visualize_flow_targets_as_strands.py`
  - Canonical strand-like visualization for flow targets.

## Diagnostic Or Historical Code

These files can be useful for analysis but must not silently replace the formal
target:

- `tools/postprocess_directional_white_tiger.py`
- `tools/directional_field/`
- older `_downloads/tiger_hair_flow_36/shell_fused_*` targets

## Source Recovery And Cross-Sample Validation

The four target-generation entry points were absent from the later packaged
working tree even though this document still named them as formal code. The
loss was packaging contamination, not an intentional module deletion:

- the accepted V4 tools existed as untracked files beside source commit
  `09de8c4`;
- a later dirty trainer refactor deleted `load_image` and `load_mask` without
  committing that deletion;
- the handoff copied the dirty trainer together with the untracked V4 tools,
  so the package contained incompatible source generations;
- the old manifest checked file hashes but did not import the complete V4
  entry point, allowing the mismatch through.

The formal tools are now restored byte-for-byte from the accepted V4 source.
Their SHA-256 values are:

- `build_white_tiger_smal_head_guides.py`:
  `623a49a56da23ffced89a254f8d2179c48925d192be8ea90820daf017c6f1986`
- `fuse_gpt_flow_multiview.py`:
  `046a964a1ffb62cfd62e96aee62b68ed32744f20cd2588a411b591af33bcfc3e`
- `fuse_gpt_flow_shell_multiview.py`:
  `7d3e111311c6c850fb40faed47121d9deebc333b91f4a1a7cc0767c0fd0c3965`
- `visualize_flow_targets_as_strands.py`:
  `a5906af7963e7fa8681d442d306a00bb14afdc5b21a48b1415b94a343ddfff13`

The exact white-tiger V4 contract was also run on the Panda sample with only
dataset paths and identity mesh alignment changed. It produced 4500 roots,
used the same 33 views with exclusions `4,24,25`, observed 4194 roots
(`93.2%`), retained 504 direction anchors, and passed finite-array and
single-component mesh-geodesic graph checks. The frozen local bundle is under
`D:\RTS\datasets\panda_v5_flow_protocol_20260827\run_output`.

### Directional observability correction

The initial V4 axis lift used image confidence, mesh visibility, and a
normal-to-view weight, but still forced every accepted 2D axis through a
pseudoinverse of the projected tangent basis. Near an edge-on tangent plane,
that inverse can be directionally ill-conditioned even when the surface-level
view weight is nonzero. It can therefore turn a poorly observed screen axis
into a confident but incorrect 3D tangent.

The corrected lift multiplies only the initial tangent-axis evidence by a
parameter-free directional observability. For projected tangent basis `B` and
screen axis `o`, it measures both how much of `o` is reproducible by `B` and
how strongly the recovered coefficient direction projects relative to the
largest singular direction of `B`. Fully observable directions retain weight
one; rank-deficient unsupported directions receive weight zero. Later
normal/lift fitting, sign cleaning, consensus, shell settings, and all CLI
parameters remain unchanged.

This is a camera/mesh/axis condition, with no species, body region, view index,
or image-coordinate rule. Synthetic rank tests cover identity, supported and
unsupported rank-one projections, and anisotropic projection. Complete Panda
and white-tiger reruns preserve root populations and pass the canonical visual
gate. White tiger keeps `4407` observed roots, while its median direction change
is `0.99` degrees. The Panda V5 target is:

`D:\RTS\datasets\panda_v5_flow_protocol_20260827\run_output\output\v5_surface_direction\guide_flow3d_shell_targets_exclude_004_024_025.npz`

SHA-256:
`0c4705fbab50e4d9ed86aae2376ac71977f4bafff46f68ed643de25eaa333455`

## Acceptance Evidence

Before using a new target in training, produce:

- Full-resolution direction arrows for the target view.
- Full-resolution strand target visualization.
- At least one head/front view and one side/body view.
- A short note explaining what changed from the previous accepted target.

V4 parent acceptance notes:

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
anchor masks, and shell-height/lambda evidence. It must not decide densification timing,
residual unlock, PSNR reporting, or training-stage schedule.
