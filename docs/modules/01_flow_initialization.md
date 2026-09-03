# Module 1: Flow / Initialization

## Purpose

This module turns image-space fur direction evidence into stable mesh-rooted
3D initialization targets. It owns the initial flow field and guide-root target
quality, not the training schedule.

## Current Formal Target

Current accepted line:

`baseline_inputs/v7_surface_direction/guide_flow3d_shell_targets_exclude_004_024_025.npz`

Current generation settings:

- Mesh alignment: `scale=1.28`, `translation=[0, 0.32, 0.02]`
- Excluded views: `4`, `24`, `25`
- Dense candidates: `65536`
- Guide roots: `500` head roots, `4000` body roots
- Clean K: `24` for head, `12` for body
- Observed roots: `4407 / 4500`
- Trusted tangent q95 roots / global sign changes / postratio updates:
  `221 / 62 / 490`
- V7 inherits the V6/V5/V4 direction changes from `v3_height_smooth`:
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
- V6 correction:
  - robust per-view contribution clusters own the tangent axis;
  - direct selected-shell evidence solves only the nonnegative normal/tangent
    ratio;
  - an LS ratio update must improve both direct residual and local graph jump;
  - old whole-vector final consensus is superseded in this mode.
- V7 correction:
  - exact final-ratio multiview evidence scores both tangent signs;
  - canonical trusted connected blocks solve the binary surface orientation;
  - root and view ordering cannot change the solution;
  - the final ratio is refit only after signs are fixed;
  - every ratio update uses signed parallel-transport angles and both passes
    fail hard if a new severe reverse edge appears.

Retained baselines:

`baseline_inputs/v6_surface_direction/guide_flow3d_shell_targets_exclude_004_024_025.npz`

`baseline_inputs/v5_surface_direction/guide_flow3d_shell_targets_exclude_004_024_025.npz`

`baseline_inputs/v4_surface_direction/guide_flow3d_shell_targets_exclude_004_024_025.npz`

`D:\petsgaussianhair\_downloads\tiger_hair_flow_36\shell_fused_smal_head500_body4000_candidate65536_headk24_bodyk12_v3_height_smooth`

`D:\petsgaussianhair\_downloads\tiger_hair_flow_36\shell_fused_smal_head500_body4000_candidate65536_headk24_bodyk12_v2_consensus`

Keep V6 as the immutable fixed-axis parent, V5 as the completed trained
rollback, and V4, `v3_height_smooth`, and `v2_consensus` for controlled
experiments and ablations. V7 is tied to the completed Panda R068 from-zero
30k cross-sample checkpoint documented in
`docs/panda_r068_v7_training_20260828.md`; that execution validates the V7
initialization route but does not by itself make the unchanged R068 learned
curl or localized coverage behavior a generalized cross-species baseline.

Rejected diagnostic: sign-aware direction consensus improved unsigned local
axis smoothness but introduced directed sign flips in the tail/body field, so it
must not be used as a formal target-generation path.

## Formal Code

- `anigroom/flow/global_sign_orientation.py`
  - Computes exact final-ratio multiview sign evidence.
  - Solves canonical trusted connected blocks over the transported graph.
  - Guarantees that no provisional non-severe edge becomes severe.

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

### V6 trusted tangent and guarded ratio

Panda view 27 later exposed two residual multiview-fusion defects in the black
shoulder band and upper-back boundary. Exact stage tracing places both in the
initial raw tangent axis, before sign cleaning, continuous lift, consensus,
training, or curl. The source 2D orientation is smooth; several accepted views
rotate the 3D fusion despite high directional observability.

The `trusted-view-cluster` axis mode records per-view additive
contributions, forms a robust axial view cluster, keeps only q95 cluster-margin
switches, and accepts residual surface propagation only with a two-thirds
direct-evidence supermajority. It is semantic-free and has Panda/white-tiger,
sign, view-order, zero-evidence, and integration tests.

The tangent axis is then frozen. Direct evidence at the final selected shell
point gives a weighted one-dimensional least-squares solve for the nonnegative
normal/tangent ratio. A ratio update is retained only if it improves both direct
multiview residual and local maximum graph jump. This supersedes old
whole-vector final consensus, which could overwrite a correct tangent axis.

The formal Panda and white-tiger reruns preserve observed populations and
improve direct multiview evidence. V6 is retained as the fixed-axis parent, but
its acceptance was reopened when directed metrics exposed arrow-head reversals
hidden by axial `abs(dot)` checks. V5 remains the trained rollback target. The
V6 Panda SHA-256 is
`b3f49317dbf9d09a2d3981dc02b48cf4dff5e67b19f900efbf0268ac270d8e29`.
See `docs/panda_v5_multiview_discontinuity_20260827.md`.

### V7 canonical global direction

V7 preserves the V6 unsigned tangent axis. At the provisional final ratio it
projects both tangent signs into every contributing view and forms one exact
cosine-squared sign unary. Intrinsic graph edges use parallel-transported
signed couplings. Edges that would become severely reversed under a
single-endpoint flip union their endpoints into a trusted block; best-gain
block updates then solve the global field. Exact float32 surface identity and
canonical float64 accumulation make the result invariant to root and view
ordering.

With signs fixed, direct multiview evidence recomputes an uncapped nonnegative
ratio. Updates are accepted in canonical evidence order only when direct
residual improves, no provisional non-severe edge becomes severe, and maximum
incident directed angle does not increase. Both global orientation and ratio
refit fail target generation if their zero-new-severe invariant fails.

Formal source commit
`0712587e2c32c621f5566b7a8706c9dc061fc85b` produces:

- Panda: `4194` observed roots, `112` global sign changes, `322` resolved
  severe edges, `527` postratio updates, zero new severe edges;
- white tiger: `4407` observed roots, `62` global sign changes, `174` resolved
  severe edges, `490` postratio updates, zero new severe edges.

The accepted white/Panda target SHA-256 values are respectively
`f009af820560adf19b6eedbb8bf2c5d29df00cca576be13161b4ee2ebaed6510`
and
`6a220f52b15ca996c88e71802d3309f9499ade442f79dc72300f1af12b5fa56f`.
Both pass fixed views `00/09/18/27`. The matched graph-streamline audit reduces
Panda/white selected severe transitions `113 -> 75` and `144 -> 113`; the
Panda view-27 crop becomes `13 -> 0` with two-cycles `2 -> 0`.
See `docs/v7_global_directed_flow.md`.

### V8 confidence-guided directed propagation

The R073 3k Panda asset exposed a second class of V7 failure: all conflicting
upper-back edges were already inside one protected global-sign supernode. A
binary block flip therefore could not repair the continuous 3D-axis seam. The
defect exists before training and curl, while the corresponding 2D source flow
is smooth.

V8 is an explicit opt-in pass after the guarded post-sign ratio refit. It forms
joint root reliability from trusted axial confidence, global unary margin, and
global vote coherence. A canonical max-confidence watershed propagates
parallel-transported directions from stronger roots into weaker roots. A basin
is accepted only with strict severe/negative/hinge improvement, zero newly
severe edges, and repair density above the sample's own graph defect density.
Accepted source roots are protected. A second local queue removes residual
negative edges only through strictly stronger decayed confidence and monotone
continuity improvement.

The method has no species, region, view-index, image-coordinate, or semantic
rule. It uses the existing trusted-view-cluster confidence decay and does not
introduce a confidence cutoff. Unobserved-only edges cannot initiate repair;
training already samples only observed clean-flow roots and reconstructs other
locations through the surface-aware interpolator.

On the frozen local Panda/white V7 targets, observed negative/severe edges move
`594/233 -> 92/5` and `1104/246 -> 202/40`, respectively, with zero new severe
edges. The user-marked Panda upper-back box moves from 87 total and 41
front-facing negative edges to zero. Formal target generation and fixed-view
acceptance remain required before V8 replaces V7 as a training input. See
`docs/panda_v8_confidence_guided_flow_20260830.md`.

### Earlier post-V8 angle/lift candidate

The earlier post-V8 tangent/angle-lift diagnostic did not pass the user-local
arrow visual. A local continuous stronger-neighbor propagation attempt improved
aggregate metrics but left top/right screen conflicts; its code was discarded.
Its sixth cycle was accepted and its seventh rejected despite `1039-1500`
locally eligible roots, establishing a local-greedy/global-gate deadlock. The
full attempt ledger is in `docs/post_v8_global_direction_field_attempt_20260903.md`.

### Post-V8 global direction-field candidate

The isolated follow-up starts from the formal V8 Panda/white parents
`5cb76945adb034e9666bfc98ae05647062d7ac4e3609e68162e561e4eebd54b1` and
`92a6d496aa39e85272f35668967f82d34df7f884681ade4e336c07256b47a3d7`. It is a
joint tangent-angle/log-lift global field solve on a parallel-transport graph.
The dimensionless objective is confidence-weighted axial reprojection data
divided by baseline plus `smooth_weight` times the mean normalized final
surface/tangent/lift connection energies. It adds an orientation barrier on
previously nonnegative/nonsevere edges and uses deterministic Adam with
powers-of-two backtracking. There is no species, region, view, image, or
protected-owner rule.

Acceptance requires nonincrease of data/surface/tangent/lift, edge
P95/P99/top-1%-CVaR/max, negative/severe edge+root counts, and zero newly bad
roots. The first no-barrier and edge-identity/root-support gate sweeps failed
cross-sample; the orientation-barrier sweep passed `10/10`. The selected shared
diagnostic is `smooth_weight=3`, orientation barrier `10`, chosen for the
strongest Panda correction while white passes every generic gate under the same
settings and automatic backtracking. This is not claimed to be the automatic
min-ranking winner.

The selected Panda/white data/surface/tangent/lift values are respectively
`0.151728958/0.081285164/0.092970185/0.047688868 ->
0.134471998/0.022716768/0.026532158/0.009225478` and
`0.210384637/0.102271296/0.111925289/0.035596021 ->
0.208973378/0.093914248/0.102144413/0.031667717`. Edge P95/P99 changes are
`55.3008/94.7821 -> 25.8670/44.9140` and
`60.1170/87.1834 -> 57.3577/83.7524`; negative/severe edges are
`365/10 -> 8/0` and `248/45 -> 230/35`; direction-change P95/max is
`44.788/68.655` and `3.467/7.857`, with zero newly bad roots.

The 71-root user QA improves incident-3D-max median/P95/max
`50.477/80.064/88.169 -> 15.955/44.632/53.527`, with `67` roots improving
more than `1°` and `4` worsening. However, `33` visible fixed-length screen
arrows remain mixed and the top group still appears reversed. V7 sign
reorientation flips `10` Panda roots but leaves the top cluster; it also worsens
white post-V8 lift `0.031668 -> 0.038818`, so the selected candidate is the
pre-reorientation global field. It remains a diagnostic target, not formal
training input, until the user/physical-asset visual gate passes. See
`docs/post_v8_global_direction_field_attempt_20260903.md` for paths, hashes,
deterministic reruns, visuals, literature basis, and isolated implementation
files.

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
