# R003 Attribute Interpolation

## Goal

R003 is the next controlled change after R002.  It applies the accepted V4
surface interpolation principle beyond clean-flow direction sampling, so groom
attributes inherited or sampled across guide/render roots use the same surface
neighborhood contract.

This is not a densification-policy experiment.  R003 must not change when roots
are inserted, how many roots are inserted, whether parents are deleted, pruning,
loss weights, training stages, child-strand count, renderer behavior, or the
flow target itself.

## Base

- Base experiment: R002.
- Flow target remains:
  `D:\petsgaussianhair\_downloads\tiger_hair_flow_36\shell_fused_smal_head500_body4000_candidate65536_headk24_bodyk12_v4_surface_direction`
- The accepted V4 clean-flow target and normal-compatible runtime interpolation
  are treated as one atomic baseline part.

## Single Variable

Replace the old Euclidean attribute interpolation/inheritance used outside the
clean-flow sampler with one unified surface-aware interpolation rule:

- Direction-like attributes are interpolated as 3D surface directions.  Source
  directions are parallel-transported from source normals to query normals
  before averaging or comparison.
- Scalar attributes such as length, width, opacity, lift, and residual scales
  use the same accepted surface-neighborhood weights, but without vector
  transport.
- Color attributes use the same neighborhood weights as scalar attributes.
  R003 does not add a new appearance model.
- Confidence and observed masks keep the current flow-module meaning: field
  confidence is for interpolation/initialization; anchor confidence is for
  strong direction supervision.

The expected effect is not merely cleaner code.  The intended behavioral fix is
that render roots and newborn roots inherit length, lift, width, color, and
shape controls from compatible surface neighborhoods instead of accidentally
mixing attributes from nearby but incompatible mesh regions.

## Current Densification Audit

The current baseline densification state is:

- Render-root densification is active from iteration 600 through 10000, every
  100 iterations.
- The active render-root route is `target_direct`: high residual pixels are
  projected through mesh depth onto mesh faces, then inserted as new render
  roots.
- Render-root parents are retained.  This is insertion/clone-style growth, not
  split-and-delete.
- Render-root attributes for inserted children are inherited from nearest old
  roots with a Euclidean KNN helper.  This is one of the R003 interpolation
  call sites.
- Guide-root densification is active from iteration 11000 through 16000, every
  200 iterations.
- Guide-root candidates are selected from render-root need interpolated to
  guide roots, capped at 32 new guides per event.
- Guide-root parents are also retained.
- Pruning exists in the code but is effectively disabled in the accepted
  baseline.
- Overlong/screen-footprint split paths can replace parents, but they are
  disabled in the accepted baseline.

Therefore the current active densification does not yet match the later
overlap-aware idea: "remove redundant parent roots after split/insertion".
That historical line must be recovered and audited separately before it becomes
an experiment.  It must not be folded into R003.

## Local Historical Evidence

The dirty local worktree `D:\petsgaussianhair` contains two relevant but
separate lines:

1. `anigroom/surface_interpolation.py` plus
   `docs/modules/04_surface_attribute_interpolation.md`.
   This is the relevant R003 evidence.  It defines topology-valid support,
   typed interpolation, parallel transport for directions, arithmetic
   interpolation for physical scalar/color values, and no special parent
   multiplier.
2. `DENSIFY_PARENT_SELECTION=evidence_local_max` in `configs/stage1.env` and
   the matching lifecycle code.  This is not R003.  It changes parent selection
   by using absolute-threshold graph-local maxima and removes the per-event
   budget.  It belongs to a later densification experiment.

R003 may audit and port the surface-interpolation implementation.  It must not
port `evidence_local_max`, pruning, or lifecycle timing in the same change.

## Implementation Targets

The first code audit targets are:

- `anigroom/flow/clean_flow.py`: keep the accepted V4 direction sampler as the
  reference behavior.
- `anigroom/roots/lifecycle.py`: replace Euclidean child-attribute inheritance
  with the unified surface-aware interpolation where children inherit groom
  attributes.
- `tools/train_white_tiger_stage1.py`: inspect guide/render interpolation call
  sites and route them through the same rule when they sample groom attributes.

## Implemented Contract

R003 keeps the current densification route and only changes how attributes are
sampled or inherited.

Implemented call sites:

- `guide -> render`: the previous Euclidean guide KNN cache is replaced by a
  cached surface support.  Support IDs are fixed until topology changes, while
  weights are recomputed from the current render-root surface position.
- `render -> guide startup initialization`: guide control initialization from
  dense render roots now uses topology-local face-ring support and transported
  3D directions instead of Euclidean render-root KNN.  This keeps startup guide
  length, lift, bend, width, and other controls consistent with the same
  surface rule.
- `guide-root newborn inheritance`: guide densification children inherit from
  topology-local guide support, not unrestricted Euclidean neighbors.
- `render-root newborn inheritance`: render densification children still inherit
  from old render roots, but the neighborhood is topology-local and typed.

Typed rules:

- 3D direction controls: `flow_xy` is first decoded with current
  flow-strength/lift into a 3D direction, parallel-transported to the query
  normal, averaged, then projected back to the query root's local control frame.
- Clean-flow direction targets: inherited as transported 3D directions.
- Physical scalar controls: length, root width, tip-width ratio, width taper,
  flow strength, lift, bend, sag, stiffness, curl radius/frequency, frizz,
  child radius, clump strength, opacity, and tip-opacity ratio are decoded to
  physical values, interpolated, then encoded back to raw parameters.
- Periodic controls: curl phase is interpolated on the unit circle with
  sin/cos, not by raw angle averaging.
- Colors: root and tip RGB are interpolated in decoded RGB space, then encoded
  back to raw sigmoid logits.
- Evidence buffers: observation confidence, anchor confidence, length target,
  and length confidence use the same surface-local scalar interpolation.

Preserved behavior:

- No root count, schedule, loss, renderer, residual, pruning, parent-retention,
  or overlong/screen-split policy changes were made.
- Existing overlong override hooks remain wired, but the accepted baseline keeps
  those routes disabled.

## Required Checks

Before training R003:

1. Source audit confirms the only algorithmic change is attribute
   interpolation/inheritance.
2. Unit or regression check confirms transported 3D direction interpolation is
   unchanged from accepted V4 for flow directions.
3. Scalar/color interpolation preserves shape, dtype, device, and old behavior
   on flat/same-normal neighborhoods.
4. Newborn-root attribute visualization is generated with the canonical
   visualization module.
5. R003 is compared to R002 with the same H100 command contract and canonical
   visualizations.

Current pre-training checks:

- `python -m py_compile anigroom\surface_interpolation.py
  anigroom\roots\lifecycle.py tools\train_white_tiger_stage1.py`: passed.
- Surface interpolation regression on a two-face mesh: topology-local support,
  normalized weights, finite physical interpolation, normalized 3D direction
  interpolation, and periodic phase interpolation all passed.
- `conda activate mygs` trainer import check: passed.
- Mini `WhiteTigerStage1Model` path check: guide-to-render interpolation,
  render-root insertion, guide-root insertion, and guide cache rebuild all
  passed with consistent tensor shapes.
- Pre-lock audit check: render-root structure updates cover all 20
  `GroomParameterField` parameters, including special 3D direction handling for
  `flow_xy` and periodic handling for `curl_phase`.
- Pre-lock audit check: guide-root structure updates cover guide physical
  controls, 3D clean-flow direction targets, length targets, confidence
  buffers, and region weights.
- Pre-lock audit fix: `initialize_guide_controls_from_roots()` was still using
  Euclidean KNN for startup guide control initialization.  It now uses local
  surface support plus transported 3D direction interpolation.  The dedicated
  mini-model path check passed.

## Training Result

The first R003 implementation was trained on the held H100 qlogin using the
same two-part accept-line contract:

- Phase A: `configs/v11_v4_parent_0_9k.env`
- Phase B: `configs/white_tiger_stage1_local_rgb_groom_v11_appearance_lenfree_from9k.env`
- Flow target:
  `/work/anigroom-accept-line/_downloads/tiger_hair_flow_36/shell_fused_smal_head500_body4000_candidate65536_headk24_bodyk12_v4_surface_direction/guide_flow3d_shell_targets_exclude_004_024_025.npz`
- Output:
  `/work/anigroom-accept-line/outputs/r003_surface_interp_20260720_182227`

Phase A completed from 0 to 9000.  Phase B first exposed one implementation
bug in the R003 interpolation helper: `interpolate_physical()` only handled
`[N]` and `[N,C]` tensors, but Phase B also interpolates high-rank child
attribute tensors such as `[N, child_count, 3]`.  The fix generalized weight
broadcasting across arbitrary trailing dimensions.  A targeted high-rank
regression passed locally and on H100, then Phase B was restarted from the
Phase A 9000 checkpoint.  No Phase A rerun was needed.

Composite PSNR:

| Iteration | Train | Test | Roots | Gaussians |
| ---: | ---: | ---: | ---: | ---: |
| 9000 | 24.1189 | 24.3421 | 184644 | 7393201 |
| 10000 | 29.3396 | 29.1907 | 194884 | 8323351 |
| 12000 | 31.2246 | 30.7253 | 195908 | 8564248 |
| 16000 | 32.5346 | 31.7983 | 195908 | 8678727 |
| 20000 | 33.2557 | 32.3962 | 195908 | 8678727 |
| 24000 | 33.4427 | 32.5815 | 195908 | 8678727 |
| 29000 | 33.5964 | 32.7515 | 195908 | 8678727 |
| 30000 | 33.6514 | 32.6048 | 195908 | 8678727 |

The best recorded test composite PSNR for that run was `32.7515` at iteration
29000.  The final 30000-step test composite PSNR was `32.6048`.  Peak
allocated CUDA memory was about `15.6 GB`; the run completed without OOM or
traceback after the high-rank interpolation fix.

Important lock-status note: this 30k metric was produced before the pre-lock
startup guide-initialization leak was fixed.  It remains useful evidence that
the typed interpolation path trains stably, but the final locked R003 baseline
must be refreshed from zero with the startup guide-init fix included before it
is tagged as the canonical baseline.

## Visual Audit

Canonical attribute visualizations were generated at 12000 and 30000:

- Local 12000 images:
  `D:\RTS\_tmp\r003_12k_attr_images`
- Local 30000 images:
  `D:\RTS\_tmp\r003_30k_attr_images`
- Local 30000 pure-fur Blender render:
  `D:\RTS\_tmp\r003_30k_strands\blender_side_y_100k_aligned.png`

Important visual findings:

- R003 does not create a global circular flow around the mesh.  The effective
  3D directions are mostly down/back and remain compatible with the V4 flow
  target.
- Length and normal/lift fields are more consistently inherited than the old
  Euclidean path, especially across newly inserted roots.
- Remaining defects are still visible near edges, belly, legs, tail, and some
  top/back regions: local direction outliers, drapey long fur, blocky length
  patches, and bend-field noise.
- Curl and frizz remain off in this route.  The remaining curled-looking
  artifacts are therefore not caused by explicit curl/frizz parameters; they
  are coming from length, bend, local direction inconsistency, or edge
  supervision.

## R003 Decision

R003 is accepted as the current attribute-interpolation direction.  It should
be kept because it makes guide/render/root inheritance follow the same surface
rule as the accepted V4 flow target, and the completed 30k run improved over
the exact R001/V11 reproduction.

R003 is not yet tagged as the final canonical baseline after the pre-lock
audit, because the startup guide-init leak changes the from-zero training
contract.  The next immediate action is to refresh the same two-phase R003 run
with the fixed code.  If the refreshed run stays stable and comparable, tag
that commit/output as the locked R003 baseline.

After that lock, the next experiments should not redo interpolation.  They
should target the remaining edge/length/bend behavior: edge confidence, edge
length initialization, length smoothing, bend smoothing, and appearance
residual handoff.

## Explicit Non-Scope

- No RGB-derived flow-loss changes.
- No new Gaussian-level appearance residual.
- No overlong split revival.
- No guide/render root count changes.
- No pruning schedule changes.
- No mesh/backing/compositing changes.
- No renderer changes.
