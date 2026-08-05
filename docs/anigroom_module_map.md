# AniGroom Module Map

Current source of truth: `docs/current_route.md`. R036 is the frozen measured
Stage 1 baseline. R034's absolute segment repair, R035's hierarchical width
profile, and R036's hierarchical child spread are retained; R034's direct
render-root width learning and the unrun R037 schedule proposal are rejected or
deferred.

## Flow Initialization

- Code: `anigroom/flow`, `anigroom/projection`, and the flow-building tools.
- Accepted target: `v4_surface_direction` documented in
  `docs/modules/01_flow_initialization.md`.

## Strand-To-Gaussian Representation

- Code: `anigroom/grooming/strand_gaussians.py`.
- It owns explicit groom decoding, strand construction, child expansion,
  adaptive segment counts, and strand-to-Gaussian conversion.
- Guide/render length, width, and child-spread controls use reference-relative,
  positive-unbounded coordinates; opacity and tip-width ratio use semantic
  unit intervals, while width taper is positive-unbounded. Segment allocation is linear in absolute physical
  length and strand complexity, with a representation floor but no upper cap
  or initialization-derived reference.

## Stage 1 Training Baseline

- Formal entry: `tools/train_white_tiger_stage1.py`.
- Generic runner: `scripts/server/run_white_tiger_stage1.sh`; `CONFIG_PATH` is
  mandatory and has no fallback.
- From-zero config: `configs/stage1_baseline.env`.
- Frozen result: train/test composite PSNR `33.42397 / 32.66322`; best test
  composite PSNR `32.83977` at 29k.
- Lock manifest: `configs/stage1_baseline.lock.json`.
- Evidence: `docs/stage1_baseline_r036.md`,
  `docs/r036_hierarchical_child_spread.md`, and
  `docs/accept_line_recovery_ledger.md`.

The active schema is strict and has no historical checkpoint migration.

## Surface Interpolation And Root Lifecycle

- Surface interpolation: `anigroom/surface_interpolation.py`.
- Root lifecycle: `anigroom/roots/lifecycle.py` and
  `anigroom/roots/statistics.py`.
- The recovery ledger records how these components reached the current
  contract.
- Guide-root lifecycle uses exact forward surface-support attribution and
  intrinsic local maxima. Guide-length smoothness is an area-integrated
  intrinsic log-gradient with fixed initial reference spacing, so
  densification does not weaken the physical regularizer.
- The formal render-root lifecycle has one path only: `pixel_to_root` evidence,
  intrinsic local maxima, and topology-local split/delete placement. Old
  target-directed placement branches are no longer present.
- Densification and pruning remain part of the multi-level training method,
  not standalone synthetic replacements.

## Visualization And Export

- `tools/render_white_tiger_stage1_checkpoint_views.py`
- `tools/export_white_tiger_checkpoint_strands.py`
- `tools/export_white_tiger_checkpoint_gaussians_ply.py`
- `tools/visualize_white_tiger_groom_attributes.py`
- `tools/blender_render_strand_npz.py`

Structural QA and asset rendering use different fixed protocols. Their exact
settings are recorded in `docs/current_route.md`; parent-only QA must not be
presented as the asset result.

## Change Rule

Training behavior changes only through a named, documented experiment. A new
result replaces the last measured reference only after metric and canonical
pure-fur acceptance.
