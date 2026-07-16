# AniGroom Module Map

## Flow Initialization

- Code: `anigroom/flow`, `anigroom/projection`, and the flow-building tools.
- Accepted target: `v4_surface_direction` documented in
  `docs/modules/01_flow_initialization.md`.

## Strand-To-Gaussian Representation

- Code: `anigroom/grooming/strand_gaussians.py`.
- The checked-in implementation is the exact v11 version.
- Details: `docs/modules/02_strand_gaussian_representation.md`.

## Stage 1 Training Baseline

- Formal entry: `tools/train_white_tiger_stage1.py`.
- Exact from-zero runner: `scripts/server/reproduce_v11_from_zero.sh`.
- Parent config: `configs/reproduce_v11_parent_0_9k.env`.
- v11 config:
  `configs/white_tiger_stage1_local_rgb_groom_v11_appearance_lenfree_from9k.env`.
- Accepted result: train/test composite PSNR `33.1472 / 32.1938`.
- Evidence: `docs/v11_exact_reproduction_audit.md`.

The formal codebase exposes only this accepted v11 training route.

## Visualization And Export

- `tools/render_white_tiger_stage1_checkpoint_views.py`
- `tools/export_white_tiger_checkpoint_strands.py`
- `tools/export_white_tiger_checkpoint_gaussians_ply.py`
- `tools/visualize_white_tiger_groom_attributes.py`
- `tools/blender_render_strand_npz.py`

These tools are restored to the versions used with the exact v11 baseline.

## Change Rule

v11 is the current baseline. Training behavior changes only when explicitly
requested, and no experiment replaces v11 without both metric and pure-fur
acceptance.
