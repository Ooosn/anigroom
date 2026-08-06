# AniGroom Module Map

Current source of truth: `docs/current_route.md`. R038 remains the last accepted
structural/lifecycle result while R039 is the strict-schema representation
candidate. R036 remains the frozen higher-PSNR metric control. R039 retains the
accepted interpolation, hierarchy, and finite 600-9000 render lifecycle while
replacing the centerline construction only.

## Flow Initialization

- Code: `anigroom/flow`, `anigroom/projection`, and the flow-building tools.
- Accepted target: `v4_surface_direction` documented in
  `docs/modules/01_flow_initialization.md`.

## Strand-To-Gaussian Representation

- Code: `anigroom/grooming/strand_gaussians.py`.
- It owns explicit groom decoding, strand construction, child expansion,
  adaptive segment counts, and strand-to-Gaussian conversion.
- R039 constructs one guide-owned quadratic normal-to-groom curve and allocates
  Gaussians from its final arc/turn complexity. Curve strength is multiplied by
  the continuous normal/direction difference. No second interior deformation
  is present in the executable schema.
- Guide/render length, width, and child-spread controls use reference-relative,
  positive-unbounded coordinates; opacity and tip-width ratio use semantic
  unit intervals, while width taper is positive-unbounded. Segment allocation is linear in absolute physical
  length and strand complexity, with a representation floor but no upper cap
  or initialization-derived reference.

## Stage 1 Training Baseline

- Formal entry: `tools/train_white_tiger_stage1.py`.
- Generic runner: `scripts/server/run_white_tiger_stage1.sh`; `CONFIG_PATH` is
  mandatory and has no fallback.
- Frozen R036 metric-control config: `configs/stage1_baseline.env`.
- Active R038 result: train/test composite PSNR `33.03637 / 32.34588`; best
  test composite PSNR `32.51677` at 29k.
- Active config and lock: `configs/r038_brush_curve_0_30k.env` and
  `configs/r038_brush_curve.lock.json`.
- R039 candidate config: `configs/r039_brush_centerline_0_30k.env`. It cannot
  load an R038 checkpoint because the groom schema intentionally changed.
- Frozen R036 metric-control lock: `configs/stage1_baseline.lock.json`.
- Evidence: `docs/r038_brush_curve_and_9k_lifecycle.md` and
  `docs/accept_line_recovery_ledger.md`.

The active schema is strict and has no historical checkpoint migration.

## Surface Interpolation And Root Lifecycle

- Surface interpolation: `anigroom/surface_interpolation.py`.
- Root lifecycle: `anigroom/roots/lifecycle.py` and
  `anigroom/roots/statistics.py`.
- The recovery ledger records how these components reached the current
  contract.
- The implemented guide-root lifecycle path uses exact forward surface-support
  attribution and intrinsic local maxima. It is disabled in R038. When enabled
  by a later isolated candidate, guide-length smoothness remains an
  area-integrated intrinsic log-gradient with fixed initial reference spacing,
  so densification cannot weaken the physical regularizer.
- The formal render-root lifecycle has one path only: `pixel_to_root` evidence,
  intrinsic local maxima, and topology-local split/delete placement. It runs
  every 100 iterations from 600 through 9000 with no event budget, then stops
  both updates and lifecycle-only statistics. Old target-directed placement
  branches are no longer present.
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
