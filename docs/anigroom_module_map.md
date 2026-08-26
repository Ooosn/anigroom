# AniGroom Module Map

Current source of truth: `docs/current_route.md`. R043 is the active
structural/lifecycle result; R036 remains the frozen higher-PSNR metric
control. R062 is the current accepted advanced-geometry/appearance/validity
baseline, with R061 frozen as its direct no-collision control. R043 retains the
accepted flow, interpolation, hierarchy, one-turn
centerline, and finite 600-9000 render lifecycle while using 400k independent
render roots, exact accelerated graph/lifecycle selection, and density-matched
K32 render support. Guide support remains K8.

## Flow Initialization

- Code: `anigroom/flow`, `anigroom/projection`, and the flow-building tools.
- Accepted target: `v5_surface_direction` documented in
  `docs/modules/01_flow_initialization.md`.
- Optional artist-guided acquisition is isolated in
  `tools/groom_flow_annotator.py` and `anigroom/seed_flow_annotations.py`.
  It stores no-length directed 2D seeds, manual-anchor ownership, and cached
  local interpolation. `anigroom/flow_annotations.py` remains only as the
  validated legacy-arrow compatibility reader.
- Automatic and manual evidence are intended to meet at the same
  confidence-weighted 3D guide-anchor interface. The annotator itself does not
  change R068 training behavior; 2D-to-3D fusion remains a separately gated
  flow-module integration.

## Strand-To-Gaussian Representation

- Code: `anigroom/grooming/strand_gaussians.py`.
- The R058 geometry implementation directly replaces the retired curl/frizz
  formula and is isolated in
  `anigroom/grooming/strand_deformations.py` and documented in
  `docs/r058_advanced_groom_geometry.md`. R057 remains the last trained shape
  checkpoint, but there is no R057 geometry compatibility path in source.
- It owns explicit groom decoding, strand construction, adaptive segment
  counts, and strand-to-Gaussian conversion.
- R043 constructs one guide-owned quadratic normal-to-groom curve and allocates
  Gaussians from its final arc/turn complexity. Curve strength is multiplied by
  the continuous normal/direction difference. No second interior deformation
  is present in the executable schema.
- The active route uses `child_count=1`. Every render root owns one independent
  strand; deterministic tangent-offset child expansion is not active.
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
- Active R043 result: train/test composite PSNR `33.46581 / 32.51159`; best
  test composite PSNR `32.71421` at 29k.
- Accepted R062 advanced result: train/test composite PSNR
  `33.26321 / 32.19214`; best test composite `32.28517` at 29k. It adds only
  mesh no-penetration to R061 and preserves the complete R061 schedule,
  lifecycle, appearance decomposition, and renderer.
- Active behavior config and lock:
  `configs/r043_density_matched_render_support_0_30k.env` and
  `configs/r043_density_matched_render_support.lock.json`.
- R038/R039 configs remain historical evidence and are not launcher fallbacks.
- Frozen R036 metric-control lock: `configs/stage1_baseline.lock.json`.
- Evidence: `docs/r043_density_matched_render_support.md` and
  `docs/accept_line_recovery_ledger.md`; advanced validity evidence is in
  `docs/mesh_no_penetration.md`.

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
  both updates and lifecycle-only statistics. Exact surface graphs and fixed
  mesh adjacency are reused without changing K, selected roots, or ordering.
  Old target-directed placement branches are no longer present.
- Densification and pruning remain part of the multi-level training method,
  not standalone synthetic replacements.

## Mesh Collision Validity

- SDF construction: `anigroom/collision/mesh_sdf.py`.
- Differentiable query and normalized penetration loss:
  `anigroom/collision/sdf.py`.
- Formal checkpoint audit: `tools/diagnose_checkpoint_no_penetration.py`.
- Accepted behavior config: `configs/r062_mesh_no_penetration_0_30k.env`.
- R062 samples continuous non-root strand points in mesh-local coordinates,
  uses positive-outside trilinear SDF queries, and rotates over 16,384 roots
  per iteration. It contains no body-part mask, absolute penetration tolerance,
  strand-length condition, fallback query, or collision-driven global scale
  gradient.
- R062 reduces all-root penetrating point fraction by `82.43%` and maximum
  normalized depth by `51.16%` versus R061 while retaining matched RGB and
  strand structure.

## Visualization And Export

- `tools/render_white_tiger_stage1_checkpoint_views.py`
- `tools/export_white_tiger_checkpoint_strands.py`
- `tools/export_white_tiger_checkpoint_gaussians_ply.py`
- `tools/visualize_white_tiger_groom_attributes.py`
- `tools/blender_render_strand_npz.py`
- `tools/diagnose_strand_crossings.py`

Structural QA and asset rendering use different fixed protocols. Their exact
settings are recorded in `docs/current_route.md`; parent-only QA must not be
presented as the asset result.

## Change Rule

Training behavior changes only through a named, documented experiment. A new
result replaces the last measured reference only after metric and canonical
pure-fur acceptance.
