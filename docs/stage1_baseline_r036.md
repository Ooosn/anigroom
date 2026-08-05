# Frozen Stage 1 Baseline: R036

Status date: 2026-08-05.

R036 is the only frozen, measured, executable Stage 1 baseline. This document
answers three questions:

1. Which exact source and input define the baseline?
2. Which accepted R-series findings are present in the executable path?
3. Which investigated ideas are intentionally absent?

The machine-readable authority is
`configs/stage1_baseline.lock.json`. Run
`python tools/verify_stage1_baseline.py` before training or changing the
baseline.

## Identity And Formal Evidence

- baseline id: `stage1-r036`
- config: `configs/stage1_baseline.env`
- launcher: `scripts/server/run_white_tiger_stage1.sh`
- formal output:
  `/home/wangyy/anigroom-r036-hierarchical-child-spread-20260805/outputs/r036_from_zero_h100_20260805`
- final checkpoint SHA-256:
  `b4bedaa3d111c529e38d43ed7ffd922391292a5977934d10c4f10b7aeaf85f6b`
- final train/test composite PSNR: `33.42397308 / 32.66321945`
- best test composite PSNR: `32.83977127` at 29k
- final render roots/Gaussians: `317245 / 14215421`
- peak live PyTorch allocation: `24096.82 MiB`

The five formal runtime files recovered from HGC are byte-identical to the
current local files. The only path normalization is the clean-flow input: its
exact bytes are now tracked inside the repository at
`baseline_inputs/v4_surface_direction/`. Its SHA-256 is
`60a33b360bb415cb47cd38173d6e0cf4504448203ef277a5861641b40fdb3141`.

## Executable Contract

- full resolution: `1920x1080`
- from-zero training: required
- initial guide/render roots: `4500 / 100000`
- child strands per render root: `4`
- render lifecycle: every 100 iterations, 600 through 20k
- guide lifecycle: every 200 iterations, 11k through 16k
- guide attributes frozen through 9k
- render child-spread residual: 1k-to-7k coverage ramp
- other render geometry residuals: 10k-to-20k ramp
- shape detail frozen through 14k
- pruning: disabled
- curl/frizz effective scales: zero
- explicit config path: mandatory; no fallback configuration

## Accepted Mechanism Audit

Every retained item below is present in the current runtime, not merely in an
experiment document.

| Origin | Retained mechanism | Current executable evidence |
| --- | --- | --- |
| V4/R002 | Clean 3D surface-flow target | tracked V4 NPZ; `anigroom/flow`; `CLEAN_FLOW_TARGET` |
| R003 | One typed, surface-aware interpolation contract | `anigroom/surface_interpolation.py`; used by initialization and both lifecycles |
| R004/R007 | Thresholded, topology-local render split/delete with no event budget | `surface_attribution_local_max`; `MAX_SPLITS_PER_EVENT=0`; parent replacement remains active |
| R006/R016 | Confidence-aware guide length initialization and surface reconstruction of unsupported values | `initialize_groom_from_clean_flow`; `interpolate_unobserved_root_values` |
| R008 | Narrow two-pixel loss-edge confidence and mild effective-field smoothness | `LOSS_MASK_EDGE_KERNEL=5`; `EFFECTIVE_SMOOTH_WEIGHT=0.006` |
| R009 | Mesh-surface smoothing graph | `SMOOTH_GRAPH_MODE=surface_hierarchical` |
| R010/R017 | Parallel-transported 3D direction interpolation and reconstruction | `parallel_transport_vectors`; `reconstruct_clean_flow_directions` |
| R011 | Exact normalized local 3D direction representation | one normalized local vector; no second flow/lift representation |
| R013 | Effective-versus-guide relative geometry continuity | effective and residual graph smoothness in the active loss |
| R018 | Removal of the aggregate absolute physical-value shape prior | retired aggregate prior is absent from config and runtime |
| R023 | Positive, guide-relative, unbounded asinh log-length residual | `zero_centered_asinh_log_length_residual` |
| R027 | Adam row-state migration across render and guide lifecycle changes | `rebuild_stage1_optimizer_with_state` |
| R030 | Sparse length-tail concentration without an absolute cap | `tail_concentration_handoff`, mean L1 plus unlock-scaled `L4-L2` |
| R031 | Guide evidence attributed through exact forward interpolation support | `surface_attribution_local_max`; forward-support tests |
| R032 | Density-invariant intrinsic guide-length smoothness | `GUIDE_LENGTH_SMOOTH_MODE=intrinsic_density_invariant` |
| R033 | Positive-unbounded guide length and semantic opacity | local length references; opacity only uses its semantic unit interval |
| R034 | Absolute physical-length/complexity segment allocation with no upper cap | linear allocator; minimum representation count only |
| R035 | Guide-owned width profile with zero-centered render residuals | positive root width/taper, semantic tip ratio, lifecycle re-encoding |
| R036 | Guide-owned positive child spread with zero-centered render residual | child reference/raw fields; 1k-to-7k measured coverage ramp |
| execution repair | Activation recomputation, expandable allocator segments, and a memory guard | active without reducing resolution, roots, strands, or Gaussians |

The current render lifecycle score is the accepted raw R004/R007 score:

`need = gaussian_gradient / opacity_weighted_visible_contribution`
`     + abs(root_gradient) / visible_gaussian_count`
`     + projected_residual / visible_gaussian_count`

Statistics accumulate over the lifecycle window and are reset after each
structure event. Gaussian visibility comes from gsplat `radii > 0`. The
separate R005 `mean_visible` score remains in diagnostic code but is not
selected by the frozen configuration.

## Intentionally Absent

- R005 mean-visible lifecycle scoring: rejected because it selected a broad,
  nearly whole-body population at the tested threshold.
- R012 uncertainty-specific extra direction smoothing: rejected.
- R014 stronger relative-length weight: insufficient and rejected.
- R019 additive physical length residual: rejected; only its direct 3D
  direction finding survived.
- R020-R022 bounded/raw exponential length variants: superseded by R023.
- R024/R025 standalone length priors: rejected as tail controls.
- R026 optimizer-reset behavior: fixed by R027, not retained.
- R028 exact early-guide branch: useful evidence but not promoted because it
  changed width, opacity, and long-hair structure together.
- R029 full fourth-moment replacement: rejected because it suppressed the
  ordinary residual field.
- R034 direct per-render-root width ownership: rejected; R035 hierarchy
  replaces it.
- R037 all-coverage-field early schedule: deferred and never formally run.
- Gaussian-level RGB appearance residual: not part of R036.
- Brush stiffness/base-curve representation: documented future work, not part
  of R036.
- Pruning: disabled.

## Freeze Procedure

1. Run `python tools/verify_stage1_baseline.py`.
2. Run the complete test suite in `mygs`.
3. Syntax-check `scripts/server/run_white_tiger_stage1.sh`.
4. Launch only with an explicit `CONFIG_PATH` pointing at
   `configs/stage1_baseline.env`.
5. Save config, optimizer, RNG, lifecycle history, and checkpoints through the
   formal trainer.
6. Compare full-resolution composite metrics and fixed single-image pure-fur
   renders; neither metric-only nor structure-only acceptance is sufficient.

Any change to a file listed in `configs/stage1_baseline.lock.json` invalidates
the lock until it becomes a separately named candidate with new evidence.
