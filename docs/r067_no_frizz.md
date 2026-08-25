# R067 No Frizz

Status: accepted current single-sample method baseline, 2026-08-25. R066 is
the learned-turn parent and R065 is the exact crossing reference. This document
records the frozen implementation and formal evidence; it does not claim
multi-scene or multi-seed generalization.

## Identity And Result

- training commit: `8c010f09576f671df92ff40cdabff5886648c55e`
- postprocess commit: `18217be56dc468fdb8e1fffc9f0c9c39689ddce1`
- checkpoint SHA-256: `2433812f8ab784f9b04d94c88a782121fc3c11ea9522f1053b8e5f7e150b5729`
- checkpoint: schema 9, `stage1_full`, iteration `30000`
- strict tests: `204` pre-training and `210` postprocess
- formal training: uninterrupted from-zero 30k, exit status 0
- final train/test composite PSNR: `33.101788 / 32.069145`
- fixed eight-view mean composite PSNR: `33.077009`
- R066 fixed eight-view mean: `33.125197`; delta R067-R066: `-0.048189 dB`
- final render roots / training-metric Gaussians: `471673 / 5382959`
- peak allocated CUDA memory: `19825.54 MB`
- wall time: `15775.028 s`

## Count Semantics: Training Metric Versus Checkpoint State

The documented `5,382,959` is the final iteration's pre-step training metric.
At iteration `30000`, `render_parameters()` produced that count before the
backward pass and optimizer update; the metrics row retained that render
statistics payload. The checkpoint was saved after the optimizer step and
evaluation. Reconstructing the saved checkpoint with the formal exporter path
produces `5,382,896` Gaussians, so the labeled difference is:

`pre_step_training_metric_minus_checkpoint_state = 5,382,959 - 5,382,896 = 63`.

The checkpoint-state count is the authoritative export count. Segment budgets
must be derived directly from the returned `root_indices` and `segment_indices`
using the formal device-side allocator and exact per-root maxima. The audit
must repeat the reconstruction exactly and require identical per-root counts,
histograms, and order hashes. No padding, deletion, rounding adjustment, or
subsampling is permitted to force the checkpoint count to the earlier metric.

R067 learned curl/final cumulative turn P50/P95 is `2.07234/21.20436`
degrees, versus R066 final `15.29414/68.64623`. Final maximum local turn
P99/max is `2.32353/3.60025` degrees, versus R066
`18.97307/45.38681`. Final arc/chord P95/P99 is `1.00553/1.02621`, versus
R066 `1.01402/1.03095`. Both audits contain zero backward strands and zero
full foldback roots. The exact crossing audit reports `14872` unique
intersecting strand pairs, versus R066 `15418`; crossing remains secondary.

Full-root direct state decode gives root-opacity means
`0.9909846485` versus R066 `0.9909962107`, tip-opacity-ratio means
`0.9454077401` versus `0.9456069964`, and tip-opacity means `0.9412825014`
versus `0.9414174664`. Gaussian RGB residual absolute mean/RMS/saturation are
`0.0497083994/0.0790892018/0.0202916788` versus R066
`0.0502568274/0.0796196752/0.0207626478`. All are slightly lower in R067;
the accepted direct state decode therefore provides no evidence that opacity
or Gaussian RGB residuals compensated for frizz removal.
The exact per-view values, quantiles, formulas, deltas, source checksums, and
direct checkpoint decode are in:

`D:/RTS/_tmp/r067_acceptance_20260825/postprocess/r067_no_frizz/analysis/r067_vs_r066_metrics.json`

## Frozen Architecture

R067 is a clean reconstruction removal, not a scale-zero mode or migration.

- `DecodedGroom` and `GroomParameterField` contain no frizz field, parameter,
  or persistent seed buffer.
- `RenderGeometryResidualField`, primary-guide controls, secondary-guide
  interpolation, effective composition, `build_strands`, and `deform_backbone`
  contain no frizz value or argument.
- Adam state, lifecycle row migration, smoothness, prior, loss, finite-state,
  crossing ownership, checkpoint state, and optimizer names contain no frizz key.
- `frizz_backbone` remains only as a deterministic, differentiable, standalone
  procedural post-edit utility and is disconnected from reconstruction.
- Crossing owns only local direction and curl-radius residuals.
- Primary-guide turns are signed, direct, zero-initialized, and phase zero;
  secondary residuals own no turns field.
- The schema-9 loader rejects schema 8 before model/config/optimizer loading;
  there is no fallback, alias, resume, or old-config migration.

The core single-sample implementation is frozen at the identity above.

## Config And Formal Contract

The R067 config inherits R066 and changes no environment value except removing
these two effective keys:

```json
{
  "GUIDE_FRIZZ_RESIDUAL_SCALE": {"r066": "1.0", "r067": null},
  "SHAPE_FRIZZ_SCALE": {"r066": "1.0", "r067": null}
}
```

The strict launcher requires the exact clean commit, fresh runtime/output,
`mygs`, frozen data/mesh/SDF, schema 9, unlimited virtual memory, full pytest,
full 1920x1080 preflight, and uninterrupted from-zero 0-30k training. Resume,
fallback, reduced resolution, reduced preflight, and hidden frizz arguments
are forbidden.

## Accepted Evidence Paths

Remote training and strict validation:

- checkpoint: `/home/wangyy/anigroom-r067-no-frizz-runtime-20260825/outputs/r067_no_frizz_0_30k_h100_20260825/checkpoint_030000.pt`
- final log: `/home/wangyy/anigroom-r067-no-frizz-runtime-20260825/logs/r067_no_frizz_0_30k_h100_20260825.log`
- strict validation: `/home/wangyy/anigroom-r067-no-frizz-runtime-20260825/contracts/r067_postrun_strict_validation.json`
- postprocess manifest: `/home/wangyy/anigroom-r067-no-frizz-runtime-20260825/postprocess/r067_protocol_20260825/r067_postprocess_manifest.json`

Local acceptance root:

`D:/RTS/_tmp/r067_acceptance_20260825/postprocess/r067_no_frizz`

Reusable artifacts include:

- `rgb_views/render_report.json` and the eight fixed full-resolution RGB,
  target, difference, shape-ablation, and Gaussian-residual-ablation views;
- `components/curl_component_report.json` and backbone, learned curl-only,
  fixed-1.2, primary, and final strand NPZs;
- `foldback/foldback_component_report.json`,
  `crossing/strand_crossing_report.json`, and
  `structure/strand_structure_audit.json`;
- `attributes_view00`, `attributes_view09`, and `attributes_view32`, including
  signed primary-guide maps, curl/radius maps, root opacity, tip opacity, and
  tip-opacity-ratio maps;
- `assets_blender_protocol_20260825`, containing final full assets for
  `side_y`, `side_y_pos`, and `front_z` plus learned/fixed/final components;
- `assets_blender_rump_closeup_ortho082_20260825`, containing corrected
  backbone, curl-only, fixed-1.2, primary, and final closeups;
- `analysis/r067_vs_r066_metrics.json` and `.md`.

The corrected closeups use the validated `1600x1200`, `side_y_pos`, target
`0.5 0 0.21`, ortho-scale `0.82`, Cycles `96` protocol. The method control
figure is frozen and validated at:

`D:/RTS/_tmp/anigroom-r067-no-frizz/paper/method/fig_parametric_groom_controls.pdf`

R065 parent/crossing and no-penetration references remain at:

- `D:/RTS/_tmp/r065_acceptance_20260815/postprocess/r065_local_crossing_residual`
- `D:/RTS/_tmp/r065_curl_frizz_component_20260816`

## Boundary And Paper Readiness

R067 inherits the accepted R065 no-penetration contract and its training logs
retain the collision loss. The completed final-checkpoint R067 all-root
no-penetration audit is recorded at
`D:/RTS/_tmp/r067_acceptance_20260825/postprocess/r067_no_frizz/no_penetration_final/report.json`
with SHA256 `da1f104c9a4f796720e72beff269525cda960c984bc46a2d240b382e527a3083`.
Its matched R065 comparison is validity evidence, not an improvement claim.
The paper readiness audit is documented in
`docs/paper_readiness_20260825.md`.
