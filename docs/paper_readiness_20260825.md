# Paper Readiness Audit: R067

Status date: 2026-08-25. This is a read-only readiness assessment for the
accepted R067 single-sample method baseline. It records what is reusable and
what remains before a publication claim. It is not a visual acceptance claim.

## Verdict

It is false that only multi-sample tests remain. R067 has a complete
single-sample formal result and substantial qualitative/diagnostic evidence,
but submission still requires matched external baselines, generalization or
repeat evidence with uncertainty, actual checkpoint editing demonstrations,
publication comparison tables/figures, and comparable runtime/memory
measurements. Training speed is a blocker before
scaling those runs.

## Frozen R067 Identity

- training commit: `8c010f09576f671df92ff40cdabff5886648c55e`
- postprocess commit: `18217be56dc468fdb8e1fffc9f0c9c39689ddce1`
- checkpoint SHA-256: `2433812f8ab784f9b04d94c88a782121fc3c11ea9522f1053b8e5f7e150b5729`
- schema 9 strict no-frizz state
- `204` pre-training tests and `210` postprocess tests
- uninterrupted formal 30k, exit status 0
- final train/test composite: `33.101788/32.069145`
- fixed eight-view mean: `33.077009` versus R066 `33.125197`, delta `-0.048189 dB`
- roots/Gaussians: `471673/5382959`
- peak allocated memory: `19825.54 MB`
- wall time: `15775.028 s` (`4.38 h`)
- root-opacity/tip-ratio/tip-opacity means: `0.9909846485/0.9454077401/0.9412825014`
  versus R066 `0.9909962107/0.9456069964/0.9414174664`
- Gaussian RGB residual absolute mean/RMS/saturation:
  `0.0497083994/0.0790892018/0.0202916788` versus R066
  `0.0502568274/0.0796196752/0.0207626478`; no compensation

The core single-sample implementation is frozen. R066 remains the learned-turn
parent and R065 remains the exact crossing reference.

## Evidence Already Reusable

| Evidence | Status and exact source |
| --- | --- |
| White-tiger final quantitative | Complete fixed eight-view RGB report, train/test metrics, roots, Gaussians, memory, wall time, and ablations under `D:/RTS/_tmp/r067_acceptance_20260825/postprocess/r067_no_frizz`; aggregate at `analysis/r067_vs_r066_metrics.json`. |
| White-tiger qualitative | Full-resolution RGB/GT/difference assets, 100k child-one strand export, Blender full assets under `assets_blender_protocol_20260825`, and corrected rump closeups under `assets_blender_rump_closeup_ortho082_20260825`. These are reusable source material, not yet a paper figure selection. |
| Fixed 1.2-turn control | `components/curl_component_report.json`, fixed-control NPZ, matching full renders, and corrected closeup. The metadata proves non-turn controls and radius/length are matched. |
| No-frizz state | Remote strict validation `/home/wangyy/anigroom-r067-no-frizz-runtime-20260825/contracts/r067_postrun_strict_validation.json`, postprocess validation log, exact two-key config delta, schema-9 checkpoint, and recursive no-frizz metadata checks. |
| Gaussian RGB residual | Per-view residual-on/off images and report statistics in `rgb_views`; direct checkpoint state decoding and exact deltas are recorded in `analysis/r067_vs_r066_metrics.json`. R050 remains the historical accepted appearance experiment. |
| Primary/secondary ownership | Signed guide maps in `attributes_view00/09/32`, `components/curl_component_report.json`, `foldback/foldback_component_report.json`, and strict secondary curl-only metadata. No secondary turns field is claimed. |
| Multilevel roots | Guide count `4500`, formal render population `471673`, canonical `100000` strands at 32 samples, and lifecycle evidence through 85 events. The route and historical results are documented in `docs/current_route.md` and `docs/r_series_evolution.md`. |
| Clean-flow | Frozen V4 target `baseline_inputs/v4_surface_direction/guide_flow3d_shell_targets_exclude_004_024_025.npz` and its README/route documentation. This is method provenance, not a new R067 training result. |
| Lifecycle/densification | R067 strict validation and formal logs preserve the lifecycle contract; R043/R042 provide the accepted lifecycle and speed-history evidence. A compact publication plot/table is still needed. |
| No-penetration/crossing | R065 local acceptance has the all-root no-penetration, length-ownership, crossing, and highlight assets at `D:/RTS/_tmp/r065_acceptance_20260815/postprocess/r065_local_crossing_residual`. R067 now has the completed matched final audit at `D:/RTS/_tmp/r067_acceptance_20260825/postprocess/r067_no_frizz/no_penetration_final/report.json`; it is validity evidence, not an improvement claim. |
| Method figure/edit controls | Frozen method figure PDF/SVG/PNG at `D:/RTS/_tmp/anigroom-r067-no-frizz/paper/method/fig_parametric_groom_controls.pdf` and validated source assets under `D:/RTS/_tmp/paper_parametric_groom_controls_r067_full_v3`. This demonstrates synthetic control semantics, not editing of a trained white-tiger checkpoint. |

Remote training and postprocess manifests are:

- `/home/wangyy/anigroom-r067-no-frizz-runtime-20260825/logs/r067_no_frizz_0_30k_h100_20260825.log`
- `/home/wangyy/anigroom-r067-no-frizz-runtime-20260825/postprocess/r067_protocol_20260825/r067_postprocess_manifest.json`
- `/home/wangyy/anigroom-r067-no-frizz-runtime-20260825/outputs/r067_no_frizz_0_30k_h100_20260825/checkpoint_030000.pt`

## Must-Have Before Submission

1. **Matched external baselines.** The local tree contains NeuralFur and
   GaussianHaircut source under `D:/RTS/_tmp/NeuralFur_official` and a plain
   3DGS baseline renderer in `anigroom/baselines`, but no aligned accepted
   white-tiger result package was found. Run the relevant methods with the same
   data, cameras, resolution, split, evaluation, asset protocol, and cost
   accounting.
2. **Generalization and uncertainty.** One white-tiger scene and one formal
   deterministic seed are insufficient for a general method claim. Add multiple
   scenes and/or independent seeds and report mean, spread, and uncertainty.
   Eight camera views and 32 samples per strand are not statistical repeats.
3. **Actual checkpoint edit demonstrations.** Use the frozen R067 checkpoint
   to show controlled direction, curl radius/turns, brush stiffness, width, and
   opacity edits on the white tiger, with before/after renders and preserved
   root IDs/geometry provenance. The synthetic method figure alone is not enough.
4. **Publication result package.** Assemble final tables and figures covering
   baselines, R065/R066/R067 ablations, RGB residual effects, geometry tails,
   crossing as secondary context, and the no-frizz state. The repository paper
   directory currently contains the method section/figure, not a full results
   manuscript.
5. **Comparable runtime and memory.** R067 has exact wall/memory numbers, but
   baseline methods do not yet have matched measurements. Report throughput,
   peak allocated/reserved memory, root/Gaussian population, and postprocess
   cost under one contract.

## Speed Blocker

R050/R055 formal runs were approximately `2.6 h`; R067 is `15775.028 s`,
approximately `4.38 h`. Scaling to methods, scenes, and repeats before profiling
would make the comparison unnecessarily expensive.

| Phase or cost source | Current evidence | Readiness implication |
| --- | --- | --- |
| R050/R055 formal training | Approximately `2.6 h` per run | Historical quality-equivalent speed reference. |
| R067 formal training | `15775.028 s` / `4.38 h` | Current single-sample cost is materially higher. |
| Curl path | Late-stage work is not isolated in the formal log | Profile strand/curl construction and repeated effective-geometry evaluation first. |
| Crossing refresh | Approximately `810 s` accumulated | Amortize or batch exact active-set refresh while preserving gradients and ownership. |
| Lifecycle | Approximately `489 s` accumulated | Profile graph rebuild, selection, migration, and synchronization; preserve all 85 events and Adam rows. |
| Smoothness/SDF | Not separately logged | Add phase timers before optimizing; avoid claiming a cost reduction without attribution. |

Quality-equivalent optimization priorities are: instrument every loss and
refresh phase; remove duplicate curl/strand graph construction; reuse immutable
surface support and graph data where semantics permit; amortize exact crossing
refresh; accelerate lifecycle graph/row migration; batch SDF queries; and verify
same-checkpoint one-step equivalence, fixed-view metrics, population, and
ownership after each change. No resolution, schedule, loss-weight, or quality
gate may be reduced to obtain speed.

## Optional Follow-Ups

- sampling-density sensitivity at 16/32/64 samples per strand;
- clump-strength and child-spread editing panels;
- standalone procedural `frizz_backbone` post-edit demonstration;
- mesh-texture/material variation and additional Blender component galleries;
- more historical R-series ablations or a user-facing edit study after the
  must-have evidence is complete.

## Decision

R067 is ready to serve as the frozen single-sample method baseline and source
for a paper package. It is not yet submission-ready as a general method claim:
the external comparison, generalization/repeats, trained-asset edit demos,
publication tables/figures, and normalized cost comparison remain open.
