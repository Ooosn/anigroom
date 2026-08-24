# R066 Learned Curl Turns

Status: accepted current learned-curl-turns baseline, 2026-08-25. R065 remains
the exact parent and crossing reference. This document records quantitative
acceptance evidence; it does not claim that frizz is solved.

## Objective

R065's curl/frizz attribution identifies a representation question: the
accepted geometry learns curl radius and frizz amplitude, but the render-root
curl-turn coordinate remains a fixed `1.20`-turn template. R066 asks one
code-only question:

> Can a signed, primary-guide curl-turn field learn a spatially coherent,
> zero-centered curl state without changing the R065 training configuration or
> allowing RGB detail to be mistaken for supported periodic geometry?

The experiment is a strict code comparison against R065. The executable R066
config sources `configs/r065_local_crossing_residual_0_30k.env` and defines no
environment-variable override. Any behavior difference must therefore come
from the reviewed code and its checkpoint schema, not a hidden config change.

## Signed Primary-Guide Turns Architecture

The reviewed implementation must satisfy this ownership contract:

1. Add one trainable scalar `guide_curl_turns_raw` per primary guide.
2. Keep it as a direct signed, zero-centered coordinate. Raw zero must decode
   exactly to `curl_turns=0`; positive and negative values encode opposite curl
   handedness. Do not reintroduce the positive-only bounded decoder or a
   hidden environment-controlled turn range.
3. Interpolate the decoded primary-guide turns through the existing
   topology-local guide-to-render-root path. Render roots must not retain a
   separately learned turns field.
4. Keep curl radius ratio non-negative and independent from turns. A zero turn
   is the neutral no-periodic-curl state even when the radius field is
   nonzero. Keep `curl_phase=0` and the persistent frizz seed fixed and
   non-trainable.
5. Transport the primary-guide turns field through guide lifecycle updates with
   the same row correspondence used by the other guide-owned controls. Newly
   inserted rows must have a documented neutral/parent initialization; they may
   not recover the retired fixed `1.20` render-root state.
6. Include primary-guide turns in the guide optimizer, guide interpolation, and
   guide smoothness audit. The crossing route remains local: crossing may update
   the existing secondary local direction/radius/frizz residuals, but not
   primary-guide turns.

The code must expose enough checkpoint and postprocess state to distinguish a
learned signed field from a constant, a collapsed zero field, or a saturated
decoder. No white-tiger-specific mask, region, absolute curl clamp, or crossing
threshold is allowed.

## Checkpoint Boundary

R066 is schema 8. The strict loader must require schema 8 and reject schema 7
with no alias, key conversion, compatibility migration, non-strict loading, or
resume path. The R066 launcher separately rejects `RESUME_CHECKPOINT`, checks
the reviewed trainer's schema declaration, and validates the final checkpoint
as `checkpoint_kind=stage1_full` at iteration `30000`.

The formal run is one uninterrupted from-zero `0-30k` execution. A v7
checkpoint is not an R066 input or fallback. A dirty checkout, stale runtime,
reduced-resolution path, batch-only preflight, or substitute renderer is a hard
failure.

## Unchanged Systems

The R065 contract remains unchanged for:

- frozen images, camera calibration, train/test split, mesh, and mesh SDF;
- full `1920x1080` resolution, sample count, root/guide counts, and lifecycle;
- clean-flow initialization, surface interpolation, smoothing graphs, density,
  pruning, guide updates, and optimizer-state transport;
- RGB, flow, mask, appearance-residual, guide, smoothness, and mesh
  no-penetration losses, weights, unlock schedules, and learning rates;
- primary/secondary ownership other than the new primary-guide turns field;
- strand-crossing active-set discovery, weight, batches, refresh schedule, and
  secondary local residual ownership;
- renderer, deterministic seeds, fixed eight-view protocol, and 100k export
  protocol.

The R066 environment snapshot must be exactly equal to the R065 snapshot. The
launcher writes both snapshots and fails unless the serialized config delta is
exactly `{}`.

## Formal Execution Contract

Before training, the launcher must prove:

1. `EXPECTED_COMMIT` is a full SHA matching `HEAD` in a clean checkout;
2. `RUNTIME_ROOT` and its output are fresh and non-existent;
3. `PYTHON` is an executable from the `mygs` environment, and the data root,
   aligned mesh, and reviewed mesh SDF exist;
4. the trainer declares checkpoint schema 8 and no schema-7 resume path is
   available;
5. R065/R066 snapshots differ by exactly zero variables;
6. `EXPECTED_WIDTH=1920`, `EXPECTED_HEIGHT=1080`, and `ITERATIONS=30000`;
7. `ulimit -v` is `unlimited`;
8. the full pytest suite passes.

The formal runner then invokes the existing stage-1 launcher with the R066
config, `RUN_PREFLIGHT=1`, `RUN_BATCH_PREFLIGHT=0`, no resume checkpoint, and
no view or preflight-only overrides. The runner must finish at the 30k
checkpoint before postprocess acceptance begins.

## Accepted Result

R066 is accepted as the current learned-curl-turns baseline. It preserves the
R065 parent for reconstruction, validity, lifecycle, and exact crossing
comparison; R065 is not replaced as the crossing reference.

### Formal Execution Evidence

- training commit: `46672fab4b1d6317fcdc041af067a955cb99f12b`;
- postprocess commit: `d912ef2fdedbcd47ccdafacb1562fbec1d2e2d53`;
- schema-8 `stage1_full` checkpoint at iteration `30000`;
- checkpoint SHA-256:
  `21e0e3a66907067215ceb3f0232432c4cd9d0ef4bea0826a62ebd6e2d1410f06`;
- `184` pre-training tests and `186` postprocess tests;
- uninterrupted full-resolution from-zero `0-30k` execution, exit code `0`;
- final train/test composite PSNR: `33.149204 / 32.107651`;
- fixed eight-view mean composite PSNR: `33.125197`, versus R065 `33.222302`,
  delta `-0.097105` dB;
- final render roots/Gaussians: `471605 / 5391612`;
- peak allocated CUDA memory: `20392.39 MB`; wall time: `15267.73 s`.

The primary turns field is a direct signed, zero-initialized per-guide
coordinate with explicit `curl_phase=0`. It is owned by primary guides,
interpolated to render roots, optimized and lifecycle-transported with guide
state, and absent from secondary residual ownership. The strict loader accepts
schema 8 only; there is no schema-7 migration, alias, or resume path.

### Quantitative Shape Evidence

- learned curl-only cumulative turn P50/P95: `2.10359 / 20.48548` degrees;
- matched fixed-`1.2` cumulative turn P50/P95:
  `24.60057 / 129.16371` degrees;
- final curl+frizz cumulative turn P50/P95:
  `15.294 / 68.646` degrees, versus R065 final `34.179 / 119.282`;
- final arc/chord P95/P99: `1.01402 / 1.03095`, versus R065
  `1.03666 / 1.08083`;
- zero backward strands and zero full foldbacks;
- crossing contacts at least 45 degrees: `217` versus R065 `198`. This small
  regression is secondary evidence and is not treated as solved by R066.

The final local-turn P99/max remains `18.973 / 45.387` degrees. The top
extremes are frizz-dominated, so the next target is frizz rather than another
crossing or turns change. R066 does not claim frizz is solved.

### Artifact Record

Remote training output and log:

- `/home/wangyy/anigroom-r066-learned-curl-turns-runtime-20260824/outputs/r066_learned_curl_turns_0_30k_h100_20260824`
- `/home/wangyy/anigroom-r066-learned-curl-turns-runtime-20260824/logs/r066_learned_curl_turns_0_30k_h100_20260824.log`
- checkpoint:
  `/home/wangyy/anigroom-r066-learned-curl-turns-runtime-20260824/outputs/r066_learned_curl_turns_0_30k_h100_20260824/checkpoint_030000.pt`

Remote postprocess and manifest:

- `/home/wangyy/anigroom-r066-learned-curl-turns-runtime-20260824/postprocess/r066_protocol_20260825`
- `/home/wangyy/anigroom-r066-learned-curl-turns-runtime-20260824/postprocess/r066_protocol_20260825/r066_postprocess_manifest.json`
- `/home/wangyy/anigroom-r066-learned-curl-turns-runtime-20260824/postprocess/r066_protocol_20260825/validation/postprocess_validation.json`

Local acceptance and quantitative report:

- `D:/RTS/_tmp/r066_acceptance_20260825/postprocess/r066_learned_curl_turns`
- `D:/RTS/_tmp/r066_acceptance_20260825/postprocess/r066_learned_curl_turns/r066_postprocess_manifest.json`
- `D:/RTS/_tmp/r066_acceptance_20260825/postprocess/r066_learned_curl_turns/analysis/r066_vs_r065_metrics.json`
- `D:/RTS/_tmp/r066_acceptance_20260825/postprocess/r066_learned_curl_turns/analysis/r066_vs_r065_metrics.md`

Signed learned-guide maps are the matching prediction overlays:

- `D:/RTS/_tmp/r066_acceptance_20260825/postprocess/r066_learned_curl_turns/attributes_view00/view00_primary_guide_curl_turns.png`
- `D:/RTS/_tmp/r066_acceptance_20260825/postprocess/r066_learned_curl_turns/attributes_view09/view09_primary_guide_curl_turns.png`
- `D:/RTS/_tmp/r066_acceptance_20260825/postprocess/r066_learned_curl_turns/attributes_view32/view32_primary_guide_curl_turns.png`

Accepted corrected Blender visual evidence is now present. Full assets are under
`D:/RTS/_tmp/r066_acceptance_20260825/postprocess/r066_learned_curl_turns/assets_blender_protocol_20260825`:

- `final_100k_side_y.png`
- `final_100k_side_y_pos.png`
- `final_100k_front_z.png`
- `curl_only_side_y_pos.png`
- `curl_fixed_1p2_turns_side_y_pos.png`
- `final_curl_frizz_side_y_pos.png`
- validated manifests: `asset_manifest.json` and `png_validation.json`.

Corrected true rump closeups are under
`D:/RTS/_tmp/r066_acceptance_20260825/postprocess/r066_learned_curl_turns/assets_blender_rump_closeup_ortho082_20260825`:

- `backbone_side_y_pos_closeup.png`
- `curl_only_side_y_pos_closeup.png`
- `curl_fixed_1p2_turns_side_y_pos_closeup.png`
- `frizz_only_side_y_pos_closeup.png`
- `primary_curl_frizz_side_y_pos_closeup.png`
- `final_curl_frizz_side_y_pos_closeup.png`
- validated manifests: `asset_manifest.json` and `png_validation.json`.

The corrected closeup protocol is `1600x1200`, camera `side_y_pos`, target
`0.5 0 0.21`, orthographic scale `0.82`, and Cycles `96` samples. Both asset
manifests and PNG validation manifests are validated.

The fixed control remains quantitative as well:

- R066 fixed control NPZ:
  `D:/RTS/_tmp/r066_acceptance_20260825/postprocess/r066_learned_curl_turns/components/curl_fixed_1p2_turns_100000_s32.npz`
- R066 component report:
  `D:/RTS/_tmp/r066_acceptance_20260825/postprocess/r066_learned_curl_turns/components/curl_frizz_component_report.json`
- exact R065 parent/crossing acceptance:
  `D:/RTS/_tmp/r065_acceptance_20260815/postprocess/r065_local_crossing_residual`
- exact remote R065 parent/crossing postprocess:
  `/home/wangyy/anigroom-r065-local-crossing-residual-runtime-20260815/postprocess/r065_local_crossing_residual`
- exact R065 curl/frizz component report:
  `D:/RTS/_tmp/r065_curl_frizz_component_20260816/curl_frizz_component_report.json`
- R065 learned/final component closeups retained as the parent reference:
  `D:/RTS/_tmp/r065_curl_frizz_component_20260816/closeups/r065_curl_only_rump_closeup.png`
  and
  `D:/RTS/_tmp/r065_curl_frizz_component_20260816/closeups/r065_final_curl_frizz_rump_closeup.png`.

The quantitative report, signed maps, full Blender assets, and corrected rump
closeups above are the acceptance evidence for this baseline. They do not
claim that frizz is solved.

## Acceptance Evidence

The result is accepted only with all of the following evidence, stored under an
isolated runtime and tied to the reviewed commit:

1. **Training contract:** full pytest, full-resolution active-path preflight,
   uninterrupted from-zero 0-30k log, no fallback/reduced resolution/OOM, and
   strict schema-8 checkpoint reload with optimizer-state integrity.
2. **Fixed RGB:** the same eight full-resolution views (`0 5 9 14 18 21 27 32`),
   their RGB composites, train/test composite metrics, and the fixed-view mean.
3. **Canonical asset:** one deterministic 100,000-strand child-1 export with
   the R065 sample/seed protocol, plus checksums and the same mesh/camera
   metadata.
4. **Component decomposition:** matched backbone-only, curl-only, frizz-only,
   primary curl/frizz, and final curl/frizz variants with identical roots,
   widths, colors, opacities, samples, cameras, mesh, and renderer. The learned
   turns contribution must be separately identifiable.
5. **Turn and shape metrics:** cumulative turn P50/P95, local-turn P50/P95 and
   maximum, backward/foldback counts, arc/chord P50/P95/P99/max, relative
   length continuity, and finite/non-collapsed signed-turn statistics.
6. **Learned-turn visualization:** signed-turn distribution, near-zero/
   positive/negative proportions, saturation checks, and a spatial guide/root
   visualization over the mesh and canonical views. The visualization must show
   where learned turns occur, not only an aggregate scalar.
7. **Resource evidence:** wallclock, iteration timestamps, peak allocated CUDA
   memory, GPU model, and final root/Gaussian population.
8. **Crossing context:** exact crossing and no-penetration metrics may be
   reported as secondary validity evidence. They cannot substitute for curl
   quality, turn-field evidence, or the component decomposition.

## Reject Criteria

Reject R066 for any contract or quality failure, including:

- non-empty R065/R066 config delta, dirty checkout, stale output, wrong commit,
  missing mygs/data/mesh/SDF, failed pytest, schema 7, resume/migration,
  fallback, reduced resolution, or incomplete 0-30k execution;
- missing, non-finite, unoptimized, spatially collapsed, signless, or saturated
  primary-guide turns; turns that remain the fixed `1.20` template; or turns
  that are owned by render roots or the crossing residual instead of primary
  guides;
- a new coherent periodic/stripe-correlated curl artifact, foldback,
  backward-strand population, or material regression in local turns,
  arc/chord, length continuity, or canonical fixed-view inspection;
- RGB/composite improvement without supported curl geometry or without the
  required turn distribution, spatial map, and component evidence;
- using lower crossing counts to rescue a failed curl-quality result.

The reject criteria remain the guard for any future R066 rerun; the accepted
record above is tied to the reviewed commits, checkpoint, and artifacts listed
in this document.
