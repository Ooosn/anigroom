# R066 Learned Curl Turns

Status: contract only. No R066 training, metrics, visual result, or acceptance
claim is made by this document.

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

No R066 result is claimed until every required artifact and reject criterion has
been reviewed.
