# Accept-Line Recovery Ledger

## Purpose

This is the single source of truth for the recovery of the accepted white-tiger
Stage 1 line. The goal is to recover a reproducible baseline first, then make
one validated change at a time. This document must be updated before any code
or configuration change is made in this worktree.

## Non-Negotiable Rules

1. `D:/petsgaussianhair` is a dirty historical/reconstruction worktree and is
   not an experiment source.
2. This worktree is isolated at a fixed Git commit. No code is copied from the
   dirty worktree without an evidence-backed audit entry below.
3. The accept line is evidence, not proof that every implementation detail is
   correct. Later fixes must be located, classified, and independently tested.
4. Every experiment changes one algorithmic variable only. Its config diff,
   hypothesis, measurements, and canonical visualizations are recorded here.
5. A run is only comparable when source revision, input contract, resolution,
   train/test split, evaluation metric, and visualization pipeline match.

## Frozen Sources

| Item | Value | Role |
| --- | --- | --- |
| Recovery worktree | `D:/petsgaussianhair-accept-line` | Only local source edited for this recovery |
| Git revision | `09de8c4911d290e64d9d3fc2c3681016d72aeca4` | Last V11/flow lock commit before the accepted July 17 run |
| Historical source archive | `D:/RTS/_tmp_upload/petsgaussianhair_sync_20260707_020044/petsgaussianhair_code_sync.tar.gz` | Evidence only; never blindly overlaid |
| Archive SHA-256 | `d495a7db6852b4167a66a899b6075c07fd55c5254a811cb28eeb53de703aa554` | Archive identity |
| Accepted output | `D:/petsgaussianhair/outputs/20260717022328` | Metric/checkpoint/config reference |
| Parent checkpoint | `D:/petsgaussianhair/outputs/20260717002440/checkpoint_009000.pt` | Required V11 handoff state |

## Accepted Line Contract

The accepted V11 result is a two-part route, not a single from-zero V11 run:

1. Parent route: iteration 0 through 9000, producing the parent checkpoint.
2. V11 continuation: construct the V11 model, load model and RNG state from
   the parent checkpoint, reset optimizer state, and continue to iteration
   30000.

Historical acceptance targets from the saved run/audit:

| Iteration | Test composite PSNR |
| ---: | ---: |
| 1000 | 21.1935 |
| 5000 | 23.2514 |
| 9000 | 24.0695 |
| 10000 | 28.4462 |
| 12000 | 30.0485 |
| 16000 | 31.2587 |
| 20000 | 31.9110 |
| 30000 | 32.1814 |

The final reference has approximately 187,993 render roots and 8.50M generated
Gaussians. Metrics alone are insufficient: full-resolution RGB, pure-fur asset,
effective direction, and effective length must all be compared with the
canonical visualization module.

## Audit Register

| ID | Status | Finding | Evidence | Decision |
| --- | --- | --- | --- | --- |
| A-001 | partial | The July 16 locked source passes every stored V11 source hash when checked on Linux. | H100 `sha256sum -c` against the V11 launcher manifest. | Fresh reproduction is still required. |
| A-002 | in progress | The July 7 archive differs from the locked source in 68 entries: 14 same-path changes and 54 archive-only, untracked historical artifacts. | Content-hash archive audit. | The archive must remain read-only evidence; it cannot be overlaid. |
| A-003 | passed | Current project `images` and `orientations_2` do not match acceptance manifests; the frozen `petsgaussianhair_v11_repro` data copy does match all image, silhouette, orientation, mesh, and camera checks. | Independent SHA-256/manifest audit plus H100 launcher preflight. | Use only the frozen data copy for H100 reproduction. |
| A-004 | open | Audit retained tensors, zero-weight losses, and lifecycle statistics for memory/runtime bugs without changing acceptance behavior. | Profile and tensor-lifetime trace. | Optimization changes become separate experiments. |
| A-005 | resolved | The saved July 17 output `config.json` is not a reliable historical command record: it says zero render/guide split budgets although the retained July 12 phase-A log records `target_direct`, 512 parents x 2 children, and 1,024 inserted roots per event. | `outputs/20260717002440/config.json` versus `petsgaussianhair_v11_repro/logs/20260712162626/phase_a.log` and the retained lifecycle records. | Treat captured phase logs and lifecycle records as runtime ground truth; use saved config only as a secondary artifact. |
| A-006 | passed | The locked V4 configuration chain no longer defined 11 variables that its unchanged runner marks required, so the V4 launcher failed before Python started. The missing values were removed from `white_tiger_stage1_multiroot*.env`, not intentionally changed in the V4 algorithm. | H100 dry-run with `PYTHON=/bin/echo`; after restoring only the retained V11 defaults, the launcher returned `RC=0` with no environment-variable injection. | The configuration inheritance regression is repaired. This is a launcher repair, not an algorithmic change. |
| A-007 | passed | The locked V4 source differs from the verified V11 source only in normal-aware clean-flow interpolation plus the V4 target path, and six trainer call-site arguments that pass normals into that interpolation. | File-level diff: trainer +6 lines, `clean_flow.py` +49/-10, V4 target path change; strand Gaussian and runner are otherwise identical. | The V4 clean-flow change can be tested as a single, isolated variable once the configuration regression is repaired. |
| A-008 | passed | The historical V11 `RootStatsWindow` assumes `radii` is scalar per Gaussian. H100 gsplat returns two radii values per Gaussian, so the first backward pass aborts before densification. | The four-layout visibility test passed; the official V11 qlogin parent preflight completed model setup, forward, backward, and root statistics in 16.0 seconds. The patch adds only `[N,2] -> amax(dim=1) > 0` visibility semantics while retaining `[N]` and `[1,N]`. | Keep this compatibility repair in the accept line; it is independent of the later batch-device failure. |
| A-009 | open | H100 batch job `124538373` saw an empty physical GPU `GPU-6cdbda3c...` at PCI `97:00.0`, but V11 failed at its first CUDA tensor placement. Separate batch diagnostics, including actual float64 OBJ-vertex transfer, pass on other scheduled H100 devices. | `124537920` and `124538373` fail at `torch.from_numpy(mesh.vertices).to(device)`; `124538174` passes basic allocation and `124538732` passes the actual mesh-vertex transfer. The stable held qlogin uses a different GPU at PCI `96:00.0` and passes the full official preflight. | Do not treat this as a model or data bug, and do not repeatedly submit full jobs to an unverified batch GPU. Run the exact reproduction on the held qlogin GPU; revisit scheduler pinning separately. |

## Experiment Register

| ID | Base | Single change | Status | Result | Artifacts |
| --- | --- | --- | --- | --- | --- |
| R-000 | Historical V11 contract | Fresh exact reproduction audit | ready | - | H100 source/data preflight, V4 flow/surface tests, and repaired launcher audit passed |
| R-001 | Historical V11 baseline | Reproduce the retained V11 phase-A/B command on H100 | passed | Phase A 9k test composite `24.0688`; Phase B 30k test composite `32.1891` versus accepted `32.1814`; `RC=0`. | `/work/anigroom-v11-reference/outputs/v11_reference_h100_qlogin_r002`, including phase-A 5k/9k and phase-B 10k/12k/14k/15k/16k/18k/20k/21k/24k/25k/27k/30k checkpoints plus metrics/configs |
| R-002 | R-001 | Replace only V3 clean-flow interpolation/target with V4 | preflight | V4 source/data hash checks passed; resolved A/B commands exposed one non-flow drift, `lr-calibration=0.0005`, now restored to R-001's `0.0`. | Must use the same resolved config and canonical visualizations |
| R-003 | R-002 | Extend the accepted V4 surface interpolation rule from flow-only sampling to the other groom attributes that are inherited or sampled across guide/render roots | locked | Refresh H100 job `124595795` completed from `c7b51a0` with exit status 0. Phase A 9k train/test composite PSNR `24.1314 / 24.3561`; Phase B final 30k `33.6493 / 32.5895`; best test composite `32.7361` at 29k; final roots `195886`, generated Gaussians `8677451`, peak allocated CUDA memory `15.4 GB`. This is the current baseline for R004. | First run: `/work/anigroom-accept-line/outputs/r003_surface_interp_20260720_182227`; refresh output `/home/wangyy/anigroom-r002-locked/outputs/r003_refresh_c7b51a0_20260721_010454`; `docs/r003_attribute_interpolation.md` |
| R-004 | R-003 | Replace render-root `target_direct` insertion with topology-local evidence parent selection and split/delete lifecycle densification | completed | H100 job `124598861` completed at commit `30f0554d88e6aa60c80f3eaeb18fb3d991d06b86` with `exit_status=0`. Phase A 9k train/test composite PSNR `23.7690 / 24.0019`; Phase B final 30k `33.6484 / 32.6023`; best test composite `32.7423` at 29k. Metrics are effectively tied with R003 refresh (`32.5895` final, `32.7361` best). Final roots `197280`, generated Gaussians `8780735`, peak allocated CUDA memory `15.5 GB`. Runtime was much slower than R003 (`27301s` wallclock versus R003 `13119s`), with long non-eval progress gaps, so R004 is an equivalent-quality lifecycle candidate but not a speed improvement. | `docs/r004_lifecycle_densification.md`; `configs/r004_evidence_localmax_0_9k.env`; `configs/r004_evidence_localmax_9k_30k.env`; HGC output `/home/wangyy/anigroom-r004-localmax/outputs/r004_localmax_30f0554_20260721_051239`; job log `/home/wangyy/logs/r004_localmax_30f0554_20260721_051239/job.log` |

## Change Log

| Time | Change | Reason | Verification |
| --- | --- | --- | --- |
| 2026-07-19 | Created isolated recovery worktree and this ledger. | Prevent the dirty reconstruction line from contaminating accept-line recovery. | Worktree is detached at `09de8c4`; no source overlay applied. |
| 2026-07-19 | Verified the locked source on H100 against the original V11 source-hash manifest. | Confirm that the Linux checkout is byte-compatible with the accepted launch contract. | All trainer, representation, flow, projection, config, and launcher hashes passed. |
| 2026-07-19 | Rejected current-project image/orientation inputs and selected the frozen V11 data copy. | Current data would make any resulting metric non-comparable. | Frozen data manifests match the original launcher values exactly. |
| 2026-07-19 | Built the H100 runtime from the locked Git checkout and verified the original launcher in `VERIFY_ONLY=1` mode. | Validate the complete source/data/runtime contract before GPU training. | Every stored source and data hash passed; Python compilation passed; no training launched. |
| 2026-07-19 | Compared the parent checkpoint runtime configuration with the frozen V11 launcher configuration. | A from-zero baseline must reproduce historical runtime behavior, not merely a similarly named script. | Found a material lifecycle-budget discrepancy; reproduction is paused until the root-growth source is identified. |
| 2026-07-20 | Recovered the retained V11 phase-A log and lifecycle records, and dry-ran the locked V4 launcher with `/bin/echo`. | Resolve the apparent split-budget contradiction without consuming a GPU run. | The phase-A log proves 1,024 direct insertions/event; the V4 launcher currently fails on 11 removed required defaults before Python begins. |
| 2026-07-20 | Ran the locked V4 clean-flow/surface unit tests on H100. | Check the isolated V4 representation before any configuration repair. | `3 passed in 57.37s`; no training source or configuration was changed. |
| 2026-07-20 | Restored 11 runner-required baseline defaults to the two configuration base files. | The values are present in the retained exact phase-A command and were removed from config inheritance, while the runner still requires them. | Pending H100 patched-checkout argument audit; no training schedule or algorithmic parameter changed. |
| 2026-07-20 | Repeated the V4 parent launcher audit on H100 after the configuration repair. | Confirm that the runner resolves its entire argument contract from the repaired config chain rather than from temporary shell variables. | `RC=0`; the expanded command has `root-init-method=stratified`, direct render-root densification from iteration 600 at interval 100 with 512 selected parents x 2 children, and the frozen V4 flow target. |
| 2026-07-20 | Rebuilt the historical V11 reference in `/work/anigroom-v11-reference` on H100 from commit `881245ab` plus the retained archive. | Keep V11 and V4 physically separate and recover the source that the exact local reproduction used. | Archive SHA-256 matches `d495a7...aa554`; all source/data hashes in `reproduce_v11_from_zero.sh` pass in `VERIFY_ONLY=1` mode after adding the exact V3 clean-flow artifact (`70a3c0...2349b`). |
| 2026-07-20 | Applied the isolated portable-radii patch only in the H100 V11 reference checkout and ran the original one-iteration parent preflight. | Verify that root visibility supports the observed gsplat `[N,2]` layout without changing lifecycle behavior. | Standalone layout test passed; qlogin preflight returned `RC=0` after full forward, backward, root-statistics update, and metric emission. |
| 2026-07-20 | Submitted the same parent preflight through H100 batch job `124537920`. | Verify the scheduler path before starting a 30k reproduction. | Job failed before model construction at first CUDA tensor placement despite the qlogin success; scheduler CUDA allocation is now isolated as A-009. |
| 2026-07-20 | Re-ran batch preflight with GPU UUID logging and performed two source-independent H100 allocation diagnostics. | Separate device/runtime instability from the V11 source repair. | The failed V11 batch sees empty GPU `...6cdbda3c` at `97:00.0`; both basic and actual float64 OBJ vertex transfers pass on other scheduled H100 devices. |
| 2026-07-20 | Started R-001 through the held qlogin H100 instead of repeatedly submitting to unverified batch devices. | Preserve the exact V11 route while avoiding the isolated scheduler-device defect. | New run ID and log path are recorded in the launch command; Phase A must create `checkpoint_009000.pt` before Phase B begins. |
| 2026-07-20 | Completed R-001 Phase A on the held H100 and handed its checkpoint to the historical V11 continuation. | Verify the 0-to-9k parent route before allowing any later V11 behavior to be interpreted. | At iteration 9000, train/test composite PSNR is `23.8249 / 24.0688` (historical target `24.0695`); 176,724 evaluated roots and 7,073,665 Gaussians. `checkpoint_009000.pt` was loaded with model/RNG state and the optimizer reset; Phase B began at iteration 9001. |
| 2026-07-20 | Completed R-001 Phase B on the same held H100. | Finish the exact accepted V11 reproduction before any single-variable V4 or new-line experiment. | `RC=0`; 30k train/test composite PSNR is `33.1498 / 32.1891` against accepted test `32.1814`; 187,988 roots and 8,498,929 generated Gaussians. Intermediate test composite PSNR also matched the retained curve: 10k `28.4588`, 12k `30.0232`, 16k `31.2821`, 20k `31.9271`, 25k `32.0800`. Peak allocated memory was `15.04 GB` and post-eval allocated memory returned to about `2.08 GB`. |
| 2026-07-20 | Audited R-002 against the exact R-001 parent and continuation commands on a held H100. | Enforce the promised one-variable V3-to-V4 comparison before consuming another full training run. | Source/data `VERIFY_ONLY=1` passed. The expanded Phase A and B commands differ only in V3/V4 clean-flow target plus an inherited `LR_CALIBRATION=0.0005` drift. The latter is reset to the R-001 value `0.0`; the portable gsplat-radii compatibility check is added to the R-002 source lock. |
| 2026-07-20 | Declared R-003 as the next single-variable experiment and audited the current densification implementation before code changes. | Prevent interpolation work from being mixed with lifecycle-policy changes. | `docs/r003_attribute_interpolation.md` records that current render-root densification is direct residual-pixel insertion with retained parents, current guide-root densification is capped retained-parent insertion, and the overlap-aware parent-replacement line must be recovered separately before use. |
| 2026-07-20 | Implemented the R-003 surface-aware typed attribute interpolation in the accept-line worktree. | Apply the accepted V4 surface interpolation principle to guide-to-render sampling and newborn guide/render root inheritance without changing lifecycle policy. | Python compilation passed; a two-face surface interpolation regression passed; `conda activate mygs` trainer import passed; a mini model exercised guide-to-render interpolation, render-root insertion, guide-root insertion, and cache rebuild successfully. |
| 2026-07-20 | Fixed the R-003 high-rank interpolation bug exposed at the Phase B handoff. | Phase B interpolates child attribute tensors such as `[N, child_count, 3]`, while the first R-003 helper only handled scalar and matrix attributes. | Local and H100 high-rank interpolation regressions passed; Phase B restarted from the already completed 9000-step checkpoint and proceeded beyond the failing iteration. |
| 2026-07-20 | Completed the R-003 H100 30k run. | Measure the single interpolation change under the same accept-line contract. | Phase A 9k train/test composite PSNR `24.1189 / 24.3421`; final 30k `33.6514 / 32.6048`; best test composite `32.7515` at 29k; peak allocated CUDA memory about `15.6 GB`; no OOM or traceback after the interpolation fix. |
| 2026-07-20 | Generated R-003 canonical attribute and pure-fur visualizations. | Confirm whether the interpolation change improves structure without hiding failures behind RGB metrics. | Visuals show no global circular flow, but edge, belly, leg, tail, and top/back regions still have direction outliers, drapey long fur, blocky length patches, and bend noise. R-003 is accepted as the interpolation baseline, not as the final Stage 1 visual solution. |
| 2026-07-21 | Ran the R-003 pre-lock interpolation audit and fixed the clean-flow startup path. | Check for implementation errors before tagging R003 as the new baseline. | Found that the accepted clean-flow route should not run `initialize_guide_controls_from_roots()`, because guide roots are already initialized directly from the clean-flow target. The trainer now skips that render-root-to-guide backfill when `GUIDE_ROOTS_FROM_CLEAN_FLOW`, `CLEAN_FLOW_TARGET`, and `CLEAN_FLOW_INIT` are active. The retained non-clean-flow path uses topology-local face-ring support, typed physical interpolation, and transported 3D direction interpolation. Compilation, high-rank interpolation regression, trainer import, lifecycle synthetic test, render/guide attribute coverage checks, and a mini guide-init path check passed. Because this affects from-zero Phase A, the final R003 baseline tag is paused until a refresh run completes. |
| 2026-07-21 | Added `scripts/server/run_r003_from_zero.sh`. | Refresh R003 without reusing the old V11 source-hash launcher. | The launcher keeps the same Phase A/B command contract and frozen V4 clean-flow target, but checks the R003 source hashes including `anigroom/surface_interpolation.py`. `bash -n` passed locally after activating `mygs`. |
| 2026-07-21 | Added the R-004 topology-local evidence densification interface. | Test the recovered overlap-aware lifecycle idea as one isolated variable after R003 is locked. | `tools/test_root_lifecycle_local_max.py` proves adjacent high-evidence roots are suppressed in favor of mesh-neighborhood local maxima; existing lifecycle and split-interpolation regressions still pass. |
| 2026-07-21 | Locked the R-003 refresh baseline and verified R-004 on HGC. | R004 must start from a completed, comparable baseline rather than from the first R003 run with the redundant clean-flow backfill. | R003 job `124595795` finished with `exit_status=0`, final test composite PSNR `32.5895`, best `32.7361` at 29k. R004 `VERIFY_ONLY=1` passed at commit `5c0abf0e45678c091ba89b5d13b0976463e84b4d` against frozen mesh/data/flow hashes. |
| 2026-07-21 | Repaired the R-004 independent-checkout path contract. | The first R004 qsub exposed that `apply_alignment_to_namespace()` overwrote explicit CLI `--data-root` and `--mesh-path` with alignment-json defaults. The second run exposed the same config-source precedence issue for `CLEAN_FLOW_TARGET`. These were invisible in the R003 checkout because the default data and flow target existed there, but invalid for the isolated R004 checkout. | Local `py_compile` and `bash -n` passed. This changes explicit path precedence and launcher wiring only; lifecycle/densification code is unchanged. |
| 2026-07-21 | Completed the R-004 H100 30k run. | Measure the topology-local evidence split/delete lifecycle as a single-variable replacement for R003 direct insertion. | H100 job `124598861` finished with `exit_status=0`. R004 matched R003 metrics within noise: 16k test composite `31.7875` versus R003 `31.7910`, 20k `32.3642` versus `32.3734`, 29k best `32.7423` versus `32.7361`, 30k final `32.6023` versus `32.5895`. Final roots increased to `197280` versus R003 `195886`. Peak allocated CUDA memory stayed normal at `15.5 GB`, but wallclock increased to `27301s`; speed analysis shows recent normal training segments around `2.7-3.1 it/s`, while long progress gaps make the run much slower end-to-end. |
