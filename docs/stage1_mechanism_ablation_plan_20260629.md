# Stage 1 Mechanism Ablation Plan 2026-06-29

This document follows the 38+ checkpoint run `20260629033147`.

The goal is not to increase iteration count. The goal is to identify which
mechanisms actually make the Stage 1 grooming system work.

## Fixed Evaluation Contract

- data: white tiger view09
- resolution: 1920x1080
- iterations: 3200
- metric: composite PSNR first, raw PSNR diagnostic
- renderer/depth/random backing: same as baseline unless the ablation explicitly
  changes it
- no fallback, no low-resolution substitute, no old overlong/screen/dark
  heuristic path

Baseline:

- config: `configs/white_tiger_stage1_multiroot_stageab_100k_lowguide_view09.env`
- run: `20260629033147`
- final: composite PSNR 39.854, raw PSNR 35.736

## Required Ablations

### A. No Render-Root Densification

Config: `configs/white_tiger_stage1_ablate_no_render_densify_view09.env`

Question:

- Is render-root lifecycle coverage actually responsible for the 38+ result?

Expected interpretation:

- If this stays far below baseline, render-root densification is necessary.
- If it reaches near baseline, densification is not the core reason and the
  current route may simply be fitting through other capacity.

### B. No Guide Residual Unlock

Config: `configs/white_tiger_stage1_ablate_no_guide_residual_unlock_view09.env`

Question:

- Is the late-stage improvement caused by residual freedom, or by simply running
  longer?

Expected interpretation:

- If it plateaus near the 1600 baseline, residual unlock is necessary.
- If it still reaches 39+, then 3200 iterations alone explain most of the gain.

### C. One Child Strand

Config: `configs/white_tiger_stage1_ablate_child1_view09.env`

Question:

- Do child strands/clump-like multiplicity help structure, or are they only
  increasing Gaussian count?

Expected interpretation:

- If PSNR and visual quality drop, child strands are useful.
- If quality is similar, child strands are not justified yet.

## Evidence To Collect

For each run:

- final metrics line;
- PSNR trajectory every 200 iterations;
- final prediction, alpha, diff, orientation visualization;
- root count and Gaussian count;
- high-capacity root diagnostic;
- visual judgement of long-strand color dragging.

## Decisions After Ablations

Only mechanisms with both metric and visual support should remain in the formal
route. Mechanisms that do not trigger or do not improve the result must be
marked diagnostic or redesigned.

## Completed Results

All runs used view09, 1920x1080, 3200 iterations, composite PSNR as the primary
metric, and raw PSNR as a diagnostic metric.

| run | config | final composite PSNR | final raw PSNR | final roots | final Gaussians | conclusion |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `20260629033147` | baseline | 39.854 | 35.736 | 139898 | 8986687 | reference route |
| `20260629050205_ablate_no_densify` | no render-root densification | 37.671 | 34.718 | 100000 | 6224052 | render-root densification is necessary |
| `20260629054243_ablate_no_residual` | no guide residual unlock / no guide densification | 39.539 | 35.533 | 139881 | 9035188 | guide-side refinement is a small gain, not the core reason |
| `20260629062953_ablate_child1` | one child strand | 39.353 | 35.636 | 144490 | 2316721 | child strands help, mainly coverage/stability, but are not the core reason |

Trajectory summary:

- No render-root densification stayed fixed at 100000 roots and ended 2.183 dB
  below baseline. It reached only 37.671 at 3200 even with the same residual
  unlock schedule, so the 39+ result is not explained by iteration count alone.
- No guide residual unlock / no guide densification matched the baseline before
  the guide phase and still reached 39.539. This means the current single-view
  gain is mostly from render-root densification and continued optimization,
  while guide-side refinement adds about 0.315 dB in this run.
- One child strand started much worse, recovered after densification and
  residual unlock, but still ended 0.501 dB below baseline. Child strands are
  useful as local coverage/thickness capacity. They should stay, but they should
  not be described as the primary mechanism.

Visual summary:

- No render-root densification does not collapse visually, but it has thinner
  coverage and softer local details, especially around the belly, legs, and
  stripe boundaries.
- No guide residual unlock / no guide densification looks close to baseline.
  Current evidence does not justify treating guide-side refinement as essential
  for the view09 result.
- One-child output is visually plausible, but it has weaker fur-layer thickness
  and less stable early optimization. The drop at iteration 1800 indicates that
  the residual phase disturbs the single-child structure more than the baseline.

Current decision:

- Keep render-root densification as a formal Stage 1 mechanism.
- Keep child strands in the formal route, but treat their role as coverage and
  local thickness, not as the main innovation by itself.
- Keep guide residual unlock and guide-root densification as optional
  refinements for now; they need multi-view validation before being treated as
  essential mechanisms.
- Guide-root densification did fire in the 38+ baseline: it ran from iteration
  1800 to 2600 and increased guide roots from 2048 to 4352. However, the
  ablation that disabled guide-side refinement still reached 39.539, so the
  current evidence says this path is helpful but not the main cause of the 39+
  view09 result.
