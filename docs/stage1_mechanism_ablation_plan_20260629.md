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
