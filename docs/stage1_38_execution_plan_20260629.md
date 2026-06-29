# Stage 1 38+ Execution Plan 2026-06-29

This document is the active execution contract for the next white-tiger Stage 1 work.
Follow this document before code, config, training, or server actions.

## Objective

Produce a Stage 1 white-tiger reconstruction that can become the next formal
baseline:

- composite PSNR on the single-view view09 checkpoint reaches 38+;
- fur structure is visually plausible, without long-strand color dragging;
- mechanisms are clean enough to extend to multiview/server training;
- no fallback, fake renderer, reduced-resolution shortcut, or old polluted config
  is allowed in the formal path.

If 38+ cannot be reached after the mechanisms below are implemented and analyzed,
stop with evidence and discuss the real blocker instead of continuing random
tuning.

## Non-Negotiable Rules

1. Composite PSNR is the primary metric. Raw PSNR is diagnostic only.
2. Render roots start from 100k global mesh-surface roots for the formal route.
3. Guide roots must start low. Do not start with 8192 guide roots as the default
   formal route.
4. Stage 1 starts with conservative short-hair geometry, but length is not
   permanently locked. The system must still support long-haired animals later.
5. Render-root densification happens first. Guide-root densification starts only
   after render-root coverage and long-strand diagnostics are stable.
6. Densification interval is 100 iterations unless a documented formal reason
   changes it. Do not silently use 1000-iteration densification.
7. Old screen/dark/luma hardcoded overlong heuristics are not formal mechanisms.
   They may be used only as diagnostics, not as the accepted solution.
8. Three required validation points must be checked before claiming the baseline:
   child strands/clump, RGB-to-flow or edge-style loss, and cleaned flow for guide
   initialization.
9. All new mechanisms need visualization and diagnostics. A metric-only run is
   not enough.
10. Server runs must use the same git state and formal config as local preflight.

## Current Diagnosis

The current structured multilevel run is not the final route:

- it used 70k render roots, not the required 100k;
- it used 8192 guide roots at startup, which is too many and encourages local
  overfitting instead of smooth guide control;
- it did not implement delayed guide-root densification;
- guide length/control still had enough freedom to create long local strokes even
  when render length residual was disabled;
- previous overlong/screen split runs reached high PSNR but relied on too many
  hardcoded parameters and are not acceptable as the final algorithm.

The real problem is not only local long strands. The deeper issue is that RGB
detail fitting and groom-geometry control are not yet cleanly separated.

## Target Formal Pipeline

### Stage 1-A: Smooth Guide, Dense Render Coverage

Purpose: let dense render roots cover image detail while low-count guide roots
provide smooth grooming geometry.

Required behavior:

- render roots: 100k initial mesh roots;
- guide roots: low-count initial guide field;
- guide roots own low-frequency length, flow, bend, curl/frizz tendency, width
  tendency, and clump tendency;
- render roots own local color/opacity/detail;
- render-root xyz/barycentric movement remains differentiable and regularized;
- render-root densification is active and runs on 100-iteration evidence windows;
- guide root position/count is held fixed during early coverage growth;
- geometry starts short/conservative so missing coverage requests more roots,
  not long color-dragging hairs;
- guide and render roots each have their own smoothness/regularization. They are
  related but not the same loss or same strength.

### Stage 1-A Late: Guide Densification

Guide-root densification starts only after render-root coverage has stabilized.

It is used to increase grooming-field resolution, not to fit pixel stripes
directly. A guide split must be justified by stable residual/coverage evidence
over neighboring render roots and should initialize from neighboring guide
values, not from a hardcoded visual category.

Required diagnostics before enabling it:

- render-root coverage map;
- long-strand/root statistics;
- per-region residual map;
- guide interpolation error or guide-field residual map.

### Stage 1-B: Controlled Residual Unlock

Purpose: recover PSNR after structure becomes stable.

Required behavior:

- render-root residuals are unlocked gradually;
- length, bend, curl/frizz, width, child radius, and clump residuals may unlock at
  different rates;
- residuals have their own smoothness/regularization;
- long-strand replacement may be used only if the criterion is evidence-based and
  general, not hardcoded by tiger stripe luma or screen category;
- densification remains active if evidence shows missing coverage; it should not
  be replaced by overlong hair growth.

## Required Validation Points

### 1. Child Strands / Clump

Question: do child strands/clump actually help structure and PSNR?

Run controlled comparisons:

- child_count = 1 baseline;
- child_count > 1 with differentiable child radius/clump;
- compare composite PSNR, long-stroke diagnostics, alpha, and high-resolution
  crop visualizations.

Do not claim child strands are useful until this comparison is done.

### 2. RGB-To-Flow Or Edge-Style Loss

Question: can a differentiable image-structure loss improve fur direction without
depending on noisy raw orientation maps?

Study and test:

- RGB/pred image passed through a fixed flow/edge/structure extractor;
- compare against current strand-splat orientation loss;
- measure whether it reduces long color-dragging strokes and improves high-res
  hair structure.

If it is unstable or expensive, keep it diagnostic only.

### 3. Clean Flow For Guide Initialization

Question: can cleaned high-confidence flow initialize guide roots better than raw
projected flow?

Required analysis:

- visualize raw orientation, confidence, cleaned anchors, projected guide values,
  and final initialized hair;
- reject obvious stripe-induced directions;
- use local coherence, mask confidence, view angle, depth visibility, and
  multiview agreement if multiview is used;
- if cleaned flow hurts or is ambiguous, keep root-direct raw projection as the
  formal default and document why.

## Visual Evidence Required

For each accepted run, save:

- full-resolution prediction;
- composite prediction;
- GT;
- diff map;
- alpha map;
- high-resolution crops of body, belly, tail, legs, and head;
- hair-only visualization without direction coloring;
- guide-root overlay;
- render-root overlay;
- long-root diagnostics;
- densification lifecycle log;
- root/guide smoothness diagnostics.

## Implementation Order

1. Clean formal config so it cannot point to 70k/8192-guide or old overlong
   heuristic runs.
2. Implement or verify 100k render-root formal config.
3. Implement low-guide Stage 1-A with delayed guide-root densification.
4. Verify render-root densification on 100-iteration evidence windows.
5. Add diagnostics proving whether long strokes come from initialization,
   guide field, render residuals, child strands, or densification.
6. Run the three mandatory validation points.
7. Tune only mechanisms that pass visual and metric analysis.
8. Push/transfer the exact code/config to server.
9. Run server training only after local preflight confirms config, CUDA/gsplat,
   depth clipping, backing compositing, and metric path.
10. Continue until composite PSNR reaches 38+ or a real blocker is documented.

## Server Discipline

- Use the server only after the local formal config is clean.
- Server must pull the exact git state, not old copied code.
- Do not use server-side old configs as references.
- Do not reduce resolution to fit memory.
- If memory is excessive, fix chunking or tensor retention, not model semantics.
- Keep logs and output names time-based and neutral.

## Current Checkpoint For This Round

The next accepted checkpoint must answer:

1. Does low-guide + 100k render root beat the current 36.7 composite PSNR
   structured result?
2. Does delayed guide-root densification improve structure without adding
   hardcoded tiger-specific thresholds?
3. Do child strands/clump help or hurt?
4. Does RGB-to-flow/edge-style loss help or hurt?
5. Does cleaned flow improve guide initialization or introduce new bias?
6. Is the remaining failure caused by initialization, guide smoothness,
   residual unlock, insufficient roots, or renderer/depth/compositing?

Do not report success without these answers.
