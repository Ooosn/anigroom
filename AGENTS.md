# AniGroom Working Rules

This project is not a smoke test. Do not optimize for quick command completion.
Every action must serve the current research goal: a correct, editable,
mesh-rooted animal grooming representation that can support the paper.

## Current Main Goal

Stage 1 must reconstruct white-tiger fur with correct visibility, projected
evidence, and groom-parameter behavior. The immediate focus is restoring and
reproducing the single-layer root baseline from `20260626015952`: one mesh root
owns explicit groom parameters, deterministic child strands are generated from
that root, and Gaussians are rendered through gsplat.

## Non-Negotiable Rules

1. Do not use fallback implementations, silent degradation, reduced
   resolution, fake renderers, or one-step smoke tests for formal modules.
   If a formal module fails, let it fail and diagnose the real cause.
2. Before changing training logic, first produce visual evidence that the
   input signal is valid. For orientation/flow, this means readable direction
   line visualizations, not just point overlays or confidence heatmaps.
3. Do not add cleaned-flow, anchor-flow, or multi-level root logic to the formal
   path until the restored single-layer baseline is reproduced and committed.
   The current formal path uses the default orientation map loading from
   `tools/train_white_tiger_stage1.py`.
4. Treat black-white tiger stripes as a known source of false orientation.
   Confidence alone is not a valid anchor criterion. Anchor selection must
   consider stripe rejection, local direction coherence, visibility, and
   eventually multi-view consistency.
5. Root visibility must not replace correct per-Gaussian depth clipping.
   Fur rendering should be clipped by mesh depth so back-side/body-hidden
   Gaussians do not contribute to forward or backward.
6. Densification/pruning must follow the confirmed root lifecycle in the formal
   config. Accumulate root evidence over 100-iteration windows, densify from
   root-level evidence, keep tensor/optimizer/root-id mappings synchronized,
   then prune only after enough visibility history.
7. Groom parameters must remain interpretable. Do not introduce carrier,
   hidden density/hairness, latent shortcuts, or unrelated appearance stories
   unless explicitly approved.
8. For white tiger Stage 1, mesh/random backing is only to prevent fur from
   exploiting transparent/white-background blending. It is not a new
   disentanglement story.
9. Do not preserve old experimental code in the formal path. Diagnostic code
   must be clearly separated from formal modules.
10. If the user points out a conceptual issue, stop and analyze that issue
    directly before writing more code.

## Required Pre-Action Checklist

Before any substantive action, answer these internally:

- What is the current module being validated?
- What visual evidence proves the input signal is correct?
- Is this formal code or diagnostic code?
- Am I using the confirmed config, not an old config?
- Am I relying on a fallback, lower resolution, or fake renderer?
- Does the change preserve the paper story: editable grooming parameters
  learned from reliable image evidence?

If any answer is unclear, do not continue with implementation. Inspect, visualize,
or ask.

## Current Orientation Decision

The formal restored baseline uses the default orientation map and confidence
path from the recovered `20260626015952` code. Later cleaned-flow or anchor-flow
experiments were removed from the formal path. If they are revisited, they must
be isolated in a new branch and compared against the restored baseline one
change at a time.

## Metric Reporting

Composite PSNR is the primary Stage 1 PSNR. Raw RGB PSNR may be logged as a
diagnostic, but summaries, comparisons, and server status reports should lead
with composite PSNR.
