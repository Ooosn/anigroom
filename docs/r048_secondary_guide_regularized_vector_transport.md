# R048 Regularized Secondary-Guide Vector Transport

## Status

Prepared as the formal candidate after the R047 causal fix. R043 remains the
accepted baseline until this run is complete.

## Question

R047 proved that 20k secondary guides can receive RGB direction gradients once
zero-centered vector transport is differentiable, but removing all geometry
regularization allowed direction residual P95 to grow to 0.4359 by 16k. R048
asks whether the fixed field remains learnable while the already-tested local
K4 geometry regularization controls those outliers.

## Single Variable

Resume the exact R044 10k checkpoint. Use the complete R045 configuration,
including 20k G1 roots, render-to-G1 K8 interpolation, G1 graph K4, all normal
losses, schedules, optimizer/RNG state, views, renderer, and root topology.

The only difference from R045 is the R047 code fix replacing
`normalize(v) * norm(v)` with linear vector-field parallel transport for
zero-centered direction residuals.

## Gate

1. Direction residual must leave zero from the repaired RGB path.
2. Direction P95 and residual smoothness must remain controlled relative to
   R047's unregularized growth.
3. Compare test composite PSNR at 12k/14k/16k with R043-R047.
4. Inspect canonical RGB and pure-strand structure before promotion.

