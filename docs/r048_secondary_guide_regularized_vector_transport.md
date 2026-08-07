# R048 Regularized Secondary-Guide Vector Transport

## Status

Completed and rejected as a promotion. R043 remains the accepted baseline.
R048 is the controlled validation of the R047 differentiability fix under the
normal R045 geometry regularization.

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

## Formal Result

Run location:

```text
/home/wangyy/anigroom-r048-regularized-vector-transport-runtime-20260808/outputs/r048_regularized_vector_transport_resume10k_16k_h100_20260808
```

The one-H100 continuation completed without fallback, OOM, lifecycle changes,
or root-topology changes and peaked at 10.33 GB allocated memory.

| Iteration | R043 accepted | R045 old transport + K4 | R047 fixed transport, no regularization | R048 fixed transport + K4 |
| ---: | ---: | ---: | ---: | ---: |
| 12000 | 30.7874 | 30.5920 | 30.6070 | 30.5906 |
| 14000 | 31.3715 | 30.8607 | 30.8951 | 30.8615 |
| 16000 | 31.7395 | 31.0501 | 31.1146 | 31.0503 |

R048's direction residual is trainable, but its normal regularizers return the
trajectory almost exactly to R045:

| Iteration | R045 direction P95 | R047 direction P95 | R048 direction P95 |
| ---: | ---: | ---: | ---: |
| 12000 | 0.0246 | 0.0391 | 0.0248 |
| 14000 | 0.0397 | 0.0987 | 0.0395 |
| 16000 | 0.0567 | 0.4359 | 0.0570 |

The fixed transport therefore repairs a real RGB-gradient path, while the
normal geometry losses dominate the learned field strongly enough that the fix
alone changes 16k test composite by only +0.0002 dB relative to R045. Removing
those losses entirely is not a solution: R047 gains only +0.0643 dB over R048
and develops an uncontrolled direction tail.

## Conclusion

Do not increase the 20k G1 population based on this run. All 20k rows are
active and supported, and R048 does not establish population sparsity. The next
audit must measure the representation itself:

1. project R043's learned render-root residual field into the fixed 20k basis
   and report the best attainable reconstruction error;
2. measure render-to-G1 interpolation radius, weight entropy, and per-parent
   density variation;
3. only if the basis upper bound is poor, test a smaller render support or a
   density-balanced placement as isolated variables.

The current placement assigns four or five G1 roots to every primary guide,
while the primary-cell area proxy varies from 17 to 140 dense candidates. This
8.24x variation is a concrete local-density risk even though the global count
is 20k; it must be measured rather than hidden by adding more roots.
