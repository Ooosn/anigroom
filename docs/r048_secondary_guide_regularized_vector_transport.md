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

## Fixed-Count Representation Audit

The follow-up audit held the G1 population at exactly 20,000 and solved the
least-squares upper bound for reproducing R043's 16k render-root residual field.
It compared the checkpoint topology against an area-proportional topology. The
latter retained every primary-guide anchor, allocated the remaining roots from
surface-area samples according to each primary guide's surface cell, and used
local FPS inside each cell. No training configuration or accepted baseline was
changed.

Reports:

```text
/home/wangyy/anigroom-r048-regularized-vector-transport-runtime-20260808/audits/r043_16k_into_20k_k4_k8.json
/home/wangyy/anigroom-r048-regularized-vector-transport-runtime-20260808/audits/r043_16k_into_20k_area_proportional_k4_k8.json
/home/wangyy/anigroom-r048-regularized-vector-transport-runtime-20260808/audits/r043_16k_into_20k_checkpoint_k1_k2.json
```

| Placement | Interpolation K | Length explained variance | Direction explained energy |
| --- | ---: | ---: | ---: |
| checkpoint | 1 | 17.006% | 28.494% |
| checkpoint | 2 | 18.837% | 30.746% |
| checkpoint | 4 | 19.298% | 31.495% |
| checkpoint | 8 | 18.781% | 30.934% |
| area proportional | 4 | 19.370% | 31.554% |
| area proportional | 8 | 18.832% | 31.001% |

Area-proportional placement changed the per-primary population from fixed 4-5
to 2-8, but improved the K4 upper bound by only 0.072 percentage points for
length and 0.059 percentage points for direction. Its render-root nearest-node
distance P95 changed only from 0.008554 to 0.008491, and all 20,000 nodes had
nonzero interpolation mass in both placements. Placement imbalance is therefore
not the cause of the observed PSNR gap. K1 and K2 also performed worse than K4,
so four-node interpolation is not erasing capacity through an unnecessarily
wide blend.

## Matched 16k Structure Check

R043 and R048 were exported at the same 16k iteration with 100,000 strands, 32
samples per strand, one child, one uniform material, and the same Blender camera,
mesh, width, and render settings. The independent full-resolution files are at:

```text
D:\RTS\_tmp\r043_r048_16k_structure_compare
```

The selected eight-view composite mean was 32.4775 dB for R043 and 31.6195 dB
for R048. The pure-strand structure, however, was smoother under R048. Four-root
Euclidean-neighbor diagnostics on the same 100k exports measured:

| Statistic | R043 render-root residual | R048 20k G1 residual |
| --- | ---: | ---: |
| neighbor arc-length difference P50 | 0.001021 | 0.000224 |
| neighbor arc-length difference P95 | 0.006150 | 0.002222 |
| neighbor arc-length difference P99 | 0.010121 | 0.004957 |
| neighbor direction difference P95 | 11.297 deg | 11.291 deg |
| strand tortuosity P95 | 1.01251 | 1.00954 |

R043 therefore gains RGB fidelity mainly by allowing high-frequency strand-level
length variation: its median local length jump is 4.55x R048 and its P95 jump is
2.77x R048, while local direction continuity is effectively unchanged. This is
also visible in the matched asset renders: R048 keeps comparable coverage and
global flow but removes local length patchiness.

## Final Diagnosis

The 20k G1 population is not sparse for its intended role as a smooth structural
residual field. The missing R043 residual energy is predominantly variation at
or below the per-strand scale, where 469k independent render-root residuals can
fit RGB texture and stripe-edge evidence through geometry. Reproducing that
field by increasing G1 count would weaken the intended geometry/appearance
separation rather than fix an inactive-node, placement, or interpolation bug.

Keep 20k as the current structural basis. The next method-level improvement
should route high-frequency RGB evidence through an appearance representation
instead of asking the structural G1 field to recover R043's per-strand length
noise. R043 remains the accepted baseline and R048 remains a diagnosed,
non-promoted experiment.
