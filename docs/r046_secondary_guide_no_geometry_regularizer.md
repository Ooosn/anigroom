# R046 Secondary-Guide No-Geometry-Regularizer Diagnostic

## Status

Completed as an isolated causal diagnostic. R043 remains the accepted baseline;
R044 and R045 remain completed experiments.

## Question

R045 showed that reducing the secondary graph from K32 to K4 changes the
physical smoothing span and releases direction residual magnitude, but recovers
only 0.0135 dB at 16k. R046 asks whether the remaining gap is caused by the
combined explicit G1 geometry losses or by the render-to-G1 interpolation basis.

## Single Conceptual Variable

Resume the exact formal R044 `checkpoint_010000.pt`, including optimizer and RNG
state, and continue to 16k. Keep 20k G1 roots, render-to-G1 K8 interpolation,
the K4 diagnostic graph, all RGB/appearance losses, primary-guide losses,
learning rates, schedules, views, renderer settings, and root topology fixed.

Disable only the three losses acting directly on secondary effective geometry:

```text
GEOMETRY_RESIDUAL_SMOOTH_SCALE=0
EFFECTIVE_SMOOTH_WEIGHT=0
CLEAN_FLOW_3D_SMOOTH_WEIGHT=0
```

`SMOOTH_WEIGHT` remains unchanged, so primary appearance-field smoothing is not
removed. The new geometry-residual scale defaults to `1.0`; existing configs and
checkpoints therefore retain their previous semantics.

## Interpretation Gate

- If test PSNR and residual magnitude recover toward R043, explicit geometry
  regularization is suppressing the 20k field.
- If they remain near R044/R045, the fixed K8 interpolation/gradient aggregation
  is the principal bottleneck. The next measurement is an offline projection of
  the R043 render residual field into the fixed 20k basis.

Do not increase G1 count or change learning rate before this distinction is
measured.

## Formal Result

The one-H100 continuation completed from the exact R044 10k checkpoint:

```text
/home/wangyy/anigroom-r046-secondary-guide-runtime-20260808/outputs/r046_no_geometry_regularizer_resume10k_16k_h100_20260808
```

| Iteration | R045 test composite | R046 test composite | R046 - R045 | R046 direction P95 | R046 length P95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12000 | 30.5920 | 30.6025 | +0.0105 | 0.0000 | 0.0150 |
| 14000 | 30.8607 | 30.8687 | +0.0080 | 0.0000 | 0.0235 |
| 16000 | 31.0501 | 31.0552 | +0.0051 | 0.0000 | 0.0369 |

Peak allocated CUDA memory was 10.60 GB. Removing all three explicit
secondary-geometry losses did not recover the PSNR gap. More importantly,
secondary length residuals learned normally while secondary direction
residuals stayed exactly zero for the entire continuation.

## Conclusion

The result rejects both explanations that 20k G1 nodes are spatially too sparse
or that explicit G1 smoothness is the primary bottleneck. Code-level gradient
inspection found that zero-initialized direction residuals passed through
`normalize(v) * norm(v)`, whose autograd derivative at `v = 0` is zero. R046
therefore became the formal control showing the broken RGB-to-direction path.
R047 fixes that vector transport without changing capacity or training values.
