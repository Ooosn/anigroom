# R046 Secondary-Guide No-Geometry-Regularizer Diagnostic

## Status

Prepared as an isolated causal diagnostic. R043 remains the accepted baseline;
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
