# R064 Shape-Only Strand-Crossing Constraint

## Status

R064 is complete and rejected as a baseline. It is retained as the immutable
shape-only ownership control between R063 and the accepted R065 local-residual
route.

## R063 Finding

R063 proved that the exact 3D active-set formulation can remove physical
crossings. Against the matched R062 100k-strand protocol, R063 reduced:

- all unique contacts from `16,291` to `12,834`;
- contacts at least 45 degrees from `230` to `56`;
- strands involved in at least 45-degree contacts from `379` to `95`.

The fixed eight-view composite PSNR changed only from `33.21203` to `33.17865`.
However, the R063 strand audit found 20 strands longer than `0.12`, including
one at `0.15085`, while R062 had none. All 20 roots belonged to the final
crossing active set. The two roots shared by the deterministic R062 and R063
100k samples grew from `0.04127/0.04568` to `0.13003/0.13475`.

This is not accepted as a length rule or anatomical exception. It exposes an
ownership error: the R063 crossing backward route was allowed to update strand
length and render-root barycentric position, so contact removal could escape by
stretching a strand or moving its root.

## R064 Change

The crossing loss keeps the exact R063 contact discovery, detached Gaussian
width envelope, continuous angle weight, active-set refresh, and dimensionless
loss. Only its trainable parameter ownership changes.

Crossing gradients may update:

- 3D direction;
- brush stiffness;
- curl radius, turns, and phase;
- frizz amplitude.

Crossing gradients may not update:

- strand length;
- render-root barycentric position;
- width, opacity, color, or Gaussian RGB residual;
- global scale or translation.

This is a semantic ownership rule rather than a white-tiger threshold: crossing
is a centerline-shape validity constraint, while length and root density remain
owned by reconstruction, flow, interpolation, smoothness, and lifecycle.

The training backward router and calibration tool share one parameter predicate
so the measured gradient scale is the scale used by formal training.

## Frozen Calibration

Calibration used the accepted R062 checkpoint at iteration 30,000. The complete
test suite passed (`170 passed`). The unweighted shape-only crossing gradient
had L2 norm `0.0536521`; the already weighted structural-gradient L2 norm was
`0.0000730371`, giving an equal-gradient crossing weight of `0.00136131`.

The formal weight remains `0.001`. It is below the equal-gradient value and is
identical to R063, so R064 tests gradient ownership rather than retuning the
constraint for this subject.

## Formal Gate

R064 must pass all of the following before acceptance:

1. all tests, including explicit gradient-ownership tests;
2. calibration on the frozen R062 30k checkpoint;
3. native full-resolution from-zero 30k training with no fallback;
4. strict checkpoint reload and fixed eight-view RGB evaluation;
5. deterministic 100k-strand exact crossing and structure audits;
6. all-root mesh no-penetration audit;
7. canonical single-image Blender assets and crossing/length highlights.

The comparison uses R062 and R063 as immutable controls. R064 is rejected if it
recovers crossing counts by introducing long strands, foldbacks, penetration,
visible local collapse, or a material reconstruction regression.

## Formal Result

The native full-resolution from-zero 30k run, strict reload, fixed eight-view
evaluation, 100k-strand audit, no-penetration audit, and canonical assets all
completed. R064 improved the exact crossing tail relative to R062, but failed
the length-ownership gate.

| Metric | R062 | R064 |
| --- | ---: | ---: |
| fixed eight-view composite PSNR | 33.21203 | 33.19708 |
| all exact contacts | 16,291 | 14,762 |
| contacts at least 45 degrees | 230 | 113 |
| contacts at least 60 degrees | 75 | 29 |
| sampled strand length max | 0.105264 | 0.154185 |
| sampled strands above 0.12 | 0 | 39 |
| checkpoint effective roots above 0.12 | 0 | 162 |
| primary-guide length max | 0.104955 | 0.162751 |
| primary guides above 0.12 | 0 | 2 |

The crossing term no longer changed length directly, yet updating the shared
primary direction/stiffness field caused ordinary reconstruction losses to
compensate through primary-guide length. R064 therefore removed crossings by
distorting the low-frequency groom rather than resolving them locally. It is
rejected without adding a length cap; R065 addresses the ownership error.
