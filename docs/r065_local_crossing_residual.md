# R065 Local Crossing Residual

Status: implementation and frozen-checkpoint gradient calibration passed;
formal from-zero 30k acceptance is pending.

## Question

Can inter-strand crossing be reduced without changing primary-guide length or
the low-frequency groom?

R063 allowed crossing gradients to alter length, root placement, and shape.
R064 removed length and root placement from that route, but still updated the
shared primary direction and brush-stiffness field. Checkpoint decomposition
shows that this caused ordinary RGB/flow optimization to compensate with two
extreme primary-guide lengths near the head.

## Evidence

R062, R063, and R064 are identical at 9k. At 10k, after crossing becomes
active, their maximum primary-guide lengths are:

| Run | guide max at 9k | guide max at 10k |
| --- | ---: | ---: |
| R062 | 0.015163 | 0.115777 |
| R063 | 0.015163 | 0.157243 |
| R064 | 0.015163 | 0.177182 |

The final R064 roots above 0.12 are driven by only 16 primary guides. Their
secondary/render length multipliers average 1.024, while the primary length
averages 0.130. The local residual is therefore not the source of the tail.

## Method Variable

The crossing forward term, active-set discovery, refresh interval, and weight
remain unchanged from R063/R064. Only backward ownership changes:

- crossing may update the active dense zero-centered local `direction`,
  `curl radius`, and `frizz amplitude` residuals;
- crossing may not update primary guides, length, root placement, width, or
  appearance;
- configuration validation fails if no active local direction residual exists.

This follows the multilevel representation: primary guides own the smooth
semantic groom, while local residuals resolve local geometric validity.

## Acceptance Protocol

1. all unit tests pass;
2. frozen-checkpoint calibration proves non-zero local residual gradient and
   zero primary-guide/length/root/appearance crossing gradient;
3. native full-resolution from-zero 30k H100 training completes;
4. strict checkpoint reload, fixed eight-view RGB decomposition, 100k-strand
   structure/crossing audit, no-penetration audit, and canonical assets pass;
5. R065 is accepted only if it reduces R062 crossings without the R063/R064
   long-strand tail or a material appearance regression.

## Frozen-Checkpoint Calibration

The complete 173-test suite passed on the held H100 checkout. Calibration used
the immutable R062 30k checkpoint and its 471,583 render roots / 5,475,249
generated Gaussians. The discovered active set contained 365,280 pairs.

| Quantity | Value |
| --- | ---: |
| unweighted local crossing gradient L2 | 0.01343185 |
| weighted existing structural gradient L2 | 0.0000399193 |
| crossing weight for equal L2 | 0.00297199 |
| retained crossing weight | 0.001 |
| retained crossing / structure gradient ratio | 0.3365 |

All three non-zero crossing-gradient tensors belong to
`secondary_geometry_residual`; no primary-guide, length, root, width, or
appearance tensor belongs to the routed parameter set. The unchanged `0.001`
weight therefore preserves the original one-third-gradient calibration without
a new sample-specific tuning parameter.
