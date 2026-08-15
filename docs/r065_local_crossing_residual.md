# R065 Local Crossing Residual

Status: complete and accepted as the current advanced
geometry/appearance/validity/crossing baseline.

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

## Formal Result

The uninterrupted native 1920x1080 H100 run completed all 30,000 iterations in
14,766.96 seconds. It finished with 471,073 render roots, 5,469,186 generated
Gaussians, and 20,371.37 MB peak allocated CUDA memory. Final train/test
composite PSNR is `33.26970/32.19859`; best test composite is `32.28711` at
29k.

The strict fixed eight-view evaluation reports:

| Metric | R062 | R064 | R065 |
| --- | ---: | ---: | ---: |
| composite PSNR mean | 33.21203 | 33.19708 | 33.22230 |
| no-Gaussian-residual mean | - | - | 31.41187 |
| Gaussian-residual gain | - | - | +1.81043 |
| no-shape-detail mean | - | - | 32.76970 |

The deterministic 100k-strand exact crossing and structure audit reports:

| Metric | R062 | R064 | R065 |
| --- | ---: | ---: | ---: |
| all exact contacts | 16,291 | 14,762 | 15,822 |
| contacts at least 45 degrees | 230 | 113 | 198 |
| contacts at least 60 degrees | 75 | 29 | 55 |
| crossing-score P95 | 0.10885 | - | 0.10635 |
| local direction P95 (degrees) | 11.62127 | 11.34429 | 11.50378 |
| maximum local turn P95 (degrees) | 9.83369 | 8.00770 | 9.21152 |
| backward strands | 0 | 0 | 0 |
| sampled length max | 0.105264 | 0.154185 | 0.109623 |
| sampled lengths above 0.12 | 0 | 39 | 0 |

The all-root length-ownership audit confirms that R065 did not trade crossing
for a stretched shared groom: primary-guide/effective/secondary-effective
maximum lengths are `0.112644/0.112774/0.112433`, with zero values above
`0.12`. The corresponding R064 primary-guide/effective maxima are
`0.162751/0.163158`, with 2 primary guides and 162 effective roots above
`0.12`.

The all-root no-penetration audit also remains matched: penetrating points are
6,602 of 29,677,599 (`0.022246%`) and penetrating roots are 1,848 of 471,073
(`0.392296%`), both slightly below R062. Canonical pure-fur assets show no tail
spikes, gross folds, or visible local collapse. Remaining exact crossings are
sparse around the head/neck, tail tip, and a few torso/limb regions.

## Decision

R065 passes the complete acceptance protocol. It reduces R062's exact
45-degree crossing count by 13.9%, preserves reconstruction and collision
validity, and eliminates the R063/R064 long-strand escape route. R064 achieves
a lower crossing count but is rejected because it violates low-frequency
length ownership. R065 becomes the accepted parent for any further crossing
work; it is a clean checkpoint, not a claim that all crossings are solved.

Formal artifacts:

- H100 output:
  `/home/wangyy/anigroom-r065-local-crossing-residual-runtime-20260815/outputs/r065_local_crossing_residual_0_30k_h100_20260815`;
- H100 postprocess:
  `/home/wangyy/anigroom-r065-local-crossing-residual-runtime-20260815/postprocess/r065_local_crossing_residual`;
- local strict QA:
  `D:/RTS/_tmp/r065_acceptance_20260815/postprocess/r065_local_crossing_residual`;
- verified postprocess archive SHA-256:
  `548deb2fc44066374626071628cd73a9b0b7fd3c7f67d06367f227ad24527e10`.
