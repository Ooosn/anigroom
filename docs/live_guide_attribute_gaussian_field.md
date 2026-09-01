# Live Guide-Attribute Gaussian Field V1

Status: isolated representation, synthetic code gate, and Panda R080
iteration-4000 initialization probe completed. K8 is retained only as the next
fixed-checkpoint candidate. No Stage 1 trainer, config, checkpoint-schema,
lifecycle, SH, canonical-visualizer, or formal-baseline integration is
authorized by this document.

## Objective

Replace hard truncated guide-to-render KNN interpolation with one positive,
smooth, differentiable Gaussian attribute field. The existing 4,500 primary
guide roots remain the only low-frequency controls. Render roots sample the
field. Neither guide-root nor render-root XYZ moves during ordinary optimizer
steps; population geometry changes only at explicit lifecycle prune/split
events.

The first gate establishes initialization, parameter domains, complete field
evaluation, gradients, influence support, fixed-checkpoint scalar behavior,
and execution cost. It does not train or change the accepted R068 route.

## Parameter Ownership

Fixed buffers:

- guide XYZ;
- render-root XYZ, native face identity, and barycentric coordinate;
- candidate guide IDs and CSR row offsets;
- guide-local reference scale;
- candidate-envelope bounds;
- any later guide-to-render direction transport geometry.

Trainable V1 state:

- three bounded guide log-scale coordinates;
- one normalized guide rotation quaternion;
- explicit guide attribute values supplied to the field.

The positive-definite guide covariance is derived, never optimized as an
unconstrained matrix:

```text
Sigma_g = R_g diag(sigma_g^2) R_g^T.
```

Guide/render XYZ, SH coefficients, and guide lifecycle are not V1 trainable
state. SH and guide densification are separately gated future extensions.

## Initialization And Bounds

For guide `g`, let `d_k(g)` be the Euclidean distance to its requested Kth
distinct non-self guide neighbor. The isotropic reference scale is

```text
sigma_ref[g] = d_k(g) / support_sigma.
```

V1 defaults are fixed before Panda output review:

- neighbor count: `16`;
- Gaussian support boundary: `3.0 sigma`;
- C2 taper start: `2.5 sigma`;
- per-axis scale ratio range: `[2/3, 3/2]` relative to `sigma_ref`;
- minimum row denominator: `1e-8`.

Raw scale parameters decode in log space and start at ratio exactly one.
Quaternions start at identity and normalize before use. Invalid spacing,
duplicates among guide centers, nonfinite values, uncovered queries, a
nonpositive denominator, or an out-of-domain configuration fails explicitly.
Render query points may coincide and remain separate CSR rows. There is no
padding, nearest-guide fallback, silent radius expansion, or post-output
clamp.

## Conservative Sparse Candidate Binding

The sparse pattern contains every guide whose maximum permitted covariance
envelope can influence a fixed render root. For each guide, initialization
queries all render roots inside

```text
support_sigma * max_scale_ratio * sigma_ref[g].
```

Pairs are converted to deterministic sorted unique query-major CSR. The
candidate pattern is fixed; numerical Gaussian weights remain live. A later
render-root split creates only new rows, pruning deletes rows, and a future
guide lifecycle changes columns and requires an explicitly local rebinding
contract.

## Live Field

For candidate pair `(r,g)`, rotate the fixed world displacement into the
current guide covariance frame and compute Mahalanobis radius

```text
rho_rg^2 = || diag(1 / sigma_g) R_g^T (x_r - p_g) ||^2.
```

The unnormalized influence is

```text
q_rg = exp(-0.5 rho_rg^2) * taper_C2(rho_rg; 2.5, 3.0).
```

The quintic taper equals one below the taper start, reaches exactly zero at
the support boundary, and has zero first and second derivative at both ends of
the transition. Its float32 implementation uses separate numerically stable
start/end forms; it is not clamped after evaluation. Normalized positive
weights are

```text
w_rg = q_rg / sum_h q_rh.
```

Every guide attribute tensor uses the same live weights:

```text
F_r = sum_g w_rg A_g.
```

Raw guide attributes are field coefficients. The effective physical value at a
guide is obtained by evaluating the complete field at that guide; exact raw
coefficient identity is not claimed. Constant fields reproduce exactly, and
all scalar output remains in the convex hull of active coefficients.

V1 code uses an eager PyTorch gather/index-add reference. A fused sparse kernel
is a later execution gate and must match the reference forward and backward
before replacing it.

## Root Gradients And Lifecycle

Fixed render XYZ does not remove lifecycle evidence. During a lifecycle-active
iteration, the fixed render points may be copied into a fresh leaf tensor with
`requires_grad=True`. The current backward then supplies `dL/dXYZ` to the
existing root-statistics accumulator, while no optimizer updates XYZ and no
position gradient persists across iterations.

Current `raw` lifecycle scoring consumes Gaussian mean/scale gradients,
render-root position gradients, and residual evidence. V1 must preserve that
evidence path when trainer integration is later proposed.

## Synthetic Verification

The isolated package is:

```text
anigroom/grooming/guide_attribute_gaussian_field/
```

It contains no trainer, lifecycle, dataset, checkpoint, visualization, or
Panda path import. Synthetic verification covers deterministic candidate CSR,
strict coverage, fixed XYZ buffers, scale bounds, normalized quaternions, SPD
covariance, C2 value/first/second derivatives, float32 taper monotonicity,
nonnegative row-normalized weights, constant/multi-channel fields, support
entry, exact guide/query coincidence, anisotropic rotation gradients, and
dynamic-query gradients.

Two actual-input defects were caught before acceptance:

1. render roots can share exact XYZ, so query duplicates must remain distinct
   CSR rows; only guide centers require uniqueness;
2. the expanded quintic taper polynomial produced tiny negative float32 values
   near the outer boundary. It was replaced by algebraically equal stable
   branches, with no output clamp.

Focused tests report `30 passed`. The complete repository reports
`681 passed`, `14 warnings`, with no failures. `py_compile` and
`git diff --check` pass.

## Panda R080 Iteration-4000 Initialization Probe

The local RTX 4080 probe uses checkpoint SHA-256
`fae9f653cbee6e8b0b56987eb1f270cd804989d296e643a05c2efe742ce4c505`,
`4500` fixed guides, and `496632` fixed render roots. It does not train or
mutate the checkpoint. All three predeclared neighbor counts pass coverage,
finite-gradient, positive-output, convex-hull, and row-sum invariants.

| Metric | K8 | K16 | K32 |
| --- | ---: | ---: | ---: |
| conservative candidate pairs | 9,779,208 | 18,831,651 | 37,572,483 |
| candidate count mean | 19.69 | 37.92 | 75.65 |
| active count P50 / P95 | 9 / 10 | 16 / 19 | 32 / 36 |
| effective-neighbor P50 | 3.71 | 6.93 | 13.61 |
| maximum-weight P50 | 0.407 | 0.244 | 0.134 |
| guide effective/raw relative error P95 | 8.85% | 14.96% | 21.87% |
| candidate/legacy relative difference P95 | 9.85% | 12.85% | 18.20% |
| support-conditioned edge log-jump P95 | 0.06131 | 0.05163 | 0.04205 |
| reduction from legacy `0.11416` | 46.30% | 54.77% | 63.16% |
| eager forward P50, RTX 4080 | 36.66 ms | 32.68 ms | 58.43 ms |
| eager forward+backward P50 | 67.23 ms | 130.14 ms | 264.86 ms |
| peak allocated / reserved | 5.36 / 5.98 GiB | 9.56 / 11.91 GiB | 20.99 / 25.55 GiB |

K8 materially removes the visible guide-cell mosaic while preserving the
checkpoint's large-scale length structure and gives the lowest guide-site
semantic drift. K16 is a smoother but more expensive/less local control. K32
is over-broad and exceeds the local execution-memory boundary; it is rejected
before any trainer proposal.

The edge statistic above uses the inherited support-conditioned render K32
graph and is therefore descriptive rather than causally independent. The next
gate must add a native-root graph that does not use legacy guide support IDs.
The current point-overlay image is retained only for canonical comparability;
it is not evidence about isolated background-colored speckles.

Artifacts:

- report:
  `D:/RTS/_tmp/panda_live_gaussian_init_probe_20260901_retry2/live_gaussian_init_probe.json`;
- manifest:
  `D:/RTS/_tmp/panda_live_gaussian_init_probe_20260901_retry2/manifest.sha256`;
- reliable runner:
  `D:/RTS/_tmp/panda_live_gaussian_init_probe_20260901_runner_retry2/result.json`;
- K8/K16/K32 canonical-comparable maps:
  `D:/RTS/_tmp/panda_live_gaussian_init_probe_20260901_retry2/view09_length_gaussian_k*.png`.

## Current Decision Boundary

K8 advances only to these bounded gates:

1. visible-guide impulse maps with numerical support/mass reports;
2. independent native-root continuity graph;
3. positive transported-direction blend, pre-normalization magnitude,
   reversal, and foldback diagnostics;
4. isolated eager/fused execution comparison and incremental memory;
5. matched white-tiger fixed-checkpoint initialization.

Trainer integration, root-freeze schema changes, SH, and guide densification
remain unauthorized until all five pass.
