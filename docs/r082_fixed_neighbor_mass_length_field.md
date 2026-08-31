# R082: Fixed-Neighbor-Mass Continuous Length Field

Status: Panda fixed-checkpoint gate completed and rejected; no white-tiger,
visualization, config, checkpoint, or training integration.

Local implementation at 2026-09-01 adds only a diagnostic candidate-K
override and an exact linear-memory support-intersection counter. The field
helper and all formal training behavior remain unchanged. Complete local
verification reports `492 passed`; deterministic width-65 support tests match
an independent pairwise reference exactly.

## Question

R081 proved that a compact Wendland C2 window with only eight active primary
guides is too concentrated. It lowers unchanged-support edge variation but
worsens support-boundary jumps and reduces the median effective guide count.

R082 asks one isolated question: does the standard particle-method principle
of a larger fixed neighbor mass let local guide density control physical field
bandwidth without sharpening the support seam?

## Exact Parent And Single Variable

R082 uses the unchanged R081 helper, arithmetic positive-length combination,
Panda R080 iteration-4000 checkpoint, learned guide values, render roots,
surface distances, exact render graph, validation, and no-training contract.

The only candidate variable is:

```text
active compact-kernel guide count: 8 -> 32
boundary support width:             9 -> 33
```

K32 is fixed before observing R082 output. It matches the active render-surface
graph neighbor count and lies at the established lower end of common SPH
neighbor populations. It is not selected from a Panda image, body region,
physical length, or tuned smoothing threshold.

The local support radius remains the intrinsic distance to the K+1 boundary
guide. Therefore denser guide regions retain a smaller physical radius and can
vary faster; sparse regions obtain a larger radius and vary more slowly.

## Scope

R082 is numeric and fixed-checkpoint only. It must not:

- alter the checkpoint or guide values;
- add a smoothness loss or global solve;
- change direction or another groom attribute;
- use Euclidean neighbors or cross a folded/disconnected surface;
- sample render roots/edges;
- create a new visualization;
- enter white-tiger evaluation if the Panda gate fails.

The inherited K8 field and rejected R081 K8 candidate remain immutable controls.

## Diagnostic Requirements

The fixed-checkpoint tool receives an explicit candidate active-neighbor count
without changing checkpoint/config state. Exact support-overlap counting must
scale linearly or log-linearly in support width; it may not allocate an
`edge_count x support_width x support_width` tensor.

Record:

- support radius, maximum weight, effective neighbor count;
- guide-site self-evaluation error;
- candidate-versus-legacy field difference;
- complete inherited-edge log-length statistics;
- unchanged/changed-support edge partitions and their ratio;
- overlap-count-conditioned jumps;
- support build/forward/edge time, bytes, and peak memory;
- exact hashes, finite/positive/convex/gradient/topology invariants.

## Decision Order

1. Panda numeric continuity and guide-semantics gate.
2. Panda performance/memory gate.
3. Only if both pass, matched white-tiger numeric gate.
4. Only if both samples pass, canonical existing `view09_length.png` review.
5. Only after all prior gates, consider a separate trainer/config integration.

No threshold is tuned after seeing the Panda map. A candidate that improves
within-support smoothness while worsening changed-support or global tails is
rejected, as is a candidate that obtains continuity by materially erasing guide
site values.

## Panda Fixed-Checkpoint Result

The full R080 iteration-4000 gate ran at commit
`2ba3e16f57b4d3d6d30490c9d613f564cc72449b` over `496632` render roots and
`15892224` exact inherited surface edges. It completed without sampling,
training, rendering, or checkpoint mutation.

K32 broadens the physical support and improves absolute edge statistics:

| Absolute log-length edge jump | Legacy K8 | R082 K32 |
| --- | ---: | ---: |
| all-edge mean | 0.026418 | 0.019446 |
| all-edge P95 | 0.114162 | 0.092634 |
| all-edge P99 | 0.226566 | 0.173890 |
| changed-support P95 | 0.164150 | 0.129191 |
| unchanged-support P95 | 0.023133 | 0.003891 |
| changed/unchanged P95 ratio | 7.096 | 33.199 |

The improvement is obtained by materially averaging away nodal semantics:

- effective-neighbor-count median/P95: `8.257 / 9.622`;
- guide-site relative self-error P50/P95/max:
  `3.026% / 17.163% / 72.691%`;
- render candidate-versus-legacy relative difference P50/P95/max:
  `1.861% / 10.615% / 67.789%`.

Thus K32 makes each support interior nearly flat while leaving a much larger
relative contrast at discrete support changes. More importantly, a guide root
no longer represents its own stored attribute with acceptable fidelity. This
violates the R082 decision rule even though global edge statistics improve.

The representation is computationally feasible but methodologically invalid:
candidate distance/weight evaluation is `0.142 s`, support build is `0.814 s`,
support storage is `327,777,120` bytes, and peak allocated CUDA memory is
`1.866 GB`. Performance is not the blocker.

Formal JSON:

`D:/RTS/_tmp/panda_r082_fixed_neighbor_mass_acceptance_20260901/panda_r080_iter4000_r082_k32_fixed_field.json`

SHA-256:
`e90c41beb6e3f09633c74046c0812846f8cd14e2ed150f11142020074ac2d6b0`.

## Decision

R082 is rejected before white-tiger evaluation or canonical visualization.
Increasing compact-kernel neighbor mass cannot simultaneously preserve exact
guide attributes and remove the discrete support structure. The next candidate
must have an explicit nodal interpolation property, partition of unity, and
continuous neighbor-entry behavior; surface Natural Neighbor/Sibson
coordinates are the documented reference rather than another kernel-width
sweep.

## Literature Basis

SPH reconstructs fields from normalized compact kernels and adapts smoothing
length to maintain a fixed neighbor population, coupling particle density to
physical resolution. Historical SPH implementations commonly used 32-64
neighbors; R082 uses the lower endpoint and does not sweep values.

- https://academic.oup.com/mnras/article/330/1/129/1018860
- https://academic.oup.com/mnras/article/471/2/2357/3906602
- https://doi.org/10.1007/BF02123482
