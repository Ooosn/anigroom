# Mesh No-Penetration

Status date: 2026-08-14.

Status: implementation and isolated verification complete; formal Stage1
training acceptance is pending. This module is not enabled in R060 or R061.

## Purpose

The loss prevents generated strand samples from entering the furless body
mesh. It is a geometric validity term, not an animal-part rule and not a
length prior. It must therefore depend only on the mesh signed-distance field
and the generated continuous strand geometry.

For a mesh-local strand sample `x`, let `SDF(x)` be positive outside and
negative inside. The normalized penetration depth is

```text
p(x) = ReLU(-SDF(x)) / diagonal(SDF bounds)
```

and the sampled loss is the mean of `p(x)` over every selected non-root strand
sample. Root samples are excluded because roots lie on the body surface by
construction.

The denominator makes the loss invariant to the shared isotropic mesh scale.
There is no body-region mask, absolute penetration distance threshold, strand
length threshold, or tolerated inside band.

## Relation To Neural Fur

Neural Fur precomputes a regular body SDF, samples it with trilinear
`grid_sample`, and minimizes `mean(ReLU(-SDF))` over non-root strand points.
The released panda recipe uses `lambda_sdf=1`.

This module adopts the same geometric principle but deliberately changes four
implementation details:

1. The white-tiger SDF uses longest-axis resolution 512 rather than the
   released low-resolution helper defaults.
2. The SDF archive is tied to the exact source mesh SHA256 and records its
   sign convention and axis order.
3. Queries outside the SDF volume receive positive distance to the volume;
   they are not silently clamped to a boundary SDF value.
4. Penetration depth is dimensionless. The training weight must be selected
   from gradient-scale evidence on this model, not copied from Neural Fur.

## SDF Construction Contract

The source mesh is:

```text
data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj
```

Its original topology has 80,007 vertices, 159,997 faces, consistent winding,
and two small boundary loops of 11 and 6 edges. SDF construction fails on an
open mesh unless explicit boundary closure is requested. The accepted builder
closes those inspected loops with oriented centroid fans and then requires the
result to be watertight.

Open3D `RaycastingScene` is the only signed-distance backend. There is no
fallback backend. The previous trimesh ray implementation was removed after
it attempted an 18.4 GiB ray allocation on this mesh.

The current 512-grid artifact is:

```text
D:/RTS/_tmp/no_penetration/white_tiger_sdf_long512.npz
```

Validation results:

| Quantity | Result |
| --- | ---: |
| Grid shape `[z,y,x]` | `[520,260,106]` |
| Voxel size | `0.0034882643` mesh units |
| Interpolated/exact sign agreement | `1.0` |
| Expected normal-offset sign agreement | `0.9999593` |
| P95 absolute interpolation error | `0.025317` voxels |
| Resident CUDA size | `54.7 MB` |

## Training Integration

The SDF is loaded once and remains a non-trainable CUDA buffer. At iteration
`t`, a deterministic cyclic block of roots is selected. Their strand geometry
is generated directly in mesh-local coordinates and every non-root sample is
queried. Length is converted with a detached copy of the shared isotropic
scale; global translation and scale are therefore calibration, not collision
escape variables. Root barycentric coordinates and all groom geometry remain
differentiable.

The default proposed batch is 16,384 roots. With roughly 400k roots this covers
the complete population in about 25 iterations without a random sampler or a
region-specific policy. Densification may change root count, so indices are
computed from the current population every iteration.

Local RTX 4080 measurements for 63 non-root samples per strand:

| Roots per step | Query time | Additional CUDA memory |
| ---: | ---: | ---: |
| 4,096 | `1.21 ms` | `21.48 MB` |
| 8,192 | `1.48 ms` | `43.27 MB` |
| 16,384 | `2.11 ms` | `86.16 MB` |

The collision term is part of the structure-owned regularization path. It is
not detached by RGB/flow gradient separation and it is not evaluated during
full-view metrics.

## Frozen R060 Evidence

A deterministic 100k-strand export from frozen R060 was evaluated with 32
samples per strand:

| Quantity | Result |
| --- | ---: |
| Penetrating non-root point fraction | `0.0877%` |
| Strands with any penetrating sample | `0.284%` (`284/100000`) |
| Mean normalized depth over all points | `6.04e-7` |
| Maximum normalized depth | `0.00940` |
| Positive-root P95 maximum depth | `0.00259` |

The canonical diagnostic is:

```text
D:/RTS/_tmp/no_penetration/r060_100k_penetration_side_y.png
```

The formal all-root audit used the exact frozen R060 checkpoint, all 473,867
render roots, and 64 samples per strand:

| Quantity | Result |
| --- | ---: |
| Evaluated non-root points | `29,853,621` |
| Penetrating non-root points | `29,043` (`0.097285%`) |
| Strands with any penetrating sample | `2,890` (`0.609876%`) |
| Mean normalized depth over all points | `5.41044e-7` |
| Maximum normalized depth | `0.00940175` |
| Positive-root P95 maximum depth | `0.00189936` |

The complete machine-readable report is:

```text
D:/RTS/_tmp/no_penetration/r060_allroot_local_geometry_512_20260814/report.json
```

A real backward pass over 16,384 roots confirms nonzero gradients on guide
length/direction/brush stiffness, secondary length/direction, curl/frizz, and
root barycentric coordinates. It confirms no gradient on global translation
or global scale. Comparing L2 gradient norms against the complete weighted
R060 structure loss gives equal-norm collision weights of approximately 316
for guide length, 265 for guide direction, 786 for secondary length, and 678
for secondary direction. A single candidate weight of `256` is therefore
proposed for the formal preflight: it is comparable to the primary guide
geometry terms and remains weaker than the secondary structure terms.

These measurements prove that penetration is sparse but real and calibrate one
global candidate weight. They do not constitute a training acceptance result.

## Formal Acceptance Gate

Before enabling this module in the accepted Stage1 route:

1. Pass one native 1920x1080 forward/backward preflight with candidate weight
   `256`, the exact formal renderer, and the selected frozen base; report added
   time and memory.
2. Verify again that collision gradients reach groom geometry and root
   barycentric coordinates but not global translation or scale.
3. Train a single-variable from-zero 30k comparison against the selected
   frozen base.
4. Compare composite metrics, canonical RGB views, canonical pure-strand
   assets, penetration statistics, crossings, foldbacks, curl/frizz structure,
   lifecycle behavior, and runtime.

Failure must remain visible. A missing/mismatched SDF, wrong mesh SHA, wrong
sign convention, disabled support with a nonzero weight, or malformed depth
shape raises an error; none of these conditions has a fallback.
