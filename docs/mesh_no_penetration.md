# Mesh No-Penetration

Status date: 2026-08-14.

Status: accepted in R062 after strict full-resolution preflight, from-zero 30k
training, all-root collision audit, fixed-view RGB comparison, and canonical
strand QA. This module is not enabled in R060 or R061; R061 remains the direct
single-variable control.

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

The canonical 1920x1080 highlight render contains all 629 penetrating strands
present in the deterministic 100k visualization subset:

```text
D:/RTS/_tmp/no_penetration/r060_allroot_local_geometry_512_20260814/r060_allroot_penetration_side_y.png
```

They concentrate around high-curvature face/muzzle regions, paws, and joint
recesses; broad torso regions remain clean. This spatial pattern is consistent
with local strand/mesh intersection and not with a global coordinate or SDF
axis error.

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

All four gates passed for R062. The full-resolution two-step H100 preflight
was `5.304 s` faster than the matched R061 preflight and added `311.45 MB` peak
allocated CUDA memory. Gradients reached guide length/direction, secondary
length/direction, and root barycentric coordinates; global translation and
scale received no collision gradient.

## Accepted R062 Result

R062 is a strict single-variable child of R061. Its formal configuration adds
only these four fields:

```text
MESH_NO_PENETRATION_SUPPORT=1
MESH_NO_PENETRATION_SDF=<reviewed artifact>
MESH_NO_PENETRATION_WEIGHT=256
MESH_NO_PENETRATION_ROOT_BATCH=16384
```

The uninterrupted from-zero H100 run completed all 30k iterations and passed
the final strict reload and postprocess checks:

| Quantity | R061 | R062 | Delta |
| --- | ---: | ---: | ---: |
| Final test composite PSNR | `32.21457` | `32.19214` | `-0.02243 dB` |
| Best test composite PSNR | `32.30076` | `32.28517` | `-0.01559 dB` |
| Fixed eight-view mean composite PSNR | `33.23565` | `33.21203` | `-0.02361 dB` |
| Final render roots | `471749` | `471583` | `-166` |
| Final generated Gaussians | `5484109` | `5475299` | `-8810` |
| H100 elapsed time | `13186.907 s` | `12512.557 s` | `-5.11%` |
| Peak allocated CUDA memory | `16312.76 MB` | `16560.75 MB` | `+247.99 MB` |

The final all-root 64-sample collision audit gives:

| Quantity | R061 | R062 | Reduction |
| --- | ---: | ---: | ---: |
| Penetrating non-root point fraction | `0.134272%` | `0.023592%` | `82.43%` |
| Roots with any penetrating sample | `0.675571%` | `0.416470%` | `38.35%` |
| Mean normalized depth over all points | `7.83501e-7` | `1.20769e-7` | `84.59%` |
| Maximum normalized depth | `0.00978804` | `0.00478041` | `51.16%` |

The fixed 100k-strand audit retains zero backward strands and zero arc lengths
above `0.12`. Local relative-length P95 changes only
`0.08033 -> 0.08110`, local direction P95 `11.526 -> 11.621 deg`, and
maximum-turn P95 `9.408 -> 9.834 deg`. Canonical side, opposite, and front/top
assets show no new loop, broad crossing cluster, tail spike, or local collapse.
The Gaussian residual decomposition is also preserved: its fixed-view gain is
`1.80923 dB`, RMS `0.08128`, and saturation `2.096%`.

Formal identities and evidence:

```text
source commit:
  100f7223ede6975862cbc6c30b27f29709f68147
checkpoint SHA256:
  d1f23c92f68b250f00ac8771f6435c63af2baf686a9329696de3b34c0cc72900
SDF SHA256:
  766e177fbeeb89fc779292f56662c7c6b256f7d4365415baa366cef04af10530
formal HGC output:
  /home/wangyy/anigroom-r062-no-penetration-runtime-20260814-v2/outputs/r062_mesh_no_penetration_0_30k_h100_20260814
local acceptance evidence:
  D:/RTS/_tmp/r062_acceptance_20260814
canonical R062 asset:
  D:/RTS/_tmp/r062_acceptance_20260814/postprocess/r062_mesh_no_penetration/assets/r062_030000_asset_side_y_v11_protocol.png
R062 penetration highlight:
  D:/RTS/_tmp/r062_acceptance_20260814/postprocess/r062_mesh_no_penetration/assets/r062_030000_penetration_highlight_side_y.png
matched R061 penetration highlight:
  D:/RTS/_tmp/r062_acceptance_20260814/reference/r061_no_penetration/r061_030000_penetration_highlight_side_y.png
```

Residual penetration is sparse and concentrated around the face, paws, joint
recesses, and a few tail samples. R062 is accepted because it materially
reduces both incidence and depth without adding body-part rules, absolute
tolerances, appearance drift, or a structural failure. Further reductions, if
needed, must remain a separate experiment rather than changing R062.
