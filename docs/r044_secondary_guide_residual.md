# R044 Parent-Conditioned Secondary Guide Residual Field

## Status

R044 is an isolated candidate built from the immutable R043 baseline. Its
from-zero full-resolution H100 forward/backward gate, strict checkpoint resume
gate, and formal 30k run all pass.

R044 is retained as a validated structural and efficiency ablation, but it is
not promoted over R043. It substantially improves local length continuity and
halves training cost, while preserving the same lifecycle population. It does
not materially improve cross-root direction continuity and loses about 0.94 dB
test composite PSNR. R043 therefore remains the active accepted baseline.

## Problem

R043 has 4500 primary guides and about 470k independent render roots after its
finite lifecycle. The primary field is appropriately smooth but too sparse to
represent controlled local geometry deviations. Giving every render root an
independent geometry residual provides enough capacity, but it also makes the
residual field expensive to smooth and allows isolated strands to depart from
their neighborhood.

R043's density-matched K32 render graph contains about 15 million directed
edges at the final population. Several per-iteration geometry losses traverse
that graph. Reducing K, using a stale graph, or smoothing generated strand
samples would change the accepted physical support or optimize the wrong
representation.

## Representation

R044 introduces a fixed secondary guide layer `G1` between primary guides
`G0` and render roots `R`:

```text
G0: direct low-frequency groom from clean-flow initialization
G1: zero-centered local geometry residuals only
R:  direct G0 base + interpolated G1 residual + render appearance
```

The base groom is never routed through `G1`. A render root samples its G0 base
exactly as in R043 and independently samples the G1 residual field. Therefore
an all-zero G1 field returns the same render geometry as direct G0
interpolation; there is no G0-to-G1-to-render double interpolation.

The G1 residual field contains the existing zero-centered geometry
coordinates:

- positive relative length;
- root width, tip/root ratio, and width taper;
- normalized local 3D direction perturbation;
- optional curl and frizz coordinates;
- child spread and clump strength.

Brush stiffness remains guide-owned. Curl and frizz retain zero effective
scale in this short-fur configuration. R044 does not add an animal-region rule,
length endpoint, split budget, or density schedule.

## Secondary Placement

The formal population is 4500 G0 roots and 20k G1 roots. Placement is balanced
and parent-conditioned rather than a second unrelated global FPS:

1. Retain every G0 point as an exact G1 anchor.
2. Draw a dense area-uniform mesh candidate pool.
3. Assign every candidate to its nearest topology-valid G0 surface cell.
4. Run seeded local FPS inside each cell with the G0 point as the fixed seed.
5. Give every G0 cell the same base child count.
6. Assign the indivisible remainder to cells with the largest uncovered local
   radius, with deterministic tie handling.

For 4500 to 20k roots, every G0 owns exactly four or five G1 roots. This keeps
the secondary capacity uniform with respect to the primary control field while
allowing the actual points to spread over each local surface cell.

## Surface Support

Each render root first obtains its accepted G0 surface support. Its G1
candidates are then restricted to children of those G0 roots. The nearest G1
support is selected only inside that topology-valid union. This prevents an
ambient-space neighbor from crossing folded or nearby disconnected surface
regions.

The formal configuration uses:

```text
G0 -> render support: K=8
G1 -> render residual support: K=8
G1 smoothing graph: K=32
render appearance graph: K=32
```

G1 local 3D direction residuals are parallel-transported into the query root's
surface frame before interpolation. They are vectors rather than axes, so the
transport path explicitly preserves residual magnitude.

## Optimization And Lifecycle

The accepted G0 base schedule is unchanged. The historical render geometry
residual parameters are replaced by the 20k-row G1 residual parameters and use
the same gradual geometry unlock. Optimizer parameter names and state follow
the active G1 field; no unused render geometry residual remains in the model.

Geometry residual smoothness, effective-groom smoothness, and clean-flow 3D
consistency operate on the fixed 20k G1 graph. The dense render graph retains
appearance smoothing for root/tip color and opacity. Sample-level strand
smoothing remains removed.

Render densification changes R only. It rebuilds G1-to-render support while G1
parameter rows and optimizer state remain fixed. Primary-guide densification
is disabled for this candidate and fails fast if requested, because a changing
G0 topology requires a separately designed G1 lifecycle rather than silent
reparenting.

## Isolated Configuration Change

Compared with `configs/r043_density_matched_render_support_0_30k.env`, the only
additional resolved assignments are:

```text
GEOMETRY_RESIDUAL_DOMAIN=secondary_guide
SECONDARY_GUIDE_ROOT_COUNT=20000
SECONDARY_GUIDE_CANDIDATE_MULTIPLIER=16
SECONDARY_GUIDE_INTERPOLATION_K=8
SECONDARY_GUIDE_SMOOTH_K=32
```

All R043 data, renderer, losses, schedules, lifecycle thresholds, resolution,
and population settings remain unchanged.

## Verification

The complete local suite passes (`98 passed`). Focused coverage includes:

- balanced deterministic local FPS;
- topology-restricted G1 support;
- zero-residual direct-G0 geometry equivalence;
- differentiable interpolation into G1 residual parameters;
- 3D residual magnitude preservation under normal rotation;
- optimizer inclusion of G1 and exclusion of render geometry residuals;
- strict model-state roundtrip;
- render lifecycle support rebuild with unchanged G1 rows.

Static checks pass for Python compilation, shell syntax, diff whitespace, and
the new module/test Ruff scope. R043's configuration and behavior source are
unchanged.

## Full-Resolution H100 Gate

The from-zero gate used the real white-tiger data, official mesh, gsplat
renderer, mesh-depth clipping, complete losses, 400k render roots, 4500 G0
roots, 20k G1 roots, and 1920x1080 images. Only the target iteration was set to
one; no model, renderer, resolution, root count, or loss was reduced.

```text
commit:       2a76723e54b2571d2f8f307400e1bb961fcb81a5
host/GPU:     pcg02 / one H100 80GB
start/end:    2026-08-07 18:11:46 / 18:17:02 +09:00
exit code:    0
G0/G1/R:      4500 / 20000 / 400000
G1 per G0:    4 or 5
support fallbacks: 0
G1 graph:     640000 directed edges
render graph: 12800000 directed edges
iteration 1:  30.93 s including traced full iteration work
peak allocated CUDA: 7673.50 MB
nvidia-smi process: 8810 MB
```

The candidate pool contained 320k points. The smallest/largest G0 cells held
17/140 candidates. Every render query had at least 32 topology-valid G1
candidates before selecting K8.

Artifacts:

```text
source: /home/wangyy/anigroom-r044-secondary-guide-20260807
gate:   /home/wangyy/anigroom-r044-secondary-guide-runtime-20260807/outputs/r044_secondary_guide_formal_gate_h100_20260807
log:    /home/wangyy/logs/anigroom_r044_secondary_guide_formal_gate_h100.log
```

## Strict Resume Gate

`checkpoint_000001.pt` was resumed with optimizer and RNG state enabled and
advanced through a second full forward/backward iteration.

```text
checkpoint SHA-256: 98fd6f93ffb617b9d31d300b1a51bc1b414c264195a8ad6278a3475ac61df7ac
optimizer state entries restored: 21
start/target iteration: 1 / 2
exit code: 0
iteration 2: 32.61 s
peak allocated CUDA: 7774.36 MB
```

The resumed checkpoint restored the exact 20k G1 face IDs, barycentric
coordinates, parent IDs, and nine residual tensors. It rebuilt the same
640k-edge G1 graph and 12.8M-edge render graph without topology migration.

Resume artifacts:

```text
/home/wangyy/anigroom-r044-secondary-guide-runtime-20260807/outputs/r044_secondary_guide_resume_gate_h100_20260807
```

## Formal 30k Run

The formal run started from zero on one H100 at full 1920x1080 resolution. It
used the same data, losses, schedules, renderer, mesh clipping, random mesh
backing, initial render population, and render lifecycle as R043.

```text
host/GPU:        pcg02 / one H100 80GB
start/end:       2026-08-07 18:56:08 / 21:31:10 +09:00
exit code:       0
elapsed:         9133.443 s (2.54 h)
final G0/G1/R:   4500 / 20000 / 469402
final Gaussians: 5319491
peak allocated:  10699.64 MB
config SHA-256:  53eb815b63ca26cbd63777c099fc0ec6cb4c63d0a4c1f8618c738ab944ead494
checkpoint SHA:  d9f2e55091c72548f973c7030cc6f1269121caeec853d4f45078a5c76d62125f
```

The finite render lifecycle completed at iteration 9000. It split 554 parents
into 1108 children, removed the 554 parents, and rebuilt render-to-G1 support.
G1 remained exactly 20k rows with the same row identity and 640k directed
smoothing edges throughout training.

Artifacts:

```text
output:     /home/wangyy/anigroom-r044-secondary-guide-runtime-20260807/outputs/r044_secondary_guide_full_0_30k_h100_20260807
checkpoint: /home/wangyy/anigroom-r044-secondary-guide-runtime-20260807/outputs/r044_secondary_guide_full_0_30k_h100_20260807/checkpoint_030000.pt
local QA:   D:/RTS/_tmp/r044_30k_final
```

## Reconstruction Result

R044 and R043 are nearly identical before geometry residuals unlock. At 10k,
their test composite PSNR differs by only 0.033 dB. The gap appears after the
residual stage begins, grows through 20k, and remains near 1 dB:

| Iteration | R043 test | R044 test | R044 - R043 |
| ---: | ---: | ---: | ---: |
| 10000 | 29.871 | 29.839 | -0.033 |
| 12000 | 30.787 | 30.579 | -0.208 |
| 14000 | 31.372 | 30.848 | -0.524 |
| 16000 | 31.739 | 31.037 | -0.703 |
| 20000 | 32.370 | 31.392 | -0.978 |
| 30000 | 32.512 | 31.575 | -0.937 |

Final aggregate results:

| Metric | R043 | R044 | Change |
| --- | ---: | ---: | ---: |
| Train composite PSNR | 33.466 | 32.234 | -1.232 dB |
| Test composite PSNR | 32.512 | 31.575 | -0.937 dB |
| RGB L1 | 0.018278 | 0.020621 | +0.002343 |
| Final render roots | 469620 | 469402 | -218 |
| Peak allocated CUDA | 19733 MB | 10700 MB | -45.8% |
| Wall time | 17389 s | 9133 s | -47.5% |

The same eight full-resolution fixed views give 33.472 dB mean for R043 and
32.327 dB for R044, a -1.145 dB change. Every inspected view is lower by
0.90--1.59 dB, so the loss is broad high-frequency capacity rather than one
bad camera, one body region, or an evaluation mismatch.

## Structural Audit

The canonical postprocess is identical for both runs: 100k exported strands,
32 centerline samples, one child, uniform material, fixed mesh/camera/lighting,
1920x1080 output, and no RGB texture. The R044 asset does not introduce a new
curl, backward-strand, long-tail, or lifecycle artifact class.

Within individual strands, R044 is measurably less extreme:

| Centerline statistic | R043 | R044 |
| --- | ---: | ---: |
| Backward segments | 0 | 0 |
| Strands with chord length > 0.12 | 6 | 0 |
| Maximum chord length | 0.13155 | 0.11892 |
| Arc/chord P95 | 1.00589 | 1.00306 |
| Maximum local turn P95 | 0.86996 deg | 0.60208 deg |
| Maximum observed local turn | 2.73359 deg | 2.02703 deg |

An independent cross-root KNN audit on the exported roots separates length
continuity from direction continuity. With eight spatial neighbors:

| Local-field statistic | R043 | R044 |
| --- | ---: | ---: |
| Nearest-root relative length difference P50 | 0.03164 | 0.00278 |
| Nearest-root relative length difference P95 | 0.15710 | 0.02936 |
| Mean-8 relative length difference P50 | 0.04796 | 0.00934 |
| Mean-8 relative length difference P95 | 0.11900 | 0.03809 |
| Nearest-root direction angle P50 | 1.940 deg | 1.914 deg |
| Mean-8 direction angle P50 | 3.768 deg | 3.782 deg |
| Max-8 direction angle P99 | 36.31 deg | 35.91 deg |
| Roots with max-8 angle > 30 deg | 1.944% | 1.908% |

The G1 representation therefore succeeds at its clearest structural objective:
local length discontinuity falls by roughly 4--11x across the central and P95
statistics. Direction discontinuity is almost unchanged. Visual inspection of
the side, front, tail/hind, belly/flank, and neck/head renders agrees with this
split: length is more uniform, while the remaining crossing regions largely
coincide with R043.

## Diagnosis

This is not an initialization, lifecycle, renderer, or checkpoint-resume
failure:

- R043 and R044 match through 10k, before geometry residual capacity matters.
- Final render populations and effective length distributions are nearly the
  same.
- G1 rows and optimizer state survive the full render lifecycle unchanged.
- The fixed-view loss is distributed across views and body regions.

The unresolved issue is the scale of the geometry operator. R043 evaluates
geometry regularizers on about 470k render roots; R044 evaluates them on 20k
G1 roots but reuses `K=32` and the same scalar loss weights. Under comparable
surface sampling, the physical spacing changes approximately with the square
root of density, so a 20k K32 neighborhood spans a much larger surface region
than a 470k K32 neighborhood. The current edge losses do not normalize for
edge length or source area. Their raw values are therefore not comparable
across the two domains and the G1 field is over-low-passed.

The training curve supports this diagnosis. At 30k, G1 residual direction P95
is 0.0476 versus 0.2423 for R043, and residual length P95 is 0.0251 versus
0.2079. R044 preserves a clean low-frequency field but suppresses useful local
geometry evidence. Since the G0 clean-flow direction remains dominant, this
strong residual bottleneck improves length continuity without materially
changing direction crossings.

## Disposition

R044 is complete and reproducible, but it does not replace R043 as the active
baseline. It establishes three useful results:

1. A fixed parent-conditioned G1 residual layer is technically correct and
   survives render-root densification without reparenting or state corruption.
2. Moving geometry regularization from roughly 470k render roots to 20k G1
   roots cuts wall time by 47.5% and peak allocated CUDA memory by 45.8%.
3. The representation strongly improves local length continuity, but fixed-K,
   unnormalized smoothing at the new density removes useful detail and does not
   solve the remaining direction-field crossings.

The next candidate must change only the G1 geometry operator: its neighborhood
and quadrature must represent a consistent physical surface scale instead of a
fixed neighbor count. It must not add an animal-region rule, absolute groom
threshold, split budget, or per-sample schedule. R043 remains immutable while
that candidate is evaluated.
