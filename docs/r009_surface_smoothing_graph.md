# R009 Surface-Consistent Smoothing Graph

## Purpose

R003 made guide-to-render interpolation surface-aware, but the explicit guide,
render, effective-groom, and strand-shape smoothness terms still use Euclidean
KNN. On folded or nearby disconnected mesh surfaces, Euclidean KNN can connect
roots that are close in XYZ but unrelated on the animal surface. It also makes
the interpolation and regularization operators disagree.

R009 keeps the R008 loss weights, edge mask, lifecycle, schedules, and
neighborhood size unchanged. Its only conceptual change is the neighborhood
provider used by every existing root-level smoothness term.

## Method

Guide-root edges reuse the intrinsic source-neighbor graph already built by
`SurfaceFieldInterpolator`. Render-root edges are built hierarchically: roots
may connect only inside overlapping guide support, then the nearest K roots are
selected in that topology-local candidate set. The graph is rebuilt after any
guide or render-root lifecycle event.

```text
SMOOTH_GRAPH_MODE=surface_hierarchical
SMOOTH_GRAPH_K=8
```

No smoothness weight is increased. This avoids restoring the accidental
low-pass behavior of the old Euclidean graph by brute force.

## Required Verification

1. A folded, disconnected two-sheet mesh must expose Euclidean cross-sheet
   edges and zero cross-sheet R009 edges.
2. Guide and render roots must each have exactly K directed neighbors, with no
   self edges.
3. The same guarantees must hold after a synthetic render-root insertion.
4. R008 and R009 must be compared from the same 9k checkpoint through the same
   14k endpoint before a full 0-30k run is accepted.
5. Acceptance requires canonical pure-fur views and local 3D direction/length
   discontinuity metrics, not PSNR alone.

The matched calibration config is
`configs/r009_surface_smoothing_probe_9k_14k.env`.

## Status

Implementation, topology regression, matched H100 calibration, and canonical
visualization passed on 2026-07-23.

The matched 9k-14k comparison used the same R007d 9k checkpoint, reset the
optimizer in both runs, and changed only the graph provider:

| Run | Train composite | Test composite | Roots | Gaussians | Peak allocated |
| --- | ---: | ---: | ---: | ---: | ---: |
| R008 | 31.5792 | 31.0097 | 200,967 | 8,866,668 | 15,642.6 MB |
| R009 | 31.5670 | 31.0020 | 201,051 | 8,865,882 | 15,648.5 MB |

The test delta is `-0.0077 dB`, so the operator change is metric-neutral and
does not alter effective capacity. Canonical RGB predictions are nearly
identical. On one strand per render root, R009 slightly reduces local 3D
discontinuity: direction chord mean/P95 changes from `4.6237/13.1382` to
`4.6146/13.1033`, and relative length jump mean/P95 changes from
`0.08087/0.28668` to `0.08023/0.28571`. This is a small structural improvement,
not a claimed visual breakthrough.

The first exact implementation exposed a runtime problem: Python set/loop
construction took about 20-24 seconds for each 201k-root render-graph rebuild.
The optimized implementation uses the theorem that the top K elements of a
union are contained in the union of each subset's top K. It evaluates K nearest
roots inside each topology-valid primary-guide bucket, then merges those
candidates per root. Support aggregation is vectorized over face-vertex source
paths. On the real 14k checkpoint, both optimized outputs match the reference
exactly: zero support-ID differences, zero support-path differences, zero
render-edge-set differences, and zero ordered-edge differences over 201,051
roots.

Held-H100 timing after optimization:

| Operation | Time |
| --- | ---: |
| Guide interpolator rebuild | 10.1 s |
| 201k guide-to-render support | 4.17 s |
| 1.61M render smoothing edges | 4.45 s |

The interpolator rebuild occurs only after guide densification; render-only
lifecycle events require only the last operation.

## Full 0-30k Result

The complete from-zero run finished on the held H100 without restart, OOM, or
fallback:

```text
/home/wangyy/anigroom-r009-surface-smooth-20260723/outputs/
  r009_surface_graph_full_0_30k_h100_20260723
```

| Iteration | Train composite | Test composite | Render roots | Gaussians |
| ---: | ---: | ---: | ---: | ---: |
| 9000 | 24.0313 | 24.2516 | 207072 | 8282880 |
| 10000 | 29.1738 | 29.0954 | 218558 | 9342609 |
| 14000 | 31.5762 | 31.0090 | 219770 | 9659209 |
| 16000 | 32.0827 | 31.4685 | 219770 | 9678997 |
| 20000 | 32.7616 | 32.0334 | 219770 | 9695290 |
| 24000 | 32.9691 | 32.2314 | 219770 | 9711020 |
| 27000 | 33.0742 | 32.3396 | 219770 | 9718202 |
| 29000 | 33.1273 | **32.4038** | 219770 | 9723866 |
| 30000 | **33.1763** | 32.2826 | 219770 | 9725837 |

Peak allocated CUDA memory was `17167.23 MB`, below the 30 GB guard. Render
densification ran every 100 iterations from 600 through 10000. The last event
replaced 1212 selected parents by two children each, taking the render-root
count from 218558 to 219770; it then stayed fixed. Guide densification ran from
11000 through 16000 and increased guide roots from 4500 to 5332. The inherited
R008 guide event cap of 32 remains present and is not a contribution of R009.

Full-resolution evaluation over views 0, 5, 9, 14, 18, 21, 27, and 32 has a
mean composite PSNR of `33.1473 dB` (range `32.4003-34.3498 dB`). Visual errors
are concentrated at the silhouette, mouth, and high-frequency stripe detail;
the historical whole-region blur and obvious curled-back strands do not recur.

## Structural QA

Canonical Blender exports use the shared visualizer, the same 100k-strand
sample count, the aligned furless mesh, and both parent-only and configured
child-strand views. The parent and child views remain coherent. Mild local
convergence remains around the shoulder/back transition, belly, rear-leg root,
and tail root, but there are no large circular fields or isolated structure
collapses.

The same 100000 parent root IDs were exported at 27k and 30k. Their 27k-to-30k
change is small:

| Measure | Mean | P95 |
| --- | ---: | ---: |
| Root displacement | 0.000155 | 0.000456 |
| Tip displacement | 0.000526 | 0.001312 |
| Chord-direction change | 0.649 deg | 1.493 deg |

Mean strand length changes from `0.022770` to `0.022868` (`+0.43%`). Arc/chord
tortuosity remains `1.00000016`, and no sampled parent strand exceeds 1.05.
Therefore the small 29k-to-30k test-PSNR decrease is ordinary late overfit, not
a late geometry failure. The retained 27k checkpoint is the preferred visual
checkpoint; 30k remains the reproducible terminal checkpoint.

## Decision

R009 is accepted as the surface-consistent structural baseline. The graph fix
is mathematically justified, removes demonstrated cross-surface neighbors,
matches R008 at 14k within `0.0077 dB`, is exact after acceleration, and remains
stable through 30k.

This does not establish the inherited R008 edge-mask policy as the best
reconstruction setting. R009 final/best test composite (`32.2826/32.4038`) is
about `0.326/0.351 dB` below R007d (`32.609/32.755`), consistent with the
approximately 0.3 dB cost already isolated for R008's 2-pixel edge band. The
surface graph stays locked; any later recovery of that metric must vary the
edge-loss policy separately rather than reverting topology-aware smoothing.

Local artifacts:

```text
D:/petsgaussianhair-accept-line/_downloads/r009_surface_smoothing_20260723/full_30k
```
