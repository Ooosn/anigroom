# R043 Density-Matched Render Support

## Status

Accepted as the active structural/lifecycle Stage 1 baseline. The
full-resolution from-zero H100 gate and formal 30k run both completed without
restart, fallback, hidden capacity limit, or memory-guard event. R042 remains
the frozen K8 parent and R036 remains the frozen higher-PSNR metric control.

## Motivation

R040 replaced four deterministic child strands per render root with four times
as many independent render roots:

- R039: 100k render roots x 4 children
- R040-R042: 400k render roots x 1 strand

The render-root graph retained `K=8`. On an approximately two-dimensional
surface, quadrupling point density while keeping K fixed approximately halves
the physical neighborhood radius. The field losses therefore became much more
local even though their weights did not change. R042's sparse tail-tip length
outliers are consistent with this lost physical support.

## Single Conceptual Change

Expand render-domain surface support from `K=8` to `K=32` while preserving all
R042 behavior otherwise. The same render support is used by:

1. render-root parameter-field smoothing;
2. render lifecycle child-attribute inheritance;
3. projected-color inpainting for unobserved render roots.

Guide-domain operations remain at `GUIDE_INTERPOLATION_K=8` because the guide
population did not increase. Clean-flow processing, guide interpolation,
guide smoothing, guide lifecycle support, densification evidence, parent
selection, and child placement are unchanged.

Sample-level strand smoothing remains removed. R043 smooths compact 3D groom
parameter fields, not generated strand samples.

## Configuration Contract

Relative to `configs/r040_child1_dense_render_0_30k.env`, the only resolved
assignment change is:

```text
SMOOTH_GRAPH_K: 8 -> 32
```

The relevant population contract remains:

```text
ROOT_COUNT=400000
CHILD_COUNT=1
GUIDE_ROOT_COUNT=4500
GUIDE_INTERPOLATION_K=8
```

## H100 Gate

Run from zero at full 1920x1080 resolution through iteration 700 so the changed
render child-inheritance path executes at the 600 and 700 lifecycle events.
No fallback, reduced resolution, resumed checkpoint, hidden split cap, or
sample-level strand loss is allowed.

Record:

- render and guide graph K from setup reports;
- graph setup and lifecycle rebuild time;
- parent selection, inserted/deleted roots, and final root count;
- iteration throughput and peak allocated CUDA memory;
- train/test composite PSNR.

The gate passes only if K32 is practical on one H100 and lifecycle semantics
remain evidence-driven and uncapped. If it passes, launch a formal from-zero
30k run without changing the configuration.

## Final Structural Audit

Compare R043 against frozen R042 using the canonical visualization module and
the same 100k-strand export protocol. In addition to RGB and composite PSNR,
report effective-length P50/P95/max and counts above `0.12`, `0.15`, and `0.20`.
Inspect the tail tip, head fringe, and ordinary coat for transferred artifacts.
These thresholds are diagnostics only; they do not enter training or loss.

## Gate Result

The gate ran on one H100 from iteration 0 through 700 at 1920x1080. It exited
normally with no fallback or memory guard event.

| Measure | R040 K8 gate | R043 K32 gate |
| --- | ---: | ---: |
| Test composite PSNR at 700 | 20.779205 | 20.780363 |
| Render roots after iteration 700 | 402252 | 402253 |
| Parents selected at 600 / 700 | 901 / 1351 | 903 / 1350 |
| Render graph edges after 700 | 3,218,016 | 12,872,096 |
| Render graph rebuild after 700 | 18.5848 s | 0.2163 s |
| Peak allocated CUDA memory | 9.83 GB | 16.15 GB |

The changed smoothing support causes only the expected small optimization
difference: lifecycle evidence remains uncapped and selects effectively the
same number of parents. The render graph reports `K=32`; the guide graph
reports `K=8`. K32 increases live memory but remains below the 25 GB guard, and
the accelerated exact graph path keeps lifecycle rebuild practical.

Formal run paths:

```text
source: /home/wangyy/anigroom-r043-density-support-20260807
gate:   /home/wangyy/anigroom-r043-density-support-runtime-20260807/outputs/r043_density_support_from_zero_700_h100_20260807
full:   /home/wangyy/anigroom-r043-density-support-runtime-20260807/outputs/r043_density_support_full_0_30k_h100_20260807
commit: 3afc5078026dd335eccc284ee04136e42dd70a41
```

## Formal 30k Result

The formal run used one H100, the exact gate source, full 1920x1080 training,
400k initial independent render roots, one strand per root, and the complete
0-30k schedule.

| Measure | R042 K8 | R043 K32 | R043 - R042 |
| --- | ---: | ---: | ---: |
| Final train composite PSNR | 33.482647 | 33.465813 | -0.016834 dB |
| Final test composite PSNR | 32.515427 | 32.511589 | -0.003838 dB |
| Best test composite PSNR | 32.719177 | 32.714214 | -0.004963 dB |
| Best test iteration | 29000 | 29000 | 0 |
| Final render roots | 469737 | 469620 | -117 |
| Training-metric Gaussians | 5319498 | 5295653 | -23845 |
| Peak allocated CUDA memory | 13187.99 MB | 19733.46 MB | +6545.47 MB |
| H100 training time | 11263.974 s | 17388.655 s | +54.37% |

All 85 lifecycle events completed by iteration 9000. Root count remained fixed
after the lifecycle ended. Checkpoint reload, eight full-resolution RGB views,
the canonical three-view 100k-strand asset export, and the numeric centerline
audit all completed successfully.

The eight fixed postprocess views remain reconstruction-equivalent. Mean
composite PSNR changes from `33.484314` to `33.471676` (`-0.012638 dB`):

| View | R042 composite | R043 composite | Delta |
| ---: | ---: | ---: | ---: |
| 0 | 33.059895 | 33.049404 | -0.010490 |
| 5 | 32.870377 | 32.802582 | -0.067795 |
| 9 | 32.549942 | 32.519249 | -0.030693 |
| 14 | 34.077782 | 34.038532 | -0.039249 |
| 18 | 33.979019 | 33.993229 | +0.014210 |
| 21 | 34.804871 | 34.887009 | +0.082138 |
| 27 | 32.719738 | 32.698261 | -0.021477 |
| 32 | 33.812885 | 33.785141 | -0.027744 |

## Structural Audit

The same 100k render roots and 32 centerline samples were audited for both
runs. Diagnostic length thresholds below are not used by training.

| Centerline statistic | R042 K8 | R043 K32 |
| --- | ---: | ---: |
| Backward segment fraction | 0 | 0 |
| Length P50 | 0.023636 | 0.023380 |
| Length P95 | 0.043938 | 0.043203 |
| Length P99 | 0.058326 | 0.055576 |
| Length P99.9 | 0.097483 | 0.080785 |
| Maximum length | 0.220872 | 0.131546 |
| Count above 0.12 | 39 | 6 |
| Count above 0.15 | 7 | 0 |
| Count above 0.20 | 1 | 0 |
| Arc/chord P99 | 1.041043 | 1.024660 |
| Maximum local turn P99 | 5.2490 deg | 2.0683 deg |
| Maximum local turn | 14.8915 deg | 2.7336 deg |

The median and P95 coat lengths remain nearly unchanged. The improvement is
concentrated in the sparse residual tail: R043 removes isolated long and sharp
render-root deviations without an absolute length cap, percentile loss, body
region rule, or reduced root capacity.

Full-resolution visual inspection agrees with the numeric audit. The tail tip,
head fringe, flank, belly, and legs keep their distinct trends; ordinary coat
flow is more continuous; and no stripe blur or transferred over-smoothing is
visible in the eight RGB views. Canonical local artifacts are under:

```text
D:/RTS/_tmp/r043_30k_final
D:/RTS/_tmp/r043_30k_final/r043_030000_asset_side_y_v11_protocol.png
D:/RTS/_tmp/r043_30k_final/r043_030000_asset_side_y_pos_v11_protocol.png
D:/RTS/_tmp/r043_30k_final/r043_030000_asset_front_z_v11_protocol.png
```

## Decision And Performance Boundary

R043 replaces R042 as the active structural/lifecycle baseline. The decision
is structural rather than metric-driven: final and best test composite differ
from R042 by less than `0.005 dB`, while the isolated centerline tail is
substantially smaller.

KNN construction is not the remaining speed problem. The exact K32 graph is
cached and rebuilds only after topology changes in approximately `0.2-0.6 s`.
The 54.37% full-run time increase comes from repeatedly traversing about 15
million render edges in several per-iteration smoothness terms. The next
candidate must be an exact execution optimization, not a weaker K, reduced
resolution, or approximate/stale graph:

1. skip smoothness terms while their fields are frozen or their weights are
   exactly zero;
2. reuse effective 3D direction and parallel-transport edge differences
   across compatible losses;
3. pack scalar groom fields so one gathered edge pass serves multiple
   graph-Laplacian terms.

That optimization is reserved for R044 and must pass fixed-checkpoint
loss/gradient equivalence before a from-zero gate. It is intentionally not
mixed into R043 acceptance.
