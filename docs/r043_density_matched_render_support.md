# R043 Density-Matched Render Support

## Status

The full-resolution from-zero H100 gate passed. A formal from-zero 30k run is
active on the same verified source and configuration.

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
