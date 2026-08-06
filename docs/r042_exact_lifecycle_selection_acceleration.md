# R042 Exact Lifecycle Selection Acceleration

## Status

R042 is the execution-only child of R041. It preserves the R040/R041 model,
losses, schedule, lifecycle evidence, thresholds, selected-parent rule,
candidate set, child placement, pruning, interpolation, graph K, and graph
ordering. The matched 600-to-700 H100 gate passes. R042 removes the remaining
obvious duplicate work in lifecycle selection; it does not add a new method.

The next gate is one formal from-zero 30k run of the unchanged R040 structural
candidate (`child_count=1`, 400k initial independent render roots).

## Profiled Bottleneck

R041 had already reduced the exact 400k-root graph rebuild from `18.5848 s` to
`0.1988 s`. Profiling the complete lifecycle event on the same H100 checkpoint
showed that selection had become the dominant cost:

- lifecycle selection: `8.991121 s`;
- parent selection: `1.474237 s`;
- child placement: `7.516030 s`;
- candidate generation: `1.107349 s`;
- candidate KNN: `6.407642 s`;
- candidate count: `561600`.

The profile run selected `1350` parents, generated `2700` children, and changed
the population from `400901` to `402251` roots. Its train/test composite PSNR
was `20.591549 / 20.779280`, with `9787.37 MB` peak allocated CUDA memory.

## Exact Execution Changes

R042 makes only two measured, semantics-preserving changes:

1. The minimum candidate-to-root distance and the K-th candidate-to-root
   distance now come from the same exact `topk` result. The old path evaluated
   a second full candidate-by-root distance matrix solely to recover the
   nearest distance. K, distance threshold, candidates, selected children,
   and ordering are unchanged.
2. Mesh face adjacency is built once as an immutable `FaceAdjacencyIndex` and
   reused by parent local-maximum selection and child candidate generation.
   The topology is fixed for the run, so rebuilding the identical adjacency at
   every lifecycle event had no semantic effect.

There is no approximate KNN, reduced K, stale graph, selected-root budget,
fallback path, or changed lifecycle threshold.

## Verification

The focused suite adds exact comparisons for:

- cached versus uncached face-adjacency structure updates;
- fused nearest/K-th distances versus the independent two-pass reference;
- selected parents, child positions, inherited attributes, delete masks, and
  lifecycle metadata.

Result: `39 passed` in the focused R040-R042 lifecycle/graph suite.

## Matched H100 Gate

The profile and optimized runs use the same R040 `checkpoint_000600.pt`, H100,
configuration, data, RNG continuation, and iteration-700 lifecycle event.

R042 profile, commit `22936dd32996114579ee7e1863e22719b04d3ae4`:

- roots before/after: `400901 -> 402251`;
- selected parents / children: `1350 / 2700`;
- selection: `8.991121 s`;
- candidate KNN: `6.407642 s`;
- test composite PSNR: `20.779280`;
- peak allocated memory: `9787.37 MB`.

R042 optimized, commit `5956b4123d63c2723ed43b18dfeced69517ce7b4`:

- roots before/after: `400901 -> 402251`;
- selected parents / children: `1350 / 2700`;
- selection: `5.157262 s`;
- parent selection: `0.527326 s`;
- candidate generation: `0.086320 s`;
- candidate KNN: `4.542047 s`;
- test composite PSNR: `20.779274`;
- peak allocated memory: `9787.37 MB`.

Selection is `42.6%` faster. Parent selection is `64.2%` faster, candidate
generation is `92.2%` faster, and candidate KNN is `29.1%` faster. Root and
child counts are exactly equal. The test composite difference is `-0.000006`
dB, which is numerical noise rather than a behavioral change.

The one-time face-adjacency construction costs `1.028532 s` at setup. It is not
repeated during lifecycle events.

## Decision

R042 passes the exact execution gate. Further KNN/backend work is deferred:
the remaining gain would require a broader implementation change, while the
obvious duplicate work is already removed. The code is cleared for one formal
from-zero 30k structural run with the unchanged R040 configuration.

## Artifacts

- branch: `codex/r042-lifecycle-selection-acceleration`;
- profile output:
  `/home/wangyy/anigroom-r042-selection-runtime-20260807/outputs/r042_profile_resume600_to700_h100_20260807`;
- optimized gate output:
  `/home/wangyy/anigroom-r042-selection-runtime-20260807/outputs/r042_exact_v1_resume600_to700_h100_20260807`;
- optimized gate log:
  `/home/wangyy/logs/anigroom_r042_exact_v1_resume600_to700_h100.log`;
- frozen parent checkpoint:
  `/home/wangyy/anigroom-r040-child1-runtime-20260806/outputs/r040_child1_from_zero_700_h100_20260806/checkpoint_000600.pt`.
