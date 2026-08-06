# R041 Exact Dense Graph Acceleration

## Status

R041 is the execution-only child of R040. It preserves the R040 model,
objectives, lifecycle thresholds, root placement, interpolation semantics, and
ordered smoothing graph. The 600-to-700 H100 gate passes. R041 removes the
dense-graph runtime blocker but is not a 30k structural acceptance result.

## Problem

R040 replaces four deterministic child strands with four times as many
independent render roots. At roughly 400k roots, each lifecycle event rebuilt
about 3.2M directed render-smoothing edges on CPU:

- iteration 600: `18.289 s`;
- iteration 700: `18.585 s`.

Repeated full CPU reconstruction would spend about 31 minutes on graph updates
over 100 events. Simply retaining old edges is not exact: retained render roots
can move barycentrically, so their distance ordering and guide support can
change even when only a small number of roots are inserted or removed.

## Exact Execution Change

R041 keeps full reconstruction and moves its regular path to CUDA:

1. bucket every render root by its primary guide root;
2. form each query root's candidate union from the same guide-support buckets
   used by R040;
3. remove the query root itself;
4. select the same K nearest roots by Euclidean distance;
5. preserve the R040 tie contract: distance first, then root ID;
6. use the original topology-expansion path for duplicate support, insufficient
   direct candidates, or a K-boundary distance tie.

Squared distance is used for CUDA ranking because it preserves Euclidean
ordering. It does not change graph membership. The CUDA path never lowers K,
duplicates a neighbor, or crosses to global ambient-space KNN.

Render-to-guide support is also rebuilt exactly on CUDA. A render-only
lifecycle event now reuses the unchanged guide Voronoi/source topology and
recomputes support for every current render root. It does not migrate retained
support or freeze stale neighbor IDs. Guide densification still triggers a full
guide-topology rebuild.

## Verification

The focused suite compares accelerated output against an independent reference
implementation and covers:

- CPU and CUDA ordered-edge equality;
- duplicate-support topology expansion;
- the exact-K case where the direct candidate count includes the query itself;
- CUDA/CPU support-ID and vertex-path equality;
- zero-centered geometry and R040 lifecycle contracts;
- the folded-surface graph regression.

Result: `43 passed`; the standalone surface-graph regression also passes.

## H100 Matched Gate

Both R041 runs resume the frozen R040 `checkpoint_000600.pt`, retain its RNG and
optimizer state, and execute iteration 700 with the same configuration.

R040 reference:

- roots before/after: `400901 -> 402252`;
- selected parents / children / deleted parents: `1351 / 2702 / 1351`;
- render edges after update: `3218016`;
- render graph rebuild: `18.5848 s`;
- test composite PSNR: `20.779205`;
- peak allocated memory: `9831.82 MB`.

R041 measured gate, commit `55daa2586a0d442f754ebf5a83fb66f2f58e583c`:

- roots before/after: `400901 -> 402252`;
- selected parents / children / deleted parents: `1351 / 2702 / 1351`;
- render edges after update: `3218016`;
- exact support rebuild for `402252` roots: `0.0508 s`;
- support fallback queries: `0`;
- exact render graph rebuild: `0.1988 s`;
- test composite PSNR: `20.779194`;
- peak allocated memory: `9787.37 MB`.

The lifecycle graph rebuild is `93.5x` faster. At this measured rate, 100 graph
updates cost about 20 seconds instead of about 31 minutes. The one-time initial
400901-root graph build, including first-use CUDA setup, is `3.1845 s` instead
of about `18.47 s`.

Separating support rebuild from unchanged guide topology also reduces
`render_update_seconds` from `10.8762 s` to `2.8472 s`. The complete measured
event falls from `20.3259 s` in the first accelerated run to `12.2782 s`.
The remaining dominant cost is lifecycle candidate selection (`8.9266 s`), not
support or graph reconstruction; it is a separate future execution target.

## Artifacts

- source branch: `codex/r041-incremental-root-graph`;
- first accelerated output:
  `/home/wangyy/anigroom-r041-graph-runtime-20260806/outputs/r041_resume600_to700_h100_20260806`;
- final support-reuse output:
  `/home/wangyy/anigroom-r041-graph-runtime-20260806/outputs/r041_support_reuse_resume600_to700_h100_20260806`;
- frozen parent checkpoint:
  `/home/wangyy/anigroom-r040-child1-runtime-20260806/outputs/r040_child1_from_zero_700_h100_20260806/checkpoint_000600.pt`.

The held H100 qlogin remains allocated. R040 and its outputs are unchanged.
The candidate-count boundary correction is included in implementation commit
`d137dbdb905e6cabe08bcec8901a6e32975d9450`; its focused regression is part of
the 43-test result above and does not change the measured real-data event.
