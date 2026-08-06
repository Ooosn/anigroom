# R042 Exact Lifecycle Selection Acceleration

## Status

R042 is the execution-only child of R041. It preserves the R040/R041 model,
losses, schedule, lifecycle evidence, thresholds, selected-parent rule,
candidate set, child placement, pruning, interpolation, graph K, and graph
ordering. The matched 600-to-700 H100 gate passes. R042 removes the remaining
obvious duplicate work in lifecycle selection; it does not add a new method.

The formal from-zero 30k gate has now completed. R042 is the active
structural/lifecycle Stage 1 baseline: `child_count=1`, 400k initial independent
render roots, exact surface neighborhoods, and the unchanged finite 600-9000
render-root lifecycle. R036 remains the frozen higher-PSNR metric control.

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

## Formal From-Zero 30k Run

The formal run used source commit
`d4e22668b365541ec0741f969ec249e65f59f500` and the unchanged
`configs/r040_child1_dense_render_0_30k.env` behavior configuration.

- clean exit at iteration `30000`;
- elapsed H100 time: `11263.974 s` (`3 h 07 m 44 s`);
- final train/test composite PSNR: `33.48265 / 32.51543`;
- best test composite PSNR: `32.71918` at iteration `29000`;
- final render roots: `469737`;
- final training-metric Gaussian count: `5319498`;
- peak allocated CUDA memory: `13187.99 MB`;
- all `85` lifecycle events completed through iteration `9000`, with no
  lifecycle update or lifecycle-only statistic retention after that point;
- no OOM, NaN, restart, second GPU, fallback, resolution change, or event cap.

The saved checkpoint postprocess produces `5319408` Gaussians. The 90-Gaussian
difference from the final metric row is a boundary-ordering detail: the metric
row is sampled around the last optimizer step, while postprocess reconstructs
the persisted post-update state and a few strands cross discrete segment-count
thresholds. Training metrics and saved-state exports therefore report their own
measured counts rather than silently forcing them to match.

## Checkpoint Integrity

`checkpoint_030000.pt` was reloaded through the formal code path and contains:

- iteration `30000`, checkpoint kind `stage1_full`, schema version `6`;
- `469737` render roots and `4500` guide roots;
- `child_count=1` and `densify_until=9000`;
- `68` model tensor keys;
- `21` optimizer-state entries across `4` named optimizer groups;
- all `85` lifecycle records and the RNG state.

The checkpoint is `201752650` bytes with SHA-256
`5d05bf9a7b5e8f95f46498f97ce2c89d5f233a5f65830002a3ae42378b9dbdf9`.

## Structural QA

The checkpoint was exported deterministically as exactly 100k independent
strands and rendered with the established V11 Blender protocol at 1920x1080.
Three canonical views and eight full-resolution RGB views were inspected.
There is no whole-body crossing field, loop, foldback, curl collapse, or
backward strand segment. Body, belly, shoulder, neck, and legs remain coherent.

The 100k-strand numerical audit confirms:

- backward segment fraction: `0`;
- median/P95 chord length: `0.023636 / 0.043937`;
- median/P95 arc-to-chord ratio: `1.000074 / 1.007365`;
- median/P95 maximum local turn: `0.0900 / 1.0068` degrees.

A sparse extreme tail remains: chord-length max `0.220872` and maximum local
turn `14.8915` degrees. Visual inspection localizes it mainly to isolated tail
tip and head-fringe strands; it is not a global structural failure. It is
recorded for later asset cleanup rather than hidden by an animal-specific cap.

## Comparison

Against R038, R042 improves final/best test composite by
`+0.16955 / +0.20241 dB`, uses about `44.0%` fewer generated Gaussians, trains
about `8.8%` faster, and lowers peak allocation about `21.4%` despite starting
from four times as many independent render roots.

R036 remains slightly higher in reconstruction metric: R042 is
`-0.14779 / -0.12059 dB` at final/best test, while using about `62.6%` fewer
Gaussians. R042 is accepted for the cleaner independent-root representation,
finite exact lifecycle, lower capacity cost, and coherent structure; it is not
reported as a PSNR win over R036.

## Decision

R042 passes the exact execution gate, formal 30k gate, checkpoint-integrity
gate, and fixed-protocol structural gate. It replaces R038 as the active
structural/lifecycle baseline. Further KNN/backend work and sparse-tail asset
cleanup are deferred; neither blocks the next method stage.

## Artifacts

- branch: `codex/r042-lifecycle-selection-acceleration`;
- behavior config: `configs/r040_child1_dense_render_0_30k.env`;
- formal output:
  `/home/wangyy/anigroom-r042-selection-runtime-20260807/outputs/r042_exact_full_0_30k_h100_20260807`;
- formal checkpoint:
  `/home/wangyy/anigroom-r042-selection-runtime-20260807/outputs/r042_exact_full_0_30k_h100_20260807/checkpoint_030000.pt`;
- local postprocess root: `D:/RTS/_tmp/r042_30k_final`;
- canonical asset:
  `D:/RTS/_tmp/r042_30k_final/r042_030000_asset_side_y_v11_protocol.png`;
- opposite-side asset:
  `D:/RTS/_tmp/r042_30k_final/r042_030000_asset_side_y_pos_v11_protocol.png`;
- front asset:
  `D:/RTS/_tmp/r042_30k_final/r042_030000_asset_front_z_v11_protocol.png`;
- profile output:
  `/home/wangyy/anigroom-r042-selection-runtime-20260807/outputs/r042_profile_resume600_to700_h100_20260807`;
- optimized gate output:
  `/home/wangyy/anigroom-r042-selection-runtime-20260807/outputs/r042_exact_v1_resume600_to700_h100_20260807`;
- optimized gate log:
  `/home/wangyy/logs/anigroom_r042_exact_v1_resume600_to700_h100.log`;
- frozen parent checkpoint:
  `/home/wangyy/anigroom-r040-child1-runtime-20260806/outputs/r040_child1_from_zero_700_h100_20260806/checkpoint_000600.pt`.
