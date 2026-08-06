# R040 Child-Free Dense Render Roots

## Status

R040 is an isolated candidate derived from the completed R039 route. R039 is
unchanged and remains the direct comparison. R040 has passed the
full-resolution correctness, memory, and first-lifecycle gates. It is not an
accepted baseline because long-run structure, quality, and runtime have not
been measured.

## Question

R039 represents roughly 400k visible hairs as 100k render roots expanded into
four deterministic child strands. The child roots are tangent-plane offsets of
their parent; they are not independent surface roots and they do not resample
the guide field at the offset location. The offset envelope also changes along
the strand. This can create crossings even when the underlying 3D direction,
length, and brush-stiffness fields are locally coherent.

R040 tests the matched alternative:

- 400k independent surface render roots;
- one strand per render root;
- the same initial total strand budget as R039;
- guide interpolation and parameter-field smoothness retained;
- reconstructed sample-level strand smoothness removed.

This is a representation test, not an attempt to improve PSNR by adding
capacity. R039 and R040 both start with 400k strands.

## Same-Checkpoint Diagnostic

The R039 30k checkpoint was exported twice with the same deterministic 100k
record selection and the same 32 samples per strand:

1. original child geometry;
2. the corresponding parent geometry substituted for every selected child.

The matched parent-only render reduces local crossings and fuzziness around the
abdomen, shoulder/neck, and tail root. It does not remove every direction break,
so child expansion is a measured contributor rather than the sole cause.

Child-induced tangent deviation relative to each parent is:

- median: 0.459 degrees;
- P90: 2.380 degrees;
- P95: 3.389 degrees;
- P99: 5.522 degrees;
- maximum: 12.485 degrees.

The maximum child offset relative to parent arc length has median 10.9%, P95
27.7%, and maximum 62.4%. This is large enough to change the visible ordering
and crossing pattern without changing the parent groom field.

Diagnostic files are outside the repository:

- `D:/RTS/_tmp/r040_child_diagnostic/r039_child4_selection_parent_geometry_100k_samples32.npz`;
- `D:/RTS/_tmp/r040_child_diagnostic/r039_matched_parent_geometry_side_y_v11_protocol.png`;
- `D:/RTS/_tmp/r040_child_diagnostic/r039_matched_parent_geometry_side_y_pos_v11_protocol.png`;
- `D:/RTS/_tmp/r040_child_diagnostic/r039_matched_parent_geometry_front_z_v11_protocol.png`.

## Runtime Change

The candidate config is
`configs/r040_child1_dense_render_0_30k.env`.

Relative to R039 it changes only:

- `CHILD_COUNT: 4 -> 1`;
- `ROOT_COUNT: 100000 -> 400000`;
- removal of the obsolete `STRAND_SHAPE_SMOOTH_WEIGHT` contract.

The trainer no longer reconstructs 32-point strands every iteration solely to
compare neighboring sample tangents and curvatures. That loss operated on
render-root strands in world coordinates, did not parallel-transport between
surface frames, and duplicated the parameter-field regularizers. R040 keeps
the existing 3D direction, length, width, coverage, and effective-groom graph
losses.

`LOCAL_CHILD_COLOR_SUPPORT=1` remains enabled. With one child this provides one
local RGB delta per render root. Its parameter count is matched to R039:
`100k x 4` deltas becomes `400k x 1` deltas.

## First Gate

Before any long run:

1. run the exact 1920x1080 one-batch preflight;
2. verify 400k roots and one child are present in the saved config/report;
3. record generated Gaussian count and setup/iteration time;
4. require full forward and backward without fallback;
5. require peak allocated CUDA memory below the configured 25 GiB guard;
6. reject any hidden child expansion or sample-level strand-loss path.

Only after this gate passes may R040 proceed to a measured training interval.
The next decision is based on fixed-protocol pure-fur structure, speed, memory,
and composite PSNR together. A denser secondary residual-guide layer is not
part of this first experiment.

## Full-Resolution Preflight

The exact R040 configuration completed a 1920x1080 H100 forward, backward, and
evaluation batch without fallback:

- source commit: `573e833b9d19b48f9273e306b4a2bab8731ac081`;
- render roots / child count: `400000 / 1`;
- initial render graph: `3200000` directed edges, built in `18.666 s`;
- generated Gaussians: `4236979`;
- peak allocated CUDA memory: `8946.78 MB`;
- `nvidia-smi` process memory: `9724 MB`;
- post-evaluation allocated / reserved memory: `1134.66 / 1710 MB`;
- process exit status: zero.

The preflight output is:
`/home/wangyy/anigroom-r040-child1-runtime-20260806/outputs/r040_child1_preflight_h100_20260806_batch_preflight`.

## 700-Iteration Lifecycle Gate

A separate from-zero gate used the exact R040 model and training values, with
only the terminal iteration, evaluation interval, and checkpoint iterations
shortened to expose the first two lifecycle events. It completed normally and
saved both 600- and 700-iteration checkpoints:

- output:
  `/home/wangyy/anigroom-r040-child1-runtime-20260806/outputs/r040_child1_from_zero_700_h100_20260806`;
- final train/test composite PSNR: `20.59137 / 20.77921`;
- final roots: `402252`;
- generated Gaussians before the 700 event: `4246569`;
- peak allocated CUDA memory: `9831.82 MB`;
- peak reserved CUDA memory: `11638 MB`;
- highest sampled `nvidia-smi` process memory: `11312 MB`;
- elapsed time including evaluation every 100 iterations and two graph
  rebuilds: `359.321 s`;
- exit status: zero.

The uncapped absolute threshold does not cause global growth in this gate:

| Iteration | Roots before | Above threshold | Local maxima / parents | Children | Deleted parents | Roots after |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 600 | 400000 | 26513 | 901 | 1802 | 901 | 400901 |
| 700 | 400901 | 30601 | 1351 | 2702 | 1351 | 402252 |

R039 selects `1269` and `1663` parents at the same two events, so R040 is not
silently densifying more roots merely because it starts with four times as
many independent roots. The threshold-pass fraction is also lower in R040.

## Measured Bottleneck

The remaining concern is runtime, not memory. R040 rebuilds a roughly `3.2M`
edge render graph after each lifecycle event:

- iteration 600: `18.289 s`;
- iteration 700: `18.585 s`.

The matched R039 events rebuild roughly `0.81M` edges in `1.568-1.601 s`.
Therefore R040 passes the minimal gate but cannot proceed directly to a 30k
acceptance run. The next experiment must preserve the parameter-field
semantics while avoiding repeated full reconstruction of the dense render
graph. Raising K or restoring sample-level strand smoothing is explicitly not
the response to this bottleneck.

## Gate Decision

R040 is a valid candidate representation:

- no hidden child expansion remains;
- the full-resolution differentiable path is intact;
- memory is safely below the 25 GiB guard;
- uncapped evidence/local-maximum densification remains sparse;
- optimizer state migrates for all 21 state-bearing parameters;
- checkpoints at both lifecycle boundaries are loadable artifacts.

It is not yet the baseline. R039 remains frozen while R040 advances only after
the dense-graph runtime issue is addressed and measured without changing the
grooming objective.
