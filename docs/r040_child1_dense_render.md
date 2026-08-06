# R040 Child-Free Dense Render Roots

## Status

R040 is an isolated candidate derived from the completed R039 route. R039 is
unchanged and remains the direct comparison. R040 is not accepted until the
full-resolution memory gate and structural validation pass.

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
