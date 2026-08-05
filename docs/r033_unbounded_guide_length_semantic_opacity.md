# R033: Unbounded Guide Length And Semantic Opacity

Status: formal 30k run complete; guide-length and opacity changes are stable,
but the segment allocator is rejected and corrected in R034.

## Base And Scope

- measured comparison base: R032;
- executable base: the strict direct-3D from-zero route in
  `configs/stage1_baseline.env`;
- coupled change: remove the guide-length physical interval, remove padded
  opacity intervals, and remove the adaptive segment allocator's dependency on
  the old length interval.

This candidate intentionally changes checkpoint schema and does not migrate a
previous checkpoint.

## Guide Length

Each guide root stores a positive reference length and a zero-centered raw
coordinate:

`length = reference * exp(asinh(raw))`

The reference is the complete surface-inpainted clean-flow length field. The
observed 5%-95% interval selects robust anchors only; it is not a physical
decoder range. Initialization sets `raw = 0`, so decoded length exactly equals
the reference.

Guide densification interpolates both the current physical length and the
reference over the same surface support, then re-encodes the child coordinate.
Surviving rows retain both values. Render-root lifecycle applies the same rule
to its dormant absolute length field, so no hidden bounded endpoint remains.

## Opacity

Root opacity and tip-opacity ratio now decode directly with `sigmoid`. Their
semantic domain is `[0, 1]`; the previous formal `[0.05, 0.98]` and
`[0.08, 0.90]` padding is removed. Lifecycle interpolation occurs in physical
opacity space and uses only numerical inverse-sigmoid epsilon when re-encoding.

## Adaptive Segments

The old allocator mixed normalized decoder length and a fixed curvature score.
It also used the default `[0.012, 0.105]` range while formal guide decoding
actually used `[0.010, 0.220]`.

R033 tested a segment reference from the median initialized guide-length field.
Target linear spacing was `reference / min_segments`. Required count was the
maximum of:

- measured strand arc length divided by target spacing;
- accumulated turn divided by the angular resolution implied by
  `max_segments`.

This part was incorrect. Final physical strand length changed only about 5-7%
from R032, but mean segment count changed from `10.928` to `20.102`, increasing
preclip Gaussians from `13.94M` to `25.92M`. The initialization-derived spacing
therefore changed representation density rather than merely removing a cap.
R034 restores the accepted absolute-length/complexity linear allocator and
removes only its upper clamps.

## Deliberately Unchanged

- render length remains the accepted guide-relative `exp(asinh(raw))` residual;
- 5%-95% clean-flow filtering remains as robust evidence selection;
- root width, tip-width ratio, taper, child radius, and inactive curl/frizz
  decoder ranges remain for separate evidence-backed audits;
- RGB/color domains, normalized directions, lifecycle thresholds, losses, and
  training schedule are unchanged.

## Verification

- Python compilation: passed;
- positive unbounded length and inverse round trip: passed;
- opacity can approach both semantic endpoints: passed;
- guide/render lifecycle preserves reference fields and strict state reload:
  passed;
- segment count responds to fixed-scale length and curvature: passed;
- full repository suite at implementation time: 52 tests passed.

Formal acceptance still requires a full from-zero run, full-resolution metrics,
and fixed-protocol pure-fur structural QA.

## Formal Local Run

- output: `E:/anigroom_outputs/r033_from_zero_20260804_003748`;
- source: current accept-line worktree at `aa5bb8f` plus the documented R033
  working-tree change;
- initialization: from zero, with no checkpoint migration or optimizer resume;
- runtime guard: 25 GiB local-process GPU memory;
- first matched gate at iteration 1000: train/test composite PSNR
  `20.59893 / 20.77040` versus the preceding direct-3D executable reference
  `20.64085 / 20.80939`;
- root count at 1000: `105888` versus reference `106152`;
- adaptive segments at 1000: mean/max `10.81 / 14`, replacing the old fixed
  `10 / 10` while remaining far below the representation ceiling `36`;
- peak live CUDA allocation at 1000: `6.28 GB`, with no numerical error,
  lifecycle failure, or memory-guard event.

The H100 continuation completed at 30k with train/test composite PSNR
`33.27337 / 32.46867`, final `322381` render roots, and no lifecycle or numerical
failure. R032 ended at `32.47268` test composite, so the decoder changes are
metric-neutral. Canonical assets remain coherent. The unnecessary segment-count
increase is the reason R033 is not promoted as-is.
