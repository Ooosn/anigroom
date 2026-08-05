# R029 Hierarchical Length Ownership

## Status

`rejected as a full-prior replacement; its population-dilution diagnosis is
retained and resolved selectively by accepted R030`.

## Observed Failure

R027 retains one or two implausibly long strands near the tail tip. R028
diagnostics show that this is not a broken FPS/KNN neighborhood, a disconnected
surface graph, or opacity alone:

- the abnormal render root belongs to a complete surface neighborhood;
- its neighboring render roots are also long, so difference-only smoothness
  sees a locally coherent patch;
- the corresponding guide root has drifted far from its confidence-bearing
  clean-flow length target;
- the render-root length residual then amplifies that low-frequency drift.

The existing guide and render smoothness losses use global means. As render
densification grows the field to roughly 300k roots, a fixed small pathological
patch contributes progressively less to those means. This is a hierarchy and
loss-reduction problem, not evidence for an animal-region rule.

## Candidate Method

1. Guide length remains a learned positive field. A confidence-weighted log
   anchor preserves only the relative spatial pattern of the clean-flow length
   evidence. Its weighted global log scale is removed, so the entire coat can
   still become uniformly longer or shorter.
2. The render-root length residual prior uses the fourth-root fourth moment
   instead of mean L1. Sparse extreme residuals therefore keep a useful
   gradient as the root population grows. There is no physical length limit,
   percentile clamp, tail mask, or selected-root count.
3. Guide densification interpolates `target * confidence` and confidence, then
   divides them. Unobserved zero targets no longer dilute a valid child target.
4. Lifecycle diagnostics report the final effective length and the base field
   separately. Historical absolute long/wide threshold counters are removed.

## Validation Contract

First run a matched R027 20k -> 22k continuation to verify that the existing
outlier contracts without destabilizing P95, PSNR, memory, or the rest of the
coat. If that passes, run the formal candidate from the same accepted 9k
checkpoint to 30k and compare:

- full-resolution train/test composite PSNR;
- effective guide/render length distributions and guide-target relative drift;
- canonical deterministic 100k-strand structural image;
- canonical V11-protocol asset image;
- lifecycle and optimizer-state integrity.

R029 is rejected if it merely clips the visible strand, moves the failure to a
different region, or causes a broad coat-length collapse.

## Experiment Isolation

The first 20k -> 22k diagnostic combined the fourth-moment render residual
reduction with a guide relative-length anchor of `0.01`. It removed the visible
tail outliers, but the guide maximum increased and the formal 9k -> 10k run
lost about `0.61 dB` before render residuals had unlocked. That isolates the
loss to the guide anchor, so the combined candidate is rejected.

The active candidate changes one variable from R027:

- `RENDER_LENGTH_PRIOR_REDUCTION=population_stable_handoff`;
- `GUIDE_LENGTH_RELATIVE_ANCHOR_WEIGHT=0`.

The handoff uses the existing render-residual unlock multiplier. At multiplier
zero it is exactly R027's mean L1; at multiplier one it is the fourth-root
fourth moment over all render-root residuals. It adds no schedule parameter,
value threshold, quantile, spatial mask, or absolute grooming bound.

## Matched 20k -> 22k Result

The single-variable candidate resumes the exact R027 20k checkpoint and Adam
state. No densification, schedule, initialization, or other loss changes.

| 22k metric | R027 mean L1 | fourth moment |
| --- | ---: | ---: |
| test composite PSNR | 32.18475 | 32.10015 |
| effective length P95 | 0.035996 | 0.036750 |
| effective length P99 | 0.047045 | 0.047767 |
| effective length P999 | 0.078458 | 0.068299 |
| effective length max | 0.318382 | 0.123438 |
| max / P999 | 4.058 | 1.807 |
| raw render residual max | 5.16565 | 0.58576 |

The canonical V11-protocol 100k-strand render removes the isolated tail-tip
whiskers while preserving the ordinary coat distribution. The `0.0846 dB`
test-PSNR cost is small enough to justify a formal 9k -> 30k run, but this is
not yet an accepted baseline. The candidate guide maximum is `0.09098`, so the
full run must check whether optimization transfers the same failure into the
guide field.

## Full-Run Correction

Running the fourth-moment norm at full strength from 9k was rejected at 14k:
test composite fell from R027's `30.89468` to `30.52666` while the gap grew
monotonically after 11k. The render residual itself is gradually unlocked from
10k to 20k, but the first implementation regularized its raw coordinate at full
strength throughout that ramp. That schedule mismatch, not the absence of a
length clamp, caused the regression.

The corrected formal candidate continuously blends mean L1 into the
fourth-moment norm with the exact multiplier already used by the model to
unlock render geometry residuals. This preserves the accepted early learning
behavior and reaches the short-run validated norm when the residual is fully
active.

The corrected handoff still lost about `0.245 dB` by 14k because it eventually
replaces the whole ordinary residual prior. R030 keeps mean L1 and adds only
`L4-L2`, which is zero for coherent equal-magnitude residuals and positive for
sparse concentration. R029 is therefore historical evidence, not an active
candidate.
