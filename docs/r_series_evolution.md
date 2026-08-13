# R-Series Evolution

Status date: 2026-08-13. Active structural/lifecycle baseline: R043. Frozen
higher-PSNR metric control: R036. Accepted appearance and strict zero-foldback
reference: R050. Latest trained advanced-geometry research checkpoint: R059.

This document is the compact decision history. Detailed measurements and
artifact paths remain in `docs/accept_line_recovery_ledger.md` and the
individual R-series documents.

## Baseline Milestones

| Milestone | Why it was locked |
| --- | --- |
| R001 | Reproduced the historical V11 result and established trustworthy data, evaluation, and checkpoints. |
| R003 | Made surface-aware typed interpolation the shared attribute contract. |
| R007d | Established uncapped, threshold/local-max render-root lifecycle. |
| R009 | Replaced Euclidean neighborhoods with mesh-surface neighborhoods. |
| R018 | Removed the active aggregate absolute physical shape prior. |
| R027 | Preserved optimizer state across lifecycle changes. |
| R030 | Removed sparse long-hair tail spikes without an absolute cap. |
| R032 | Made guide-length smoothing invariant to guide density. |
| R036 | Completed the current positive guide/render hierarchy for length, width profile, and child spread. |
| R038 | Added the explicit normal-to-groom brush curve and a finite 600-9000 render lifecycle. |
| R039 | Replaced the ambiguous interior deformation with one direction-aware quadratic centerline. |
| R040 | Replaced four deterministic children with 400k independent render roots and `child_count=1`. |
| R041 | Made the dense surface graph exact and practical by removing redundant rebuild work. |
| R042 | Completed exact lifecycle-selection acceleration and passed the formal from-zero 30k gate. |
| R043 | Restored density-matched K32 render support and removed the sparse long/sharp centerline tail without a physical cap. |
| R049 | Added the smooth secondary-guide geometry field used by the appearance branch. |
| R050 | Added generated-Gaussian RGB residual without degrading the R049 strand field. |
| R054 | Moved curl/frizz ownership back to the smooth primary guide after R053 exposed dense residual noise. |
| R055 | Staged primary shape/appearance before zero-centered secondary shape residual and reduced R054 foldback. |
| R057 | Corrected RGB-flow gradient ownership without changing the staged-shape forward contract. |
| R058 | Replaced invalid curl/frizz deformation with physical curl and independent band-limited frizz. |
| R059 | Trained the R058 geometry under the unchanged R057 contract and removed most, but not all, sparse foldback. |

## Recovery And Interpolation: R000-R007

| Run | Test | Decision and lasting result |
| --- | --- | --- |
| R000 | Recovery audit | Established immutable evidence, input hashes, evaluation contract, and no-fallback rules. |
| R001 | Exact V11 reproduction | Passed: 30k test composite `32.1891`, matching historical `32.1814`. Became the trustworthy origin. |
| R002 | V4 clean-flow only | Preflight passed after fixing configuration drift. V4 3D surface flow became the accepted input. |
| R003 | Extend V4 surface interpolation to groom attributes | Passed and locked: final/best `32.5895/32.7361`. Shared typed surface interpolation retained. |
| R004 | Topology-local evidence split/delete | Passed: final `32.6023`, metric-neutral but much slower. Placement/lifecycle semantics retained. |
| R005 | Mean-visible Gaussian gradient score | Diagnostic rejected: selected 45%-56% of roots and behaved almost globally. Not active. |
| R006 | Better clean-flow length/outward initialization | Passed: final `32.607`; initialization improvements retained. |
| R007 | Remove render event cap and calibrate absolute score threshold | R007d passed: final/best `32.609/32.755`. Threshold `0.00075`, no event budget, local maxima retained. |

## Surface Continuity And Direction: R008-R018

| Run | Test | Decision and lasting result |
| --- | --- | --- |
| R008 | Edge confidence plus effective smoothness | Two-pixel edge band and `0.006` effective smooth retained; wider edge bands rejected. |
| R009 | Mesh-surface smoothing graph | Accepted structural baseline: final `32.2826`. Prevents smoothing across nearby but disconnected surface sheets. |
| R010 | Parallel-transported direction and relative length | Useful control; surface-covariant metric retained, full run not final. |
| R011 | Exact normalized local 3D direction | Representation fix retained; direction round-trip error removed. |
| R012 | Extra low-confidence direction smoothing | Rejected: no intended improvement. |
| R013 | Effective/guide relative-length continuity | Formulation retained; tested weight alone was too weak. |
| R014 | Five-times stronger R013 weight | Rejected: could not repair guide-field outliers. |
| R015 | Guide log-length Laplacian | Useful diagnosis, later superseded by R032's density-invariant formulation. |
| R016 | Surface reconstruction of unreliable initialized length | Retained: guide floor occupancy fell from `22.044%` to `0.133%`. |
| R017 | Parallel-transport consensus reconstruction of 3D direction | Retained: head-side direction P95 fell from `34.99` to `24.26` degrees. |
| R018 | Remove aggregate absolute physical shape prior | Accepted: final/best `32.1385/32.2899`; structure became shorter and more surface-following. |

## Residual Representation And Tail Control: R019-R030

| Run | Test | Decision and lasting result |
| --- | --- | --- |
| R019 | Zero-centered render geometry | Direct 3D direction retained; additive physical length rejected because 13.55% hit the minimum. |
| R020 | Bounded log-ratio length | Underpowered and superseded. |
| R021 | Raw exponential log-ratio | Rejected: one tail escaped to a 14.2x ratio. |
| R022 | Raw-coordinate smoothness | Rejected: exponential decoder still amplified a sparse tail. |
| R023 | `exp(asinh(raw))` length ratio | Decoder retained: positive, relative, unbounded, and tail-moderated. Standalone route superseded. |
| R024 | Saturating coordinate-consistent prior | Rejected: maximum length reached `0.7801`. |
| R025 | Non-saturating raw prior | Improved tail but rejected: visible tail line remained. |
| R026 | Densification through residual unlock | Exposed optimizer-state reset every 100 iterations; run stopped. |
| R027 | Lifecycle-aware Adam row migration | Accepted: final/best `32.3811/32.5517`; optimizer continuity retained. |
| R028 | Guide optimization from iteration one | Strong metric evidence (`33.0205` test) but exact branch rejected for width/opacity/length drift. |
| R029 | Replace L1 with full L4 norm | Rejected: reduced tail but over-suppressed ordinary residuals and lost about `0.245 dB`. |
| R030 | Mean L1 plus unlock-scaled `L4-L2` concentration | Accepted: max length fell 63.94% with only `-0.045 dB`; tail spikes disappeared. |

## Guide Lifecycle And Hard-Range Removal: R031-R038

| Run | Test | Decision and lasting result |
| --- | --- | --- |
| R031 | Exact forward-support guide evidence and uncapped local maxima | Lifecycle retained; exposed density-dependent guide smoothing. Improved best test by `0.148 dB`. |
| R032 | Density-invariant intrinsic guide-length smoothness | Accepted: final/best `32.4727/32.6652`; global/tail roughness fell 57.3%/72.4% from R031. |
| R033 | Positive-unbounded guide length and semantic opacity | Decoder accepted; initialization-relative segment allocator rejected because Gaussians doubled without matching length growth. |
| R034 | Absolute uncapped linear segment allocation plus direct width trial | Segment allocator accepted: `14.27M` Gaussians, mean/max `10.95/17`. Direct width ownership rejected due cylindrical saturation. |
| R035 | Hierarchical width profile | Accepted: final/best `32.6597/32.8440`; no width collapse. Guide owns low-frequency profile, render roots own relative residuals. |
| R036 | Hierarchical positive child spread | Accepted and frozen: final/best `32.6632/32.8398`; coherent positive spread without physical endpoints. |
| R037 | Move every coverage control to early schedule | Deferred before formal run. Only R036 child spread keeps its measured 1k-to-7k ramp. |
| R038 | Guide-owned brush curve plus finite render lifecycle | Accepted structural/lifecycle baseline: final/best `32.3459/32.5168`; 34.0% fewer roots, 33.2% fewer Gaussians, and 43.2% less H100 time than R036, with zero backward strand segments. |

## What R036 Adds Over The Last Major Baseline R032

R036 keeps R032's flow, interpolation, lifecycle, smoothing, optimizer, and
tail-control results, then adds three representation corrections:

1. R033 removes physical guide-length endpoints and padded opacity endpoints.
2. R034 restores segment count to absolute physical length and complexity,
   with no upper cap.
3. R035-R036 move width profile and child spread into the same guide-owned,
   render-residual hierarchy instead of direct bounded render fields.

Measured change from R032 to R036:

- final test composite: `32.47268 -> 32.66322` (`+0.19054 dB`)
- best test composite: `32.66519 -> 32.83977` (`+0.17458 dB`)
- final train composite: `33.28816 -> 33.42397` (`+0.13581 dB`)

The gain is secondary to the representation result: length, root width,
width taper, and child spread no longer depend on animal-scale physical
decoder endpoints, while the canonical pure-fur coat remains coherent.

## What R038 Adds Over R036

R038 retains R036's clean-flow, interpolation, hierarchy, smoothing, optimizer,
and positive-field contracts, then makes two isolated changes:

1. A guide-owned brush strength explicitly controls a smooth
   normal-to-groom transition while preserving root, tip, straight length, and
   endpoint 3D direction. Bend becomes a smooth unbounded interior offset;
   curl/frizz remain disabled.
2. Evidence-driven render lifecycle runs every 100 iterations from 600 through
   9000, then both root updates and lifecycle-only statistics stop.

The final test metric is `0.31734 dB` below R036, so R038 is not a metric win.
It is accepted because fixed-protocol structure remains coherent while root,
Gaussian, memory, and runtime cost fall substantially. R036 stays frozen for
metric comparisons.

## Independent Dense Roots And Exact Execution: R039-R043

| Run | Test | Decision and lasting result |
| --- | --- | --- |
| R039 | One-turn direction-aware centerline | Completed: final/best test `32.2164/32.4111`; frozen as the direct structural parent. |
| R040 | 400k independent roots, `child_count=1` | Representation and memory gates passed. Exact graph rebuild cost `18.3-18.6 s` per lifecycle event blocked a formal run. |
| R041 | Exact dense surface-graph acceleration | Preserved selected roots and ordered edges while reducing graph rebuild from `18.5848 s` to `0.1988 s`. |
| R042 | Exact lifecycle-selection acceleration plus formal 30k | Selection fell from `8.9911 s` to `5.1573 s` with identical parent/child counts. Formal 30k completed at final/best test `32.51543/32.71918`, `469737` roots, `5.319M` Gaussians, and `13.188 GB` peak allocation. Accepted, then frozen as R043's K8 parent. |
| R043 | Density-matched K32 render support | Final/best test is `32.51159/32.71421`, effectively tied with R042. In the fixed 100k-strand audit, maximum length falls from `0.22087` to `0.13155`, count above `0.15` falls from 7 to 0, and maximum local turn falls from `14.89` to `2.73` degrees. Accepted as active structural/lifecycle baseline. |

R043 improves final/best test composite over R038 by
`+0.16571/+0.19744 dB`, uses about `44.2%` fewer Gaussians, and retains the
independent-root representation. It remains `0.15163 dB` below R036 at the
final test metric, so
R036 remains the metric control rather than being rewritten by the structural
decision.

## Appearance And Optional Shape: R049-R055

| Run | Test | Decision and lasting result |
| --- | --- | --- |
| R049 | Secondary-guide geometry through 30k | Accepted as R050's structural parent. The 100k-strand audit reaches `0.02047/0.07741` local relative-length mean/P95 with zero backward segments. |
| R050 | Generated-Gaussian RGB residual | Accepted appearance checkpoint: final/best test `32.12111/32.20936`; the residual adds `0.88-2.73 dB` over fixed views without changing strand geometry. |
| R053 | Simultaneous primary and secondary curl/frizz | Rejected as a shape baseline. PSNR rises, but local turning and stripe-correlated shape noise worsen even with Gaussian RGB residual. |
| R054 | Primary-guide-only curl/frizz | Retained as a cleaner ownership control. Backward strands fall from R053 `560` to `375`, but local-turn P95 remains `57.30 deg`. |
| R055 | Primary shape plus Gaussian appearance at 20k-25k, then secondary shape residual at 25k-30k | Accepted as the latest controlled shape checkpoint, not the default baseline. Versus R054, backward strands fall `375 -> 159`, local-turn P95 falls `57.30 -> 50.03 deg`, and local length continuity improves, with `-0.132 dB` fixed-view mean PSNR. R050 remains the strict structural reference. |
| R057 | Exclude color parameters only from the existing RGB-flow backward | Accepted gradient-ownership correction. Reconstruction is metric-neutral versus R055; the sparse foldback tail is not improved. |
| R058 | Redesign curl/frizz forward geometry | Accepted implementation with 133 passing tests after the R059 lifecycle/schema gates were added. |
| R059 | Train the strict R058 schema on the complete R057 contract | Accepted as the latest trained advanced-geometry research checkpoint. Final/best test is `32.25537/32.33647`, fixed eight-view mean is `33.32501`, and Gaussian RGB residual contributes `+1.60937 dB`. Backward strands fall `177 -> 34`, maximum-turn P95 `50.309 -> 10.265 deg`, and arc/chord P99 `1.20297 -> 1.14865`. The remaining 34 hooks form one compact head-crown patch, so R050 remains the strict zero-foldback reference. |

## Effective Findings

The strongest reusable findings are:

- interpolate and smooth physical groom fields on the mesh surface, not in UV
  or ambient Euclidean neighborhoods;
- transport 3D directions between surface frames before interpolation;
- use guide-owned low-frequency fields and zero-centered render residuals;
- use positive reference-relative coordinates for positive physical fields;
- preserve optimizer row state when roots are inserted or replaced;
- densify from accumulated absolute Gaussian/root evidence and local maxima,
  not from a global percentile or fixed event budget;
- make guide regularization density invariant when guide roots densify;
- control sparse residual tails with a concentration term instead of a hard
  physical length cap;
- allocate Gaussian segments from final physical curve length/complexity, not
  an initialization-relative capacity multiplier;
- judge every route with both composite reconstruction and fixed-protocol
  pure-fur structure.
- hand off optional shape from the smooth primary guide before enabling a
  zero-centered secondary residual; Gaussian RGB residual remains a separate
  high-frequency appearance outlet.

## Current Boundary

R043 remains the base structural/lifecycle route and does not enable
curl/frizz or Gaussian-level RGB residual. R050 is the accepted appearance and
strict zero-foldback checkpoint built on the smooth R049 geometry. R059 is the
latest trained advanced-geometry research checkpoint: it preserves R057's
appearance handoff while removing `80.8%` of strict foldbacks, but one compact
head-crown cluster remains. These roles must remain distinct: R043 for
independent-root lifecycle, R050 for strict appearance/structure, and R059 for
subsequent controlled advanced-geometry work.
