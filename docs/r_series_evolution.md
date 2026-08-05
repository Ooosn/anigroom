# R-Series Evolution

Status date: 2026-08-05. Current frozen baseline: R036.

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

## Guide Lifecycle And Hard-Range Removal: R031-R036

| Run | Test | Decision and lasting result |
| --- | --- | --- |
| R031 | Exact forward-support guide evidence and uncapped local maxima | Lifecycle retained; exposed density-dependent guide smoothing. Improved best test by `0.148 dB`. |
| R032 | Density-invariant intrinsic guide-length smoothness | Accepted: final/best `32.4727/32.6652`; global/tail roughness fell 57.3%/72.4% from R031. |
| R033 | Positive-unbounded guide length and semantic opacity | Decoder accepted; initialization-relative segment allocator rejected because Gaussians doubled without matching length growth. |
| R034 | Absolute uncapped linear segment allocation plus direct width trial | Segment allocator accepted: `14.27M` Gaussians, mean/max `10.95/17`. Direct width ownership rejected due cylindrical saturation. |
| R035 | Hierarchical width profile | Accepted: final/best `32.6597/32.8440`; no width collapse. Guide owns low-frequency profile, render roots own relative residuals. |
| R036 | Hierarchical positive child spread | Accepted and frozen: final/best `32.6632/32.8398`; coherent positive spread without physical endpoints. |
| R037 | Move every coverage control to early schedule | Deferred before formal run. Only R036 child spread keeps its measured 1k-to-7k ramp. |

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

## Current Boundary

R036 does not yet contain Gaussian-level RGB appearance residuals or the future
brush-stiffness/base-curve representation. Those are separate candidates and
must not be described as part of this frozen baseline.
