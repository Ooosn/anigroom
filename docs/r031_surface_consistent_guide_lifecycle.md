# R031: Surface-Consistent Guide Lifecycle

Status: completed control. The lifecycle change is retained as a valid method
component, but R030 remains the accepted baseline until R032 tests the exposed
guide-smoothness discretization error.

## Observed R030 Failure

The remaining distal-tail artifact is not the sparse render-length escape fixed
by R030. Parent-only export retains a coherent comb-like shaft and broom-like
tip. Distal parent lengths are ordinary, child expansion is not causal, and the
median guide-to-effective direction change is about one degree.

The guide lifecycle audit exposed two general inconsistencies:

1. Forward grooming attributes use mesh-surface interpolation support, but
   guide densification reassigns render evidence with a separate Euclidean
   8-nearest-neighbor query.
2. Every guide event applies a global top-32 budget after the absolute evidence
   threshold. All 26 R030 events are budget-saturated: 3179-4047 guides pass the
   threshold at each event, exactly 32 are selected, and the guide count grows
   from 4500 to 5332. Only one of the 832 new guides lies on the distal tail.

This explains why a small appendage can remain under-resolved while larger
high-residual regions keep taking the global budget. It does not justify a tail
mask, tail direction rule, or tail length threshold.

## Single Method Change

R031 replaces the guide selection policy as one coherent lifecycle operation:

1. Attribute each render root's normalized evidence to the exact guide IDs and
   weights used by the forward surface interpolator.
2. Keep guides above the existing absolute evidence threshold.
3. Apply non-maximum suppression on the intrinsic guide surface graph, retaining
   one representative maximum per high-evidence neighborhood.
4. Insert all resulting local maxima. No global event count cap is active.

Render-root lifecycle, score definition, thresholds, schedules, child
placement, optimizer migration, losses, R030 length prior, initialization, and
visualization protocol remain unchanged.

## Acceptance Checks

- Focused unit/regression tests must pass without fallback.
- The first 11k guide event must report forward surface attribution, an
  intrinsic local-max count, no parent budget, and a finite memory/runtime cost.
- Guide growth must be evidence-driven rather than a constant count per event.
- Distal-tail guide coverage must grow when its local evidence exceeds the same
  global threshold; no body-region label is allowed.
- The 16k lifecycle state must remain stable enough to continue to 30k.
- Final comparison uses full-resolution RGB plus the fixed parent-only and
  canonical asset protocols. R031 is accepted only if the tail improves without
  transferring artifacts or materially degrading reconstruction.

## Formal Result

The formal H100 route completed through 30k without fallback or OOM. The final
checkpoint has a matching local/remote SHA-256 and a formal `RUN_COMPLETE`
marker. Guide roots increased from R030's `5,332` to `11,875`; render-root
capacity remained matched (`318,898` versus `318,963`). Final generated
Gaussians are `13,930,065`, and peak allocated CUDA memory is `24,271.51 MB`.

Reconstruction improved consistently at matched late iterations:

| Iteration | R030 test composite | R031 test composite | Delta |
| ---: | ---: | ---: | ---: |
| 24k | 32.34557 | 32.48486 | +0.13929 |
| 26k | 32.35966 | 32.51921 | +0.15955 |
| 27k | 32.46486 | 32.60917 | +0.14431 |
| 29k | 32.53251 | 32.68041 | +0.14790 |
| 30k | 32.33575 | 32.48577 | +0.15002 |

The final train/test composite PSNR is `33.31456 / 32.48577`. The fixed
100k-strand side, opposite-side, and front protocols show no global curl,
cross-region contamination, or capacity collapse. The remaining visible error
is still concentrated at thin structures, especially a fan-like distal tail.

## Exposed Smoothness Error

R031 also proves that the accepted guide-length smoothness is not invariant to
guide sampling density. At 30k, the denser guide graph reports a smaller loss
while the physically normalized surface gradient is much larger:

| Quantity | R030, 5,332 guides | R031, 11,875 guides |
| --- | ---: | ---: |
| accepted relative edge loss | 0.003626 | 0.002454 |
| area-weighted intrinsic log-gradient | 10.723 | 18.335 |
| distal-tail intrinsic log-gradient | 23.845 | 52.972 |

The final distal-tail median interpolated guide length changes only from
`0.02565` to `0.02647`, and the median render-residual ratio changes from
`0.9675` to `0.9738`. The visible degradation is therefore not a new sparse
render-residual escape. It is a denser guide field whose old edge-difference
loss becomes numerically weaker as edges shorten.

A second independent effect remains. On the original 4,500 guides at 30k,
distal-tail learned/target length drift correlates with clean-flow view count at
`-0.448` and axis evidence confidence at `-0.529`; this relationship is absent
outside that thin diagnostic region. Smoothness can remove local roughness but
cannot by itself remove a coherent low-observability drift.

## Decision

Retain R031's forward-support evidence attribution, intrinsic local-max
selection, and uncapped guide insertion as the preferred guide lifecycle. Do
not yet replace R030 with the complete R031 run: the lifecycle gain exposes a
separate discretization defect in the guide-length regularizer. R032 tests only
that regularizer. No tail mask, body-part coefficient, absolute length rule, or
animal-specific threshold is justified by this result.

Formal artifacts:

- HGC output: `/home/wangyy/anigroom-r031-guide-lifecycle-20260801/outputs/r031_surface_consistent_guide_lifecycle_16k_30k_20260802_h100`
- Local checkpoint: `D:/RTS/_tmp/r031_30k_final/checkpoint_030000.pt`
- Checkpoint SHA-256: `b4df69259e3d82e3ad462701d3db197c9ece5f8ffd250df3a6ba0bdeddab5e29`
- Canonical side asset: `D:/RTS/_tmp/r031_30k_final/r031_030000_asset_side_y_v11_protocol.png`
- Opposite-side asset: `D:/RTS/_tmp/r031_30k_final/r031_030000_asset_side_y_pos_v11_protocol.png`
- Front asset: `D:/RTS/_tmp/r031_30k_final/r031_030000_asset_front_z_v11_protocol.png`
- Structural diagnostics: `D:/RTS/_tmp/r031_30k_final/guide_smoothing_diagnostic.json`
