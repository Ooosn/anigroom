# R049 Secondary-Guide 30k Validation

## Status

Running. R043 remains the accepted baseline until this structural validation
finishes. R048 established the corrected differentiable vector transport and
the matched 16k structure comparison; R049 changes no algorithm setting and
only continues that exact state to the formal 30k endpoint.

## Source State

- Code: R048 commit `4718451` plus this continuation config and ledger.
- Resume checkpoint:
  `/home/wangyy/anigroom-r048-regularized-vector-transport-runtime-20260808/outputs/r048_regularized_vector_transport_resume10k_16k_h100_20260808/checkpoint_016000.pt`
- Secondary guide count: 20,000.
- Render root count at 16k: 469,402.
- Geometry residual domain: secondary guide.
- Render-to-secondary interpolation: K8.
- Secondary geometry regularization: surface K4.
- Densification ended at 9k; no lifecycle topology change can occur in this
  continuation.

## Single Variable

There is no method variable. The R048 endpoint changes from 16k to 30k so the
late residual schedule can be judged under the same optimizer, RNG, views,
renderer, root topology, losses, and regularization.

## Acceptance Gate

1. Finish 30k without fallback, OOM, topology drift, or configuration change.
2. Use composite PSNR only as a diagnostic; a lower score than R043 is
   acceptable when it comes from refusing per-strand RGB-driven geometry.
3. Under the fixed 100k-strand asset protocol, retain continuous direction and
   length fields without isolated long strands, crossings, curl-back, or local
   coverage holes.
4. Recompute local arc-length and direction discontinuity statistics at 30k.
5. Promote the secondary-guide representation only if its structural advantage
   remains through the late schedule.

## Result

Pending.
