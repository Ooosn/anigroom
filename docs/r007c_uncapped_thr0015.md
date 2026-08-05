# R007c Uncapped Render-Root Lifecycle, Lower Threshold

Historical status: completed conservative calibration and superseded by R007d
at threshold `0.00075`. This file records the intermediate hypothesis only.

## Purpose

R007b proved that removing the fixed render-root split count can be stable when the evidence threshold is meaningful. However, `DENSIFY_SCORE_THRESHOLD=0.003` was conservative: it produced fewer final roots than R006 and stopped growth early.

R007c keeps the fixed count cap removed, but lowers the absolute threshold:

```bash
MAX_SPLITS_PER_EVENT=0
DENSIFY_SCORE_THRESHOLD=0.0015
LIFECYCLE_SCORE_MODE=raw
```

The goal is not simply to create more roots globally. The goal is to let high-evidence regions exceed the old fixed 1024 parent/event budget while still preventing the inherited low threshold from becoming near-uniform global growth.

## Acceptance Checks

R007c is acceptable only if:

- root growth is higher than R007b and can exceed R006 locally;
- growth is still evidence-driven, not uniform over the whole surface;
- memory remains under the 30 GB guard;
- phase-A PSNR is not worse than R006/R007b;
- final pure-hair structure is not noisier than R006.

If growth returns to near-global behavior, the threshold is still too low and R007c is rejected.

## Run Log

The `0.0015` route was stable but too conservative. R007d lowered the threshold
to `0.00075` and completed the accepted 30k calibration.
