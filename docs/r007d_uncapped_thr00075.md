# R007d Uncapped Render-Root Lifecycle, Lower Threshold

## Purpose

R007b (`0.003`) was stable but too conservative. R007c (`0.0015`) was also stable but only marginally denser than R007b by early phase A.

R007d lowers the absolute evidence threshold again:

```bash
MAX_SPLITS_PER_EVENT=0
DENSIFY_SCORE_THRESHOLD=0.00075
LIFECYCLE_SCORE_MODE=raw
```

This is still not the rejected bare-uncapped variant. The rejected variant inherited `2.5e-5`, which was below the median evidence score and caused near-global densification. R007d remains an absolute evidence gate, but should allow noticeably more local growth than R006/R007b/R007c.

## Acceptance Checks

Accept only if:

- selected parents per event are clearly higher than the old 1024 cap in high-evidence regions;
- root growth does not return to the 5k-6k/event near-global behavior;
- phase-A memory stays below 30 GB;
- phase-A PSNR does not collapse.

If early growth is still too close to R007c, the bottleneck is likely not the threshold but local-max neighborhood suppression or evidence construction.

## Result

Run:

```text
/home/wangyy/anigroom-r006-current/outputs/r007d_uncapped_thr00075_h100_20260722_191139
```

The run completed to 30k without traceback or OOM. It is the useful member of
the R007 threshold series so far:

| Iteration | Train composite PSNR | Test composite PSNR | Render roots |
| ---: | ---: | ---: | ---: |
| 9000 | 23.879 | 24.123 | 188224 |
| 10000 | 29.436 | 29.318 | 197690 |
| 12000 | 31.164 | 30.667 | 198762 |
| 19000 | 33.079 | 32.255 | 198762 |
| 24000 | 33.442 | 32.571 | 198762 |
| 29000 | 33.617 | 32.755 | 198762 |
| 30000 | 33.662 | 32.609 | 198762 |

Final generated Gaussians: `8832675`. Peak allocated CUDA memory was about
`15.47 GB`.

Compared with R006 (`197280` roots, final test composite `32.607`, best
`32.751`), R007d is metric-equivalent and slightly denser. The important
algorithmic result is that the fixed per-event cap is no longer required:
densification naturally stops after the evidence falls below threshold.

Rejected/paused variants:

- bare uncapped with inherited `2.5e-5`: rejected, because it selected
  roughly `5.5k-6k` parents per event and behaved like near-global growth.
- `0.003`: stable but too conservative, final root count was lower than R006.
- `0.0015`: stable but still too close to `0.003` in early growth.

Local visual artifacts downloaded to:

```text
D:/petsgaussianhair-accept-line/_downloads/hgc_r007d_iter30000_preview
```

Canonical Blender pure-fur preview:

```text
D:/petsgaussianhair-accept-line/_downloads/hgc_r007d_iter30000_preview/blender_pure_fur_100k_view09.png
```

Visual judgment: the pure-fur structure is usable as an R007 checkpoint but not
the final answer. Density is not worse than R006, and the body/tail coverage is
reasonable. Remaining issues are local direction discontinuity, visible
strand-level segmentation on the back/hip/tail, and some edge-region length
outliers. These should be handled in later structure/regularization work, not
by reintroducing a hard event cap.
