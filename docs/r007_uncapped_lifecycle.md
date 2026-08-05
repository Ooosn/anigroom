# R007 Uncapped Render-Root Lifecycle

Historical status: completed calibration. R007d's uncapped local-max lifecycle
and `0.00075` absolute threshold are retained by the frozen R036 baseline. R006
below is the parent at the time of this experiment, not the current baseline.

## Purpose

R006 was the baseline candidate at this point. It was better than the failed rebuilt lines and avoided the global mesh-circling failure, but its render-root densification still had a fixed per-event cap:

- selected parents per event: 1024
- inserted children per event: 2048
- deleted parents per event: 1024
- net growth per event: about 1024 roots

R007 tests whether render-root densification can be controlled by evidence itself instead of a fixed event budget.

## Formal Delta From R006

Only these phase configs change:

- `configs/r007_uncapped_lifecycle_0_9k.env`
- `configs/r007_uncapped_lifecycle_9k_30k.env`

Both inherit R006 and set:

```bash
LIFECYCLE_SCORE_MODE=raw
MAX_SPLITS_PER_EVENT=0
DENSIFY_SCORE_THRESHOLD=0.003
```

In the lifecycle code, `MAX_SPLITS_PER_EVENT=0` means `parent_budget=None`. The split count is not fixed by a per-event budget. It is determined by:

- absolute evidence threshold
- visibility threshold
- mesh-topology local-max suppression
- replace-parent split/delete

Guide-root densification is not uncapped in R007. It stays inherited from R006 so the experiment only tests render-root lifecycle behavior.

## Acceptance Checks

R007 is acceptable only if:

- early events do not explode root count uncontrollably;
- `threshold_candidate_count`, `local_max_candidate_count`, and selected parent count naturally decrease or stabilize after dense areas are filled;
- memory stays within the known safe range;
- visual structure is not worse than R006;
- composite PSNR is at least comparable to R006.

If uncapping causes repeated huge split events without evidence decreasing, that variant is rejected and the next step is not a hard count cap. The next step should be a better evidence threshold or cooldown rule that follows the same lifecycle logic.

## Run Log

### R007a Diagnostic: Bare Uncapped

Config:

```bash
MAX_SPLITS_PER_EVENT=0
DENSIFY_SCORE_THRESHOLD=2.5e-5  # inherited R006 value
```

H100 diagnostic output:

`/home/wangyy/anigroom-r006-current/outputs/r007_uncapped_diag_h100_20260722_153236`

Result:

- roots: `100000 -> 187292` by 2000 iters
- selected parents per event: about `5.5k-6.0k`
- memory stayed safe, but growth pattern was wrong

Conclusion: rejected. The inherited threshold is below the median evidence score, so almost every visible root passes the threshold and local-max suppression becomes a near-uniform surface growth rule. This does not test evidence-driven densification.

### R007b Diagnostic: Uncapped With Evidence Threshold

Config:

```bash
MAX_SPLITS_PER_EVENT=0
DENSIFY_SCORE_THRESHOLD=0.003
```

H100 diagnostic output:

`/home/wangyy/anigroom-r006-current/outputs/r007b_uncapped_thr003_diag_h100_20260722_155755`

Result:

- roots: `100000 -> 117283` by 2000 iters
- selected parents per event: roughly `0.8k-1.3k`
- gaussians: `4.00M -> 4.64M`
- peak allocated memory: `8151.89 MB`

Conclusion: accepted for full R007 training. This keeps the fixed split budget removed, but avoids global uniform growth by using a meaningful absolute evidence threshold.
