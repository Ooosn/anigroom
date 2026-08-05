# R023 Asinh Log-Length Residual

## Status

`completed as representation evidence; standalone route superseded`. The
asinh-log decoder prevented R022's exponential escape and is retained by the
current baseline. A sparse late residual tail remained, so R023 itself was not
the final accepted training line; R024-R030 subsequently repaired the prior
coordinate, optimizer lifecycle, and tail concentration behavior.

## Parent And Isolated Change

R023 inherits the complete R022 training contract: the same verified R020 9k
checkpoint, schedule, optimizer reset, lifecycle, losses, graph, and
raw-coordinate surface smoothness. The only method variable is the positive
relative-length decoder:

```text
R022: effective = guide * exp(ramp * raw)
R023: effective = guide * exp(ramp * asinh(raw))
```

The coordinate remains zero-centered, positive, scale-relative, and
unbounded. Near zero, `asinh(raw)` has unit slope, so ordinary residuals retain
R022's local optimization behavior. In the tail, the ratio grows polynomially
instead of exponentially. For the R022 failure coordinate `raw=6.343` at the
16k ramp `0.6`, the same coordinate produces a ratio of about `4.61` rather
than `44.97`.

No absolute length limit, region rule, animal-specific threshold, or fallback
is added. Lifecycle insertion interpolates the raw coordinate directly so a
new render root inherits the same residual semantics.

## Formal Gate

The calibration starts from the same verified 9k checkpoint and continues
through 18k. The long-horizon run resumes the exact 18k optimizer and RNG state
and continues through 30k. It must pass both the matched 14k comparison and the
20k fully unlocked transition.

1. No non-finite value, OOM, or sparse effective-length explosion.
2. Test composite remains competitive with R019/R021/R022 at matched steps.
3. Effective/guide tails and graph-edge jumps stay stable through 30k.
4. Canonical 100k-strand renders at 14k, 16k, 18k, and the retained late
   checkpoints contain no isolated long
   sheets, curled-back collapse, or new short/long fragmentation.
5. Flattened configuration differs from R022 only in the decoder mode and the
   longer diagnostic/save horizon.

## Measured 14k Gate

The held-H100 run uses the same verified R020 9k checkpoint and a reset
optimizer. At 14k:

| Metric | R022 raw log | R023 asinh log | Difference |
| --- | ---: | ---: | ---: |
| Test composite PSNR | 30.850235 | 30.845266 | -0.004969 dB |
| Effective length P50 | 0.018311 | 0.018357 | +0.000047 |
| Effective length P95 | 0.035946 | 0.036008 | +0.000062 |
| Effective length max | 0.176764 | 0.118001 | -33.2% |
| Raw-coordinate max | 5.069181 | 3.194534 | -37.0% |

The body of the learned distribution is unchanged within noise while the
sparse upper tail is materially smaller.

## Transition And Full-Ramp Measurements

| Iteration | Residual multiplier | Test composite PSNR | Length P50 | Length P95 | Length max | Raw max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16000 | 0.6 | 31.262106 | 0.018508 | 0.036358 | 0.115697 | 3.650732 |
| 18000 | 0.8 | 31.647703 | 0.018604 | 0.036407 | 0.179461 | 4.216400 |
| 19000 | 0.9 | 31.757362 | - | 0.036394 | 0.362874 | 5.884933 |
| 20000 | 1.0 | 31.909903 | 0.018484 | 0.036400 | 0.391355 | 4.497168 |
| 21000 | 1.0 | 31.977486 | - | 0.036421 | 0.496986 | 4.533877 |
| 22000 | 1.0 | 31.954067 | - | 0.036367 | 0.370792 | 4.454503 |

The central distribution remains fixed across the unlock transition. The
maximum is sparse and non-monotonic rather than a distribution-wide escape:
it rises at 19k-21k and falls again at 22k. This is materially different from
R022, whose 16k maximum reached `2.165497` with an effective/guide ratio of
`44.965`. It is still an open tail risk until the 30k graph diagnostic and
canonical asset render are complete.

The continuation log explicitly reports `start_iteration=18000`,
`optimizer_state_entries=24`, and restored RNG state. It is not a reset or a
new optimization trajectory.

## 18k Graph And Visual QA

At 18k, raw-coordinate P99/P99.9 are `0.45035/0.90940`; effective-length
P99/P99.9 are `0.04687/0.07373`. The raw top 0.1% contains 218 roots in 107
surface-graph components; the largest components contain 16, 15, 12, and 10
roots. The internal directed-edge fraction is `0.2603`, so the tail is a mix
of small local clusters and isolated roots rather than a whole-body shift.

Canonical QA uses child count 4, deterministic 100k strands, 32 samples per
strand, Blender 1920x1080 at 96 samples, width scale 1.65, the fixed mesh
alignment, and one full-resolution image per view. The 14k/16k/18k side views
and 18k opposite-side, top, head, rear, and fixed three-quarter views contain
no R022-style long sheet, curled-back collapse, or large fragmented region.

Local QA artifacts are under:

```text
D:/RTS/_tmp/r023_visuals
```

The formal H100 continuation is:

```text
/home/wangyy/anigroom-r023-asinh-log-20260729/outputs/r023_asinh_log_length_residual_18k_30k_20260729_h100
```
