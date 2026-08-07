# R049 Secondary-Guide 30k Validation

## Status

Completed and accepted as the structural parent for the next appearance
experiment. R043 remains the frozen RGB metric control. R048 established the
corrected differentiable vector transport and the matched 16k structure
comparison; R049 changed no algorithm setting and only continued that exact
state to the formal 30k endpoint.

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

The one-H100 continuation completed at 2026-08-08 05:13 JST without fallback,
OOM, root-topology drift, or configuration change.

```text
source:     /home/wangyy/anigroom-r049-secondary-guide-30k-20260808
runtime:    /home/wangyy/anigroom-r049-secondary-guide-30k-runtime-20260808
output:     /home/wangyy/anigroom-r049-secondary-guide-30k-runtime-20260808/outputs/r049_secondary_guide_resume16k_30k_h100_20260808
checkpoint: /home/wangyy/anigroom-r049-secondary-guide-30k-runtime-20260808/outputs/r049_secondary_guide_resume16k_30k_h100_20260808/checkpoint_030000.pt
```

| Measurement | R049 30k |
| --- | ---: |
| 16k-to-30k elapsed time | 3784.51 s |
| final train composite PSNR | 32.262772 |
| final test composite PSNR | 31.603195 |
| best observed test composite PSNR | 31.741003 at 29k |
| render roots | 469,402 |
| generated Gaussians | 5,323,659 |
| peak allocated CUDA memory | 10,319.53 MB |
| effective length P95 / maximum | 0.044195 / 0.123368 |

R043 remains about 1.1 dB higher on the final test metric. That difference is
expected evidence for the method question rather than a promotion failure:
R043 lets 469k independent render-root geometry residuals absorb high-frequency
RGB evidence, while R049 constrains that geometry through the 20k secondary
field.

## Fixed Structural QA

R043 and R049 were exported with the same fixed protocol: 100,000 strands,
32 samples per strand, one child, one uniform material, and identical Blender
5.0 cameras, mesh, width, 1920x1080 resolution, and 96 render samples. The
independent R049 images are:

```text
D:\RTS\_tmp\r049_30k_final\r049_030000_asset_side_y_v11_protocol.png
D:\RTS\_tmp\r049_30k_final\r049_030000_asset_side_y_pos_v11_protocol.png
D:\RTS\_tmp\r049_30k_final\r049_030000_asset_front_z_v11_protocol.png
```

All three views retain a continuous coat without isolated long roots,
backward segments, curl-back, or visible crossing clusters. The matched numeric
audit reports:

| Statistic | R043 | R049 |
| --- | ---: | ---: |
| local 4NN relative length difference mean | 0.103637 | 0.021087 |
| local 4NN relative length difference P95 | 0.315815 | 0.079554 |
| local chord-direction difference P95 | 11.3489 deg | 11.3911 deg |
| arc/chord ratio P95 | 1.005887 | 1.003271 |
| maximum local turn P95 | 0.869957 deg | 0.627962 deg |
| maximum local turn maximum | 2.733589 deg | 2.062634 deg |
| maximum arc length | 0.131546 | 0.119080 |
| strands longer than 0.12 | 6 | 0 |
| strands with a backward segment | 0 | 0 |

R049 reduces local relative length discontinuity by about 4.9x in the mean and
4.0x at P95 while leaving the local direction trend effectively unchanged. It
also improves the tortuosity and long-tail statistics. This confirms that the
secondary field removes per-strand geometric texture fitting rather than
merely reducing capacity everywhere.

## Decision

Lock R049 as the geometry parent for R050. Do not claim it as an RGB metric
gain and do not increase the secondary-guide count. R050 must add only a true
Gaussian-sample RGB residual and must preserve R049's geometry, lifecycle,
losses, interpolation, fixed structural QA protocol, and one-H100 execution.
The next acceptance gate is dual: recover RGB fidelity while retaining the
R049 structural statistics and visual continuity.
