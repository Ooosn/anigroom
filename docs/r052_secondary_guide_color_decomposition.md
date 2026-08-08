# R052 Secondary-Guide Color Decomposition

## Status

Completed and rejected as an R050 replacement. The experiment confirms that
the Gaussian RGB residual must remain, but a flat 20k secondary-guide base
color field does not produce the intended clean low-frequency appearance
layer. R050 remains the accepted appearance checkpoint.

## Hypothesis

R050 proves that a generated-Gaussian RGB residual absorbs real image evidence:
turning it off costs `0.88-2.73 dB` over eight fixed views and visibly increases
noise. The residual must stay. The unresolved problem is color ownership:
R050's render-root/local base and Gaussian residual both reconstruct broad tiger
appearance, while R051 proved that 4,500 primary guides are too sparse to carry
the structured base.

R052 places root/tip base color on the already validated 20k secondary-guide
surface field. It is roughly 23x sparser than the final render-root set but 4.4x
denser than the primary guide field, so it can preserve stripe regions without
becoming a per-strand noise channel.

## Single Variable

Relative to R050:

- keep the Gaussian RGB residual and its 10k-20k multiplier unchanged;
- keep all geometry, lifecycle, losses, resolution, and schedules unchanged;
- disable local render-root color residuals;
- interpolate root/tip base color from the existing secondary-guide support;
- initialize observed secondary colors by the existing visibility-aware
  multiview projection and reconstruct unobserved values on the topology-safe
  secondary graph;
- optimize structured base color through 10k, then freeze it with `grad=None`;
- retain Gaussian residual as the sole high-frequency color outlet.

No tiger-region mask, image-frequency threshold, residual smoothness, or new
per-sample schedule is introduced.

Configuration:

```text
configs/r052_secondary_guide_color_gaussian_residual_0_30k.env
```

## Acceptance Gates

1. Full test suite and native-resolution active-path preflight pass.
2. At 10k the structured base must retain stripe regions without R051's broad
   smearing; otherwise stop before the Gaussian residual can hide the failure.
3. At 30k report full and residual-off renders for the same eight fixed views.
4. Gaussian residual removal must again increase noise, while the structured
   base alone remains spatially coherent.
5. The fixed 100k-strand asset audit must preserve R050/R049 geometry quality.

## Formal Run

The one-H100 from-zero run and strict postprocess completed without fallback,
OOM, or runtime error:

```text
source:     /home/wangyy/anigroom-r052-secondary-guide-color-decomposition-20260808
runtime:    /home/wangyy/anigroom-r052-secondary-guide-color-decomposition-runtime-20260808
output:     /home/wangyy/anigroom-r052-secondary-guide-color-decomposition-runtime-20260808/outputs/r052_secondary_guide_color_gaussian_residual_0_30k_h100_20260808
checkpoint: /home/wangyy/anigroom-r052-secondary-guide-color-decomposition-runtime-20260808/outputs/r052_secondary_guide_color_gaussian_residual_0_30k_h100_20260808/checkpoint_030000.pt
sha256:     2961c4f560a6af7dc0baee70fc9e199d113827373ae264cc7edd24d951533b29
```

| Measurement | R050 | R052 | Delta |
| --- | ---: | ---: | ---: |
| final train composite PSNR | 33.253601 | 32.140289 | -1.113312 |
| final test composite PSNR | 32.121105 | 31.454292 | -0.666813 |
| best test composite PSNR | 32.209358 | 31.527248 | -0.682110 |
| best iteration | 29k | 29k | - |
| render roots | 469,757 | 469,066 | -691 |
| generated Gaussians | 5,358,403 | 5,367,687 | +9,284 |
| peak allocated CUDA memory | 11,327.52 MB | 10,975.50 MB | -352.02 MB |
| from-zero elapsed time | 9,243.02 s | 8,042.83 s | -1,200.19 s |

The secondary color field is frozen after 10k. Its root-color standard
deviation remains bit-identical at `0.3256891966` from 12k through 30k, so all
later appearance recovery comes from the Gaussian residual and the unchanged
geometry schedule rather than hidden base-color drift.

## Same-Checkpoint Color Ablation

The fixed renderer loads the final checkpoint once and changes only the
Gaussian residual multiplier from one to zero. Geometry, alpha, camera, mesh
backing, and evaluation remain fixed.

| View | Full PSNR | Residual-off PSNR | Direct gain |
| ---: | ---: | ---: | ---: |
| 00 | 31.683607 | 29.178619 | +2.504988 |
| 05 | 31.387550 | 27.010956 | +4.376595 |
| 09 | 31.670288 | 27.177189 | +4.493099 |
| 14 | 32.649796 | 28.243151 | +4.406645 |
| 18 | 32.882202 | 29.810272 | +3.071930 |
| 21 | 33.321682 | 28.991974 | +4.329708 |
| 27 | 31.940050 | 27.212246 | +4.727804 |
| 32 | 32.527020 | 28.183939 | +4.343081 |
| mean | 32.257774 | 28.226043 | +4.031731 |

This validates the user's visual observation: removing the Gaussian residual
makes the result noisier, not cleaner. The residual is signed, and therefore
can cancel false color as well as add missing detail. Across the fixed views,
its rendered correction opposes the base error on `81.45%` of fur pixels and
reduces masked squared error by `58.15%` on average.

The result does not validate the proposed decomposition, however. The residual
is doing almost twice as much direct reconstruction work as in R050:

| Residual statistic | R050 | R052 |
| --- | ---: | ---: |
| mean eight-view PSNR gain | +2.113587 dB | +4.031731 dB |
| decoded absolute mean | 0.061171 | 0.075970 |
| decoded RMS | 0.092599 | 0.106948 |
| active fraction | 69.081% | 74.437% |
| near-saturation fraction | 5.480% | 11.106% |

The signed residual render contains broad stripe reconstruction and large
color corrections, not only hair-scale shadow and noise. It is compensating
for an inadequate base layer.

## Noise and Frequency Audit

For each fixed view, errors are measured inside the rendered fur alpha. A
Gaussian blur with sigma four pixels separates low-frequency error from the
remaining high-frequency error. This is a diagnostic only; it is not a
training loss or a tuned acceptance threshold.

| Mean masked error | R050 base | R052 base | R050 full | R052 full |
| --- | ---: | ---: | ---: | ---: |
| total RMSE | 0.070809 | 0.096199 | 0.057349 | 0.062525 |
| low-frequency RMSE | 0.029154 | 0.049192 | 0.022578 | 0.026559 |
| high-frequency RMSE | 0.059169 | 0.071473 | 0.048593 | 0.051782 |
| chroma RMSE | 0.007174 | 0.010663 | 0.005073 | 0.006435 |

R052's residual-off base is worse than R050 in every band. The largest failure
is low-frequency error (`+68.7%`), followed by chroma error (`+48.6%`), which
matches the visible green/purple regions on the ear, back, flank, and legs.
The Gaussian residual removes much of this error, but the final R052 image
still remains worse than R050.

The cause is ownership, not missing residual capacity. Twenty thousand
independently optimized color nodes are dense enough to retain multiview color
inconsistency and local artifacts. Surface interpolation does not make their
values low-frequency by itself, and the inherited weak graph term does not
turn this field into a clean base representation before it is frozen.

## Fixed Structural QA

R050 and R052 use the same deterministic 100k-strand, 32-sample, child-one
export and the same three Blender cameras.

| Statistic | R050 | R052 |
| --- | ---: | ---: |
| local 4NN relative length difference mean | 0.020467 | 0.023077 |
| local 4NN relative length difference P95 | 0.077408 | 0.086349 |
| local chord-direction difference P95 | 11.2959 deg | 11.5289 deg |
| arc/chord ratio P95 | 1.006726 | 1.011366 |
| maximum local turn P95 | 0.954593 deg | 1.139979 deg |
| maximum local turn P99 | 2.430127 deg | 5.697701 deg |
| maximum local turn maximum | 3.187551 deg | 27.834636 deg |
| maximum arc length | 0.128519 | 0.128932 |
| strands longer than 0.12 | 5 | 9 |
| strands with a backward segment | 0 | 0 |

The canonical assets remain globally recognizable and contain no whole-body
collapse, but R052 does not preserve R050's structure quality. Color pressure
still reaches geometry: local length continuity regresses and a sparse
high-curvature tail appears despite curl and frizz remaining disabled.

Local fixed-protocol outputs:

```text
D:\RTS\_tmp\r052_30k_final\postprocess_030000\rgb_views\render_report.json
D:\RTS\_tmp\r052_30k_final\r052_030000_asset_side_y_v11_protocol.png
D:\RTS\_tmp\r052_30k_final\r052_030000_asset_side_y_pos_v11_protocol.png
D:\RTS\_tmp\r052_30k_final\r052_030000_asset_front_z_v11_protocol.png
D:\RTS\_tmp\r052_30k_final\r050_r052_color_decomposition_audit.json
D:\RTS\_tmp\r052_30k_final\r050_r052_strand_audit.json
```

## Decision

Reject R052 as an R050 replacement. Keep Gaussian-level RGB residual exactly
as an accepted component: both the same-checkpoint ablation and the signed
correction audit prove that it removes false high-frequency appearance as well
as restoring real detail.

Do not promote the current flat secondary-guide color field. Any next color
decomposition attempt must first make the base representation intrinsically
low-frequency, for example through a true multiscale color hierarchy or a
low-capacity decoder, rather than relying on node count plus a weak graph loss.
This conclusion adds no animal mask, image threshold, or residual smoothing,
and no further training is authorized by this document.
