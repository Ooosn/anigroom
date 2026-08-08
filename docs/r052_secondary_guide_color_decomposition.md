# R052 Secondary-Guide Color Decomposition

## Status

Implementation and validation in progress. R050 remains the accepted line
until this document records a completed 30k run and fixed visual audit.

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
