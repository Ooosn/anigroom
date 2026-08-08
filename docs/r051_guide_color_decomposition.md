# R051 Guide-Color / Gaussian-Residual Decomposition

## Status

Implementation verified locally. Formal from-zero H100 training is pending.
R050 remains immutable as the numerical and structural diagnostic parent; R051
is not accepted until the complete 30k and fixed visual protocol pass.

## Diagnosis From R050

R050 established that its Gaussian RGB residual is useful, but its initial
interpretation was too strong. In the final checkpoint, disabling only that
residual makes the image visibly noisier and reduces view09 composite PSNR from
`32.539284` to `29.819038`. The residual is therefore removing error rather
than adding noise.

The complete same-checkpoint layer ablation exposes the actual failure:

| Variant | Mean eight-view PSNR | View09 PSNR |
| --- | ---: | ---: |
| full R050 | 33.1913 | 32.5393 |
| Gaussian residual off | 31.0777 | 29.8190 |
| local render color off | 28.3788 | 27.1601 |
| root/tip only | 27.1539 | 25.8152 |

The old local render color has decoded absolute mean `0.12466` and RMS
`0.12960` under a `0.15` scale, so it is globally near its representational
limit. Frequency-separated error confirms that both appearance layers repair
low-frequency as well as high-frequency error:

| Variant | Total RMSE | Low-frequency RMSE | High-frequency RMSE |
| --- | ---: | ---: | ---: |
| full | 0.04879 | 0.03002 | 0.03238 |
| Gaussian residual off | 0.06476 | 0.04350 | 0.03776 |
| local render color off | 0.09268 | 0.07799 | 0.03761 |
| root/tip only | 0.10742 | 0.09222 | 0.04022 |

R050 therefore learned two entangled image-fitting fields whose errors partly
cancel. It is retained as evidence that Gaussian-level residuals work, but is
rejected as evidence of a clean main-color/noise decomposition.

## Single Representation Change

R051 changes color ownership while keeping R050 geometry, lifecycle, losses,
resolution, and schedules unchanged:

```text
low-frequency fur color
  = sparse primary-guide root/tip colors
  -> topology-safe surface interpolation
  -> render roots
  -> strand samples

final Gaussian RGB
  = low-frequency fur color
  + scheduled Gaussian RGB residual(sample position)
```

The previous trainable render-root root/tip colors and local child-color field
are absent from the optimizer. The sparse guide color field is smooth by
construction and by the existing guide surface-graph regularizer. Unobserved
guide colors are initialized by physical surface-graph harmonic inpainting,
not ambient 3D KNN.

## Handoff Contract

- Iterations 0-10k: Gaussian RGB residual is exactly inactive. Guide root/tip
  colors learn the main fur color.
- After 10k: guide color gradients are set to `None`, so Adam momentum cannot
  move the frozen base. The Gaussian residual uses the existing 10k-to-20k
  ramp and becomes the only trainable color outlet.
- No extra sample-specific schedule, frequency threshold, color mask, or
  residual smoothness is introduced.

Guide lifecycle insertion interpolates decoded guide colors and observation
confidence through the same topology-safe surface support as the other guide
attributes. Surviving Adam rows are migrated exactly and new rows start with
zero optimizer state.

## Acceptance Gate

1. Full-resolution one-batch H100 forward/backward and strict checkpoint reload
   pass without fallback.
2. The from-zero 30k run preserves R050 geometry, lifecycle, memory, and pure
   fur structure.
3. Full composite PSNR remains competitive with R050.
4. The residual-off render is a visibly clean low-frequency tiger appearance,
   not the noisy image seen in the R050 ablation.
5. The Gaussian residual image contains remaining stripe detail, fur shadow,
   and high-frequency effects rather than canceling corruption in the base.
6. Eight fixed RGB views, residual-off views, residual images, three canonical
   assets, and the numeric strand audit are all generated before acceptance.

## Verification Before Formal Run

- full local suite: `114 passed`;
- focused guide-color and lifecycle tests pass;
- Python compilation, launcher syntax, and `git diff --check` pass;
- a regression test verifies that the handoff sets guide gradients to `None`
  while preserving Gaussian-residual gradients.
