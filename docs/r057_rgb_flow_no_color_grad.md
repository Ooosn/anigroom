# R057 RGB-Flow Gradient Ownership

## Question

Can RGB-derived flow supervise geometry without directly changing root/tip
color or generated-Gaussian RGB residual?

## Frozen Parent

R055 is immutable. R057 inherits its forward render, final-RGB flow source,
losses, weights, schedules, interpolation, lifecycle, capacity, and learning
rates exactly.

R057 does not inherit R056's residual-free flow image or its RGB-to-geometry
gradient attenuation. R056 demonstrated that removing RGB geometry evidence as
appearance residual activates causes a sharp reconstruction collapse.

## Single Change

Each iteration still uses one render and one optimizer step. Backward is routed
in two parts:

1. RGB and all non-flow regularizers backpropagate normally.
2. Weighted RGB-flow loss backpropagates only to optimized non-color
   parameters.

The excluded color family is exactly:

- root color
- tip color
- optional child color delta
- generated-Gaussian RGB residual

All optimized non-color parameters retain both their original RGB gradient and
their RGB-flow gradient. No gradient multiplier, projection, extra render,
base-fur flow source, or optimizer step is introduced.

## Acceptance

1. A focused gradient test proves geometry receives `RGB + flow`, while base
   color and Gaussian RGB residual receive RGB only.
2. One Adam step advances every active tested owner exactly once.
3. The full-resolution H100 preflight completes with the flag recorded in its
   checkpoint and finite nonzero active optimizer states.
4. The from-zero 30k run is compared against synchronized R055 metrics and the
   same fixed RGB, residual, attribute, strand, and canonical asset protocol.
5. R057 is accepted only after both reconstruction and strand structure are
   inspected. PSNR alone is not sufficient.

## Status

Accepted as the active gradient-ownership baseline on top of R055. This is a
correctness change, not a claimed curl/foldback solution.

## Formal Run

- commit: `a5245fdc3daaca3ebd6822ccb499cf9289197c98`
- configuration: `configs/r057_rgb_flow_no_color_grad_0_30k.env`
- full-resolution preflight: passed at `1920x1080`
- training: uninterrupted from zero through iteration 30000
- final checkpoint:
  `/home/wangyy/anigroom-r057-flow-gradient-ownership-runtime-20260811/outputs/r057_rgb_flow_no_color_grad_0_30k_h100_20260811/checkpoint_030000.pt`
- postprocess:
  `/home/wangyy/anigroom-r057-flow-gradient-ownership-runtime-20260811/postprocess/r057_rgb_flow_no_color_grad`
- downloaded QA:
  `D:/RTS/_tmp/r057_h100_postprocess_20260811/postprocess/r057_rgb_flow_no_color_grad`

The held qlogin node experienced a system-wide swap/page-fault storm late in
the run. The training process itself had no swap, OOM, traceback, CUDA error,
or restart and completed normally. Wall-clock time from this allocation is
therefore not used as an algorithm-speed measurement.

## Reconstruction Result

| metric | R055 | R057 | R057 - R055 |
| --- | ---: | ---: | ---: |
| final train composite PSNR | 33.40419 | 33.39819 | -0.00599 |
| final test composite PSNR | 32.25507 | 32.27096 | +0.01589 |
| best observed test composite PSNR | 32.32984 | 32.35660 | +0.02676 |
| render roots | 469771 | 473705 | +3934 |
| generated Gaussians | 5645519 | 5703294 | +57775 |

The synchronized test delta is positive at 1k, 9k, 10k, 20k, and every
recorded 25k-30k gate. There is no R056-style reconstruction collapse.

Across the same eight fixed full-resolution views, mean composite PSNR is
`33.33922` for R055 and `33.33974` for R057. The final view-09 predictions
differ by only `0.00968` normalized RGB RMSE.

## Appearance-Residual Attribution

| fixed eight-view statistic | R055 | R057 |
| --- | ---: | ---: |
| mean composite PSNR | 33.33922 | 33.33974 |
| mean PSNR without Gaussian RGB residual | 31.69895 | 31.69299 |
| mean gain from Gaussian RGB residual | +1.64027 | +1.64675 |
| Gaussian residual parameter RMS | 0.08254 | 0.08189 |
| residual saturation fraction | 0.02020 | 0.01897 |

Removing flow gradients from color does not disable the appearance path. The
Gaussian RGB residual retains the same reconstruction contribution with
slightly lower parameter RMS and saturation. This is the intended ownership:
ordinary RGB supervises color and geometry, while RGB-derived flow adds only
geometry evidence.

## Strand Audit

The matched audit uses deterministic 100k strands with 32 samples each.

| statistic | R055 | R057 |
| --- | ---: | ---: |
| local 4-NN relative-length mean | 0.02226 | 0.02204 |
| local 4-NN relative-length P95 | 0.08422 | 0.08292 |
| local chord-direction mean | 3.8607 deg | 3.8444 deg |
| local chord-direction P95 | 11.3906 deg | 11.3640 deg |
| strands with any backward segment | 159 | 177 |
| arc/chord P99 | 1.18989 | 1.20297 |
| maximum-turn P99 | 65.3899 deg | 68.6494 deg |

Average local continuity improves slightly, but the sparse extreme folded tail
is slightly worse. Canonical three-view assets show no global structure or
coverage regression. R057 is therefore accepted for gradient ownership, while
the remaining sparse foldback problem stays open and must be addressed by a
geometry mechanism rather than by reverting flow-to-color gradients.

## Curl/Frizz Ownership Ablation

The final R057 checkpoint was exported three times without retraining. All
three exports use the same deterministic 100k render-root subset, 32 samples
per strand, seed 29, brush-stiffness field, lengths, directions, and widths.
Only the contribution of the primary and secondary curl/frizz fields changes:

| export | primary curl/frizz | secondary residual | backward strands | arc/chord P99 | maximum-turn P99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| no curl/frizz | 0 | 0 | 0 | 1.01768 | 1.6988 deg |
| primary only | 1 | 0 | 146 | 1.18094 | 62.6852 deg |
| primary + secondary | 1 | 1 | 177 | 1.20297 | 68.6494 deg |

All 146 foldbacks present in the primary-only export remain present in the
full export. The secondary field adds 31 and removes none. Therefore the main
source of R057's sparse foldback tail is the absolute curl/frizz field owned by
the primary guides; the zero-centered secondary residual modestly amplifies
that tail. This is not caused by brush stiffness, which is unchanged across
the three exports, and R057 has no separate legacy bend parameter.

The effect is sparse enough that the matched full-animal Blender renders are
almost indistinguishable. The result does not justify removing curl/frizz from
the representation: genuinely curly fur still needs these degrees of freedom.
It does establish that a later structural fix must prevent unsupported sparse
activation in the primary field before constraining the secondary residual.

Local artifacts:

- audit:
  `D:/RTS/_tmp/r057_h100_postprocess_20260811/postprocess/r057_rgb_flow_no_color_grad/shape_ablation/strand_audit.json`
- matched Blender assets:
  `D:/RTS/_tmp/r057_h100_postprocess_20260811/postprocess/r057_rgb_flow_no_color_grad/shape_ablation/assets`

## Decision

R057 replaces R055 as the active staged-shape training branch because its
gradient ownership matches the method's decomposition and reconstruction is
preserved. R055 remains the exact parent/control, and R050 remains the strict
structural/appearance reference. No RGB-to-geometry attenuation or gradient
projection is accepted by this experiment.
