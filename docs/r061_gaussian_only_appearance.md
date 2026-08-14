# R061: Gaussian-Only High-Frequency Appearance

Status date: 2026-08-14.

Status: prepared as a strict single-variable child of accepted R060. Formal
from-zero H100 validation is pending.

## Question

The active route has one independent strand per render root
(`child_count=1`). The historical `child_color_delta_raw` therefore no longer
represents variation among child strands. It is one unconstrained RGB offset
per render root, applied before the generated-Gaussian RGB residual.

This conflicts with the intended appearance decomposition:

- smooth root/tip color carries the low-frequency strand color;
- the true Gaussian-level RGB residual carries high-frequency image effects;
- pure-fur asset export omits the Gaussian residual.

R061 asks whether the Gaussian-level residual can take over the high-frequency
evidence when the obsolete render-root RGB outlet is absent, while preserving
R060 strand geometry.

## Existing Evidence

The accepted R060 fixed eight-view postprocess reports:

- full composite mean: `33.29358 dB`;
- mean after zeroing only local render-root color: `28.25906 dB`;
- local render-root color ablation drop: `5.03452 dB`;
- Gaussian RGB residual ablation drop: `1.60593 dB`;
- local render-root color RMS: `0.12947`;
- Gaussian RGB residual RMS: `0.07857`.

The old outlet currently dominates appearance. A same-checkpoint ablation is
not the R061 result because the Gaussian residual never had to learn without
that competitor. R061 must train from zero.

## Single Method Change

R061 sets:

```text
LOCAL_CHILD_COLOR_SUPPORT=0
```

This means the model contains no `child_color_delta_raw` parameter. It does not
set a learned field to zero after training and does not add a replacement.

The following remain byte-for-byte inherited from R060:

- clean-flow v4 initialization and all 3D surface interpolation;
- primary, secondary, and render-root geometry;
- curl/frizz ratio representation and schedules;
- all smoothness, anchor, RGB, RGB-flow, and backing losses;
- Gaussian RGB residual representation, scale, controls, and unlock schedule;
- render-root lifecycle, density, cameras, resolution, and evaluation.

## Acceptance Gates

The formal decision requires one uninterrupted 0-30k H100 run and the fixed
R060 protocols:

1. strict checkpoint reload with no local render-root color parameter;
2. full-resolution train/test composite metrics and eight fixed RGB views;
3. Gaussian residual on/off decomposition and residual statistics;
4. canonical 100k-strand assets and the fixed strand-structure audit;
5. memory, lifecycle, optimizer-state, and root/Gaussian population checks.

R061 is not required to beat R060 PSNR. It is accepted only if the appearance
decomposition becomes semantically correct without causing a material geometry
regression. If Gaussian residual capacity is insufficient, its representation
or schedule is a later single-variable experiment; R061 will not tune it while
removing the competing outlet.
