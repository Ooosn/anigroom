# R061: Gaussian-Only High-Frequency Appearance

Status date: 2026-08-14.

Status: completed and accepted as the current advanced-geometry/appearance
baseline. The strict from-zero H100 run, fixed postprocess, checkpoint reload,
and canonical strand audit all pass.

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
or schedule is a later single-variable experiment; R061 does not tune it while
removing the competing outlet.

## Formal Result

The uninterrupted 0-30k H100 run completes without restart, fallback, OOM, or
lifecycle instability:

- final train/test composite PSNR: `33.27953 / 32.21457`;
- best test composite PSNR: `32.30076` at 29k;
- final render roots/training-metric Gaussians: `471749 / 5484109`;
- peak allocated CUDA memory: `16312.76 MB`.

Relative to R060, final and best test composite change by only
`-0.02455/-0.02272 dB`. Root and Gaussian populations change by `-0.45%` and
`-0.05%`, respectively, so removing the old color outlet did not alter the
render-root lifecycle.

## Appearance Decomposition

Under the fixed eight-view, full-resolution protocol:

- full composite mean: `33.23565 dB` versus R060 `33.29358 dB`;
- mean without Gaussian RGB residual: `31.41477 dB`;
- Gaussian RGB residual contribution: `+1.82087 dB` versus R060 `+1.60593 dB`;
- Gaussian residual parameter RMS: `0.08131` versus R060 `0.07857`;
- Gaussian residual saturation: `2.10%` versus R060 `1.84%`.

The Gaussian residual recovers the retired outlet's high-frequency evidence
without approaching saturation. Smooth root/tip color remains the only strand
color field, and the pure-fur export excludes the Gaussian residual. This is
the intended semantic split.

## Structure Audit

The matched 100k-strand, 32-sample audit reports zero backward segments and no
arc length above `0.12` in both R060 and R061. R061 changes local relative-length
mean/P95 from `0.02062/0.07726` to `0.02160/0.08033`, local direction P95 from
`11.378` to `11.526` degrees, and maximum-turn P95 from `8.853` to `9.408`
degrees. The fixed side, opposite-side, and front assets show no new foldback,
spiral, tail spike, or local collapse. The small numeric tail change is not a
material geometry regression.

## Decision And Frozen Evidence

R061 is accepted. It removes a semantically redundant per-render-root RGB
outlet, preserves R060 reconstruction and structure, and makes the true
Gaussian-level residual responsible for high-frequency appearance.

Formal evidence:

- server output: `/home/wangyy/anigroom-r061-gaussian-only-appearance-runtime-20260814/outputs/r061_gaussian_only_appearance_0_30k_h100_20260814`;
- local postprocess: `D:/RTS/_tmp/r061_acceptance_20260814/postprocess/r061_gaussian_only_appearance`;
- canonical audit: `D:/RTS/_tmp/r061_acceptance_20260814/r060_r061_strand_audit_canonical.json`;
- canonical side asset: `D:/RTS/_tmp/r061_acceptance_20260814/postprocess/r061_gaussian_only_appearance/assets/r061_030000_asset_side_y_v11_protocol.png`.

Frozen SHA256 values:

- checkpoint: `c90052175aa1d1b1a8cfe79fe52ae0e4fb9c9dd2fe8bf76472e2e572f993d538`;
- configuration: `10a2462737967477deaf440d312b394f3e97518cde2d2a98d08d1e0913d9a065`;
- render report: `b8d0e30d8ce56f28fe60c10224dd641f80374fb50ca90d40e549bdfc9d861e88`;
- strand export: `9c6b9262ff72a952a7ff25d6ecd39b87285aec17b2973a6d6df2710768e2c9ec`;
- canonical audit: `1ddfbcb324d499f6ca2a0ad9990746f7fbb746c07ad089c860d586a6903c164`.
