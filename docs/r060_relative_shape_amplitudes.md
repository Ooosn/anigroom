# R060: Length-Relative Curl And Frizz Amplitudes

Status date: 2026-08-14.

Status: completed and accepted as the current advanced-geometry baseline.
R059 remains frozen as the matched absolute-amplitude control.

## Question

R059 stores curl radius and frizz amplitude in absolute mesh units. Its final
100k-strand audit has only 34 strict foldbacks, but all 34 occupy one compact
head-crown patch. The component ablation attributes 29 of them to primary curl
alone, and guide 140 has:

- length: `0.010079`
- curl radius: `0.011989`
- curl-radius / length: `1.1895`

The same learned absolute radius is therefore disproportionately strong on a
short strand. R060 asks whether these amplitudes should instead be intrinsic
strand-shape ratios.

## Single Method Change

For strand length `L`, R060 optimizes two positive dimensionless controls:

```text
curl_radius_ratio       = rho
frizz_amplitude_ratio   = alpha
```

Physical amplitudes exist only at final strand construction:

```text
curl_radius       = L * rho
frizz_amplitude   = L * alpha
```

Curl turns remains dimensionless and unchanged. The curl/frizz deformation
formula from R058/R059 is unchanged after this conversion.

This semantic change is applied through the complete hierarchy:

- primary guide roots own positive, unbounded `rho` and `alpha`;
- primary-guide interpolation operates directly on those ratios;
- secondary-guide and render-root detail are zero-centered log-ratio residuals;
- guide, secondary, render, and effective smoothness operate in ratio space;
- guide and render lifecycle children inherit interpolated ratio coordinates;
- diagnostics report both ratios and derived physical amplitudes;
- checkpoint schema is raised from `6` to `7` and rejects R059 absolute fields.

There is no absolute curl/frizz endpoint, animal-part rule, percentile clamp,
or compatibility migration. R060 must train from zero.

## Preserved Contract

R060 inherits R059 without a configuration assignment. The following are
unchanged:

- data, cameras, mesh alignment, and clean-flow v4 initialization;
- primary/secondary/render root populations and interpolation support;
- RGB, RGB-derived flow, anchor, smoothness, and appearance losses;
- all unlock schedules and learning rates;
- render and guide lifecycle evidence, placement, pruning, and timing;
- Gaussian RGB residual and gradient ownership;
- full-resolution evaluation and fixed visualization protocols.

Initialization preserves R059's near-neutral physical seed: after clean-flow
sets final guide lengths, the accepted root-width scale is divided by each
guide length to initialize the two ratios. Training thereafter is entirely in
ratio coordinates.

## Local Verification

The full local suite passes:

```text
140 passed
```

Focused coverage includes strict rejection of R059 state, zero-residual
invariance, guide/secondary/render gradient ownership, lifecycle transport,
clean-flow-aware initialization, finite gradients, and scale equivariance.

The canonical scale test uses lengths `0.012`, `0.024`, and `0.048` with
identical `rho=0.16` and `alpha=0.055`. Physical amplitudes scale linearly and
the maximum difference between normalized strand shapes is
`8.215650382226158e-15`.

Artifacts:

- `D:/RTS/_tmp/r060_relative_shape_scale_20260813/relative_shape_scale_report.json`
- `D:/RTS/_tmp/r060_relative_shape_scale_20260813/relative_shape_scale.npz`
- `D:/RTS/_tmp/r060_relative_shape_scale_20260813/relative_shape_scale_blender.png`

## Formal H100 Result

The reviewed commit `88c953988c30b68bb9cdc09ed082c5b5b4b0577c` completed
one uninterrupted, strict from-zero 0-30k H100 run. It used the unchanged
1920x1080 data/evaluation contract, 400k initial render roots, 4500 primary
guides, 20k secondary guides, and 30 training views. There was no resume,
fallback, reduced resolution, or compatibility migration.

| Metric | R059 absolute amplitudes | R060 relative amplitudes | Delta |
| --- | ---: | ---: | ---: |
| final train composite | 33.38828 | 33.37045 | -0.01784 dB |
| final test composite | 32.25537 | 32.23912 | -0.01625 dB |
| best test composite | 32.33647 | 32.32348 | -0.01299 dB |
| fixed eight-view composite mean | 33.32501 | 33.29358 | -0.03143 dB |
| final render roots | 474054 | 473867 | -187 |
| final generated Gaussians | 5535197 | 5486787 | -48410 |
| elapsed time | 13351.595 s | 13593.378 s | +241.783 s |
| peak CUDA allocation | 21938.24 MB | 16698.83 MB | -5239.41 MB |

The two routes remain matched before optional shape activates. At iteration
20k their test-composite difference is only `-0.00048 dB`, so the result is
not caused by early capacity, lifecycle, or initialization drift. All 85
lifecycle events finish by 9k, as in R059.

## Structural Result

The canonical audit uses the same 100k roots, 32 samples, and seed 29 as R050
and R059.

| Metric | R050 | R059 | R060 |
| --- | ---: | ---: | ---: |
| strict foldback strands | 0 | 34 | 0 |
| local relative-length mean | 0.02047 | 0.02125 | 0.02062 |
| local relative-length P95 | 0.07741 | 0.07919 | 0.07726 |
| local direction difference mean | 3.828 deg | 3.946 deg | 3.916 deg |
| local direction difference P95 | 11.296 deg | 11.475 deg | 11.378 deg |
| arc/chord P95 | 1.00673 | 1.06189 | 1.03476 |
| arc/chord P99 | 1.02534 | 1.14865 | 1.08576 |
| maximum local-turn P95 | 0.955 deg | 10.265 deg | 8.853 deg |
| maximum local turn | 3.188 deg | 56.163 deg | 41.063 deg |

R060 has zero strict foldbacks at 20k, 22k, 25k, 27k, and 30k. The matched
R059 timeline has 14 at 25k, 25 at 27k, and 34 at 30k. Curl-only,
frizz-only, primary-only, and secondary-disabled R060 component audits also
remain at zero foldbacks.

This is not a disabled-shape result. At 30k the effective curl-radius ratio
has mean/P95/max `0.01769/0.05822/0.16837`, while effective frizz ratio has
mean/P95/max `0.00621/0.01769/0.11901`. Their derived physical amplitudes vary
with strand length, which removes the R059 failure where one short crown hair
received a curl radius larger than its own length.

## Appearance Handoff

The fixed eight-view Gaussian RGB residual gain is `+1.60593 dB`, compared
with `+1.60937 dB` in R059. Residual RMS is `0.07857`, saturation is `1.84%`,
and mean shape-detail image magnitude remains nonzero at `0.00165`. Therefore
the appearance outlet remains active and optional shape has not collapsed.

Canonical side, opposite-side, and front assets show no new body spike,
tail hook, leg foldback, or head-crown spiral. Some top/front regions still
show direction-field convergence and crossing bands. That is retained as a
clean-flow/direction-field or asset-postprocess issue; R060 does not claim to
solve it through curl/frizz amplitude semantics.

## Decision And Frozen Evidence

R060 is accepted because it removes the short-strand scale failure and all 34
R059 strict foldbacks while preserving reconstruction, appearance handoff,
lifecycle behavior, and local continuity. R059 is retained only as the
absolute-amplitude comparison. R050 remains the near-straight appearance
reference, and R043 remains the structural/lifecycle base.

Formal evidence:

- server output: `/home/wangyy/anigroom-r060-relative-shape-runtime-20260813/outputs/r060_relative_shape_amplitudes_0_30k_h100_20260813`
- local postprocess: `D:/RTS/_tmp/r060_h100_postprocess_20260814/postprocess/r060_relative_shape_amplitudes`
- foldback timeline: `D:/RTS/_tmp/r060_h100_postprocess_20260814/diagnostics/foldback_timeline_20260814`
- canonical cross-run audit: `D:/RTS/_tmp/r060_h100_postprocess_20260814/postprocess/r050_r059_r060_strand_audit_canonical.json`
- canonical side asset: `D:/RTS/_tmp/r060_h100_postprocess_20260814/postprocess/r060_relative_shape_amplitudes/assets/r060_030000_asset_side_y_v11_protocol.png`

Frozen SHA256 values:

- checkpoint: `5300eabe5495f6f6fd254ad1911b874b80c0b4a8ae01f98ba90ad6ddb40f7060`
- configuration: `d76386759e59b1f288a223306f69c7053b9a4ebcfa2d6218c5765ed6ad6ad3b4`
- render report: `b6d85bfbc4e6e2d6b2863fc21efdf58606e18ea8d81db777adafce77566b2b6d`
- strand export: `23c7cbbbedb2c2cea5e8f2ef0c877b97750e2eeca4bd08d326e5148e6e415140`
- canonical cross-run audit: `5c6b30022f50fe2d96a62753e02eabd4228b198bae3cf59202b0e49b9608a6f0`
