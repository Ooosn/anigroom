# R060: Length-Relative Curl And Frizz Amplitudes

Status date: 2026-08-13.

Status: implemented and locally verified; formal from-zero H100 training is
pending. R059 remains frozen until the complete R060 run and structural QA are
accepted.

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

## Formal Acceptance Gate

The formal route is:

1. clean checkout at the reviewed R060 commit;
2. full-resolution active-path forward/backward preflight;
3. strict from-zero 0-30k H100 training;
4. eight fixed RGB views, canonical 100k-strand export, attribute diagnostics,
   and the same foldback component audit used for R059.

Acceptance requires no reconstruction collapse and a structural improvement
in the R059 head-crown patch. In particular, the decision will use strict
foldbacks, component attribution, normalized curl/frizz tails, local turn, and
canonical assets together. PSNR alone cannot accept or reject the route.
