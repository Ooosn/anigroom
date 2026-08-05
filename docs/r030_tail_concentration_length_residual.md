# R030: Sparse Tail Concentration for Render Length Residuals

## Problem

R027 contains a small connected tail-tip patch whose render-root raw length
residual reaches `5.17`; its effective strands reach `0.318` while neighboring
guide lengths remain ordinary. The roots are opaque and correctly surface
attached, so this is neither alpha nor visualization failure. A mean-reduced
prior dilutes a fixed sparse failure as render-root densification increases the
population.

R029 proved that a fourth-moment norm removes the visible isolated strands, but
it also suppresses the main residual distribution and costs `0.245 dB` by 14k.

## Single Change

Keep the accepted mean-L1 prior and add a tail-concentration term:

```text
L = mean(abs(r)) + unlock * (L4(r) - L2(r))
```

`L4-L2` is zero for a coherent field with equal residual magnitude and positive
when residual energy is concentrated in sparse roots. `unlock` is the existing
render-geometry residual multiplier; there is no new schedule. The formulation
contains no physical length bound, percentile cutoff, body-part rule, selected
root count, or animal-specific threshold.

## Acceptance

- Reproduce R027 at 10k before geometry residuals unlock.
- Preserve normal effective-length P95 and render-root count.
- Prevent the sparse raw/effective length tail from growing toward R027.
- Improve the 11k-14k PSNR gap over rejected R029.
- Complete 30k and compare canonical pure-fur renders before acceptance.

## Verification And Result

Focused geometry-residual tests pass locally (`45/45`). The regression suite
now also checks population scaling directly: when a single nonzero residual is
embedded in 4096 roots, the accepted mean-L1 gradient is diluted by `1/N`,
while the tail-concentration gradient remains more than two orders of
magnitude larger. A coherent equal-magnitude residual field still receives
exactly the accepted mean-L1 loss.

The formal H100 run starts from the exact accepted R027 9k checkpoint with a
fresh optimizer, matching the R027 continuation protocol. It completed at 30k
without restart, fallback, OOM, or lifecycle cap. Matched results are:

| Iteration | R027 test composite | R030 test composite | Delta | R027 / R030 roots | R027 / R030 effective max | R027 / R030 raw residual max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | 29.78481 | 29.78800 | +0.00319 | 215829 / 215837 | 0.10258 / 0.10019 | 0 / 0 |
| 11k | 30.28021 | 30.27262 | -0.00759 | 227973 / 228088 | 0.11088 / 0.10730 | 1.549 / 0.236 |
| 12k | 30.35728 | 30.26454 | -0.09274 | 238900 / 238966 | 0.11960 / 0.10965 | 2.104 / 0.373 |
| 13k | 30.75217 | 30.62146 | -0.13072 | 249851 / 250076 | 0.12004 / 0.11270 | 2.513 / 0.526 |
| 14k | 30.89468 | 30.75879 | -0.13589 | 260721 / 260897 | 0.12387 / 0.11780 | 2.895 / 0.533 |
| 15k | 31.39061 | 31.27306 | -0.11755 | 271065 / 271231 | 0.11869 / 0.11493 | 3.525 / 0.586 |
| 16k | 31.32454 | 31.19526 | -0.12928 | 281192 / 281421 | 0.12359 / 0.11940 | 3.750 / 0.555 |
| 17k | 31.67669 | 31.55908 | -0.11761 | 290755 / 291140 | 0.17415 / 0.12198 | 4.173 / 0.568 |
| 18k | 31.75850 | 31.67046 | -0.08804 | 300350 / 300650 | 0.19537 / 0.12164 | 5.206 / 0.588 |
| 19k | 31.87753 | 31.83082 | -0.04671 | 308922 / 309325 | 0.26539 / 0.12273 | 5.768 / 0.520 |
| 20k | 32.07711 | 32.02435 | -0.05276 | 317850 / 318229 | 0.31967 / 0.12046 | 5.151 / 0.522 |
| 21k | 32.19175 | 32.14723 | -0.04452 | 318532 / 318963 | 0.26786 / 0.12350 | 4.487 / 0.513 |
| 22k | 32.18475 | 32.13140 | -0.05335 | 318532 / 318963 | 0.31838 / 0.12337 | 5.166 / 0.590 |
| 23k | 32.28532 | 32.23275 | -0.05257 | 318532 / 318963 | 0.38652 / 0.11658 | 5.205 / 0.482 |
| 24k | 32.36071 | 32.34557 | -0.01514 | 318532 / 318963 | 0.41621 / 0.12075 | 5.960 / 0.479 |
| 25k | 32.16598 | 32.18820 | +0.02222 | 318532 / 318963 | 0.38827 / 0.11904 | 6.175 / 0.486 |
| 26k | 32.40933 | 32.35966 | -0.04966 | 318532 / 318963 | 0.40072 / 0.12437 | 6.177 / 0.477 |
| 27k | 32.50315 | 32.46486 | -0.03829 | 318532 / 318963 | 0.50639 / 0.12082 | 6.799 / 0.490 |
| 28k | 32.27441 | 32.22398 | -0.05043 | 318532 / 318963 | 0.46702 / 0.11872 | 6.660 / 0.601 |
| 29k | 32.55174 | 32.53251 | -0.01923 | 318532 / 318963 | 0.34725 / 0.11836 | 5.696 / 0.528 |
| 30k | 32.38110 | 32.33575 | -0.04535 | 318532 / 318963 | 0.33016 / 0.11907 | 5.104 / 0.548 |

At 30k, the distribution proves this is not global shortening:

| Statistic | R027 | R030 | Change |
| --- | ---: | ---: | ---: |
| effective mean | 0.019948 | 0.020719 | +3.87% |
| effective P95 | 0.036316 | 0.037552 | +3.40% |
| effective P99.9 | 0.084822 | 0.070465 | -16.93% |
| effective max | 0.330164 | 0.119073 | -63.94% |
| raw residual P99.9 | 1.073390 | 0.323215 | -69.89% |
| raw residual max | 5.103891 | 0.548016 | -89.26% |

The final train/test composite PSNR is `33.15194 / 32.33575`; the best test
composite is `32.53251` at 29k. This is `-0.04535 dB` final and `-0.01923 dB`
best versus R027. Final capacity is `318963` render roots and `13905203`
Gaussians, only `+0.135%` roots versus R027. Peak PyTorch allocated memory is
`24262.64 MB` under the 30 GB guard.

Opacity and attachment diagnostics reject an alpha or root-placement cause:
the failed R027 tail roots are surface attached and have opacity near `0.98`.
R030 removes their raw length escape rather than hiding them with transparency.
The remaining R030 maximum is a locally supported guide trend plus a moderate
render residual; its max/P99.9 ratio is `1.690`, versus `3.892` in R027.

Fixed-protocol 1920x1080 pure-fur renders at 22k, 27k, and 30k show the R027
tail-tip spikes are absent. Opposite-side and front views show no transferred
long-hair artifact. Full-resolution RGB remains visually matched apart from
the removed tail-tip strands.

## Decision

R030 is accepted as the Stage 1 continuation baseline. It resolves the sparse
long-hair failure without an absolute length cap, animal/body-region rule,
percentile clamp, selected-root budget, or new schedule. R029's full fourth-
moment replacement remains rejected because it suppresses the ordinary
residual field and loses substantially more PSNR.

## Method-Facing Formulation

The paper-facing method point is **hierarchy-aware residual concentration
regularization**, not the invention of an Lp norm and not a local smoothness
loss. For render-root raw length residuals `r` and the existing residual unlock
`u(t)`, define

```text
M_p(r) = (mean_i |r_i|^p)^(1/p)
L_render-length = mean_i |r_i| + u(t) * (M_4(r) - M_2(r)).
```

This formulation has four useful properties:

1. By the power-mean inequality, `M_4 >= M_2`, so the added term is
   non-negative.
2. If all residual magnitudes are equal, `M_4-M_2=0`; a coherent field keeps
   exactly the accepted mean-L1 prior.
3. For one fixed nonzero residual among `N` roots, mean L1 scales as `N^-1`,
   while the leading concentration response scales as `N^-1/4`. Densification
   therefore cannot silently dilute a sparse failure at the same rate.
4. The term acts only on render-root detail. Coherent long-hair structure
   remains owned by the interpolated guide field instead of being prohibited by
   an absolute physical bound.

The general use of norm relations and fourth moments as sparsity/concentration
measures is established prior mathematics. The contribution candidate is their
specific use to enforce guide/render ownership under dynamic root
densification, together with the unlock-coupled formulation and the measured
failure analysis. Paper claims must preserve this boundary.

Required ablation story:

- R027 mean L1: reconstruction is strong, but its sparse gradient is diluted
  and tail-tip residuals escape.
- R029 fourth-moment replacement: controls the tail but over-regularizes the
  ordinary residual distribution.
- R030 mean L1 plus `M4-M2`: controls only concentration, preserves the normal
  coat distribution, and retains reconstruction quality.

Related mathematical precedents:

- Hoyer, *Non-negative Matrix Factorization with Sparseness Constraints*, JMLR
  2004: norm relations as a sparseness measure.
- Mounir et al., *Guitar Note Onset Detection Based on a Spectral Sparsity
  Measure*, EUSIPCO 2016: L2/L4 relations as an energy-concentration measure.

Formal artifacts:

- HGC output: `/home/wangyy/anigroom-r028-early-guide-20260731/outputs/r030_tail_concentration_9k_30k_20260801_h100`
- Local checkpoint: `D:/RTS/_tmp/r030_full_visuals/checkpoint/00_checkpoint_030000.pt`
- Checkpoint SHA-256: `2ed7640af0c1954ac447d25f97789bc2eac167ce680e815d651d43526cb5ffba`
- Local statistics: `D:/RTS/_tmp/r030_full_visuals/length_analysis.json`
- Canonical final asset: `D:/RTS/_tmp/r030_full_visuals/candidate_030000_asset_side_y_v11_protocol.png`
- Opposite-side asset: `D:/RTS/_tmp/r030_full_visuals/candidate_030000_asset_side_y_pos_v11_protocol.png`
- Front asset: `D:/RTS/_tmp/r030_full_visuals/candidate_030000_asset_front_z_v11_protocol.png`
