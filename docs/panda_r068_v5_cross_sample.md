# Panda R068 + V5 Cross-Sample Checkpoint

Status: formal from-zero 30k training and fixed postprocess completed. This is
accepted as cross-sample evidence and a reproducible checkpoint, but not as a
generalized structure baseline because the learned primary curl is too strong.

## Contract

The run preserves the accepted Tiger R068 behavior and changes only
sample-specific inputs:

- source commit: `58bba7b7ea66745cf79346aa8e7046b08b9ea3a5`
- behavior config: `configs/r068_no_crossing_zero_curl_0_30k.env`
- resolution / iterations: `1920x1080`, from-zero `0..30000`
- initial render / primary guide / secondary guide roots:
  `400000 / 4500 / 20000`
- child count: `1`
- render lifecycle: every 100 iterations from 600 through 9000
- Gaussian RGB residual: enabled, ramp `20000..25000`
- primary curl: enabled, ramp `20000..25000`
- secondary shape residual: ramp `25000..30000`
- frizz: absent
- crossing training: disabled
- mesh no-penetration: enabled

Panda substitutions:

- mesh scale / translation: `1.0 / [0, 0, 0]`
- mesh SHA-256:
  `20bd4d3cd2c48c886e2df96d8f183e75e5bacb1f6ebe0bba3c392677550d6c20`
- V5 flow SHA-256:
  `0c4705fbab50e4d9ed86aae2376ac71977f4bafff46f68ed643de25eaa333455`
- Panda SDF SHA-256:
  `a8ddedc9cd4bea81d9cda83610f57dcf0c30b3e6cabf8554360f8936fad9b7ab`
- Panda SDF grid: `[520, 317, 196]`, outside-positive, validation sign
  agreement `0.99987793`

No Panda body-region rule, schedule change, capacity cap, or fallback was used.

## Execution Fixes

Two execution defects were found before iteration one and repaired without
changing R068 behavior:

1. The transferred Git bundle omitted the historical `stage1-r036`,
   `stage1-r042`, and `stage1-r043` tags required by baseline-lock tests. The
   original tag objects were restored; HGC then passed all `295` tests.
2. The trainer loaded the default white-tiger alignment JSON after parsing and
   restored only explicit paths, so explicit Panda scale/translation could be
   overwritten. Explicit mesh alignment now has the same precedence as
   explicit data/mesh/output paths. Focused and full regression tests pass.

The first failed launch never entered training and contained only config-lock
files. It was reset through an exact content whitelist before the formal run.

## Training Result

The uninterrupted H100 run completed in `14854.3 s`.

| Iteration | Train composite | Test composite | Render roots | Gaussians |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 14.776 | 15.043 | 400000 | 4311585 |
| 3000 | 18.958 | 19.056 | 486854 | 5246044 |
| 6000 | 20.323 | 20.322 | 577372 | 6216637 |
| 9000 | 21.249 | 21.210 | 657696 | 7079921 |
| 10000 | 27.193 | 26.939 | 660391 | 7232264 |
| 20000 | 28.393 | 27.928 | 660391 | 7226186 |
| 25000 | 29.106 | 28.340 | 660391 | 7339873 |
| 30000 | 29.613 | 28.773 | 660391 | 7780665 |

- peak PyTorch allocated memory: `22261.17 MB`
- maximum observed process GPU usage: `30031 MiB`
- final checkpoint SHA-256:
  `0ef74b44ce17a87fd5db7a085808725704cb58e5b7c07145c7e8f16e96c3f5fc`
- final checkpoint:
  `/home/wangyy/panda-r068-v5-runtime-20260827/outputs/panda_r068_v5_0_30k_h100_20260827/checkpoint_030000.pt`

The complete schema-9, no-frizz, crossing-disabled, finite-metric postvalidation
passes.

## RGB And Appearance

The fixed eight-view full-resolution mean composite PSNR is `29.42054`.
Gaussian RGB residual contributes `+1.25195 dB` on average and is active on
`70.53%` of roots, with `1.68%` saturation. It therefore transfers as intended:
it absorbs substantial high-frequency appearance rather than remaining inert.

Shape detail contributes another `+1.36033 dB` on average, but its structural
cost is not acceptable as a generic Panda groom prior.

## Structure

The matched 100k-strand audit reports:

- backward strands: `0`
- full foldbacks: `0`
- arc length P50/P95/max: `0.02087 / 0.03584 / 0.07819`
- lengths above `0.12`: `0`
- local relative-length difference mean/P95: `0.02029 / 0.08200`
- local chord-direction difference mean/P95: `4.67 / 14.22` degrees
- maximum local-turn P50/P95/max: `0.84 / 9.18 / 12.28` degrees

The backbone asset is coherent and free of long strands or foldbacks. The final
asset has excessive high-frequency waves on the shoulder and legs. Component
isolation attributes this to the primary-guide curl, not V5 direction, length,
collision, or secondary residual:

| Curl statistic | Tiger R068 | Panda R068 V5 |
| --- | ---: | ---: |
| radius ratio mean | 0.02127 | 0.10409 |
| radius ratio P95 | 0.07213 | 0.32719 |
| turns P50 | -0.01496 | -0.22905 |
| turns P95 | 0.11430 | 0.55460 |
| curl turn-excess P95 | 5.23 deg | 170.31 deg |

No-penetration remains numerically shallow: penetrating point fraction is
`0.1601%`, mean dimensionless depth is `1.22e-7`, and maximum depth is
`0.00131`.

## Decision

V5 flow and the R068 execution stack transfer successfully. The latest Tiger
curl learning rule does not yet generalize: RGB/RGB-flow evidence can still
drive primary curl toward high-frequency image detail even with an active
Gaussian appearance residual.

Do not fix this by disabling curl only for Panda or by copying a Panda-specific
limit. The next method experiment must make curl ownership depend on genuine
geometric evidence while preserving the successful V5 backbone and Gaussian
appearance decomposition.

## Artifacts

- local acceptance root:
  `D:/RTS/_tmp/panda_r068_v5_acceptance_20260827/postprocess/panda_r068_v5_protocol_20260827`
- RGB views:
  `.../rgb_views`
- fixed 100k strand NPZ:
  `.../strands/r068_no_crossing_zero_curl_030000_render_child1_100k_samples32.npz`
- final assets:
  `.../assets_blender_protocol_20260827`
- backbone / primary / final curl assets:
  `.../assets_blender_components_20260827`
- postprocess manifest:
  `.../r068_postprocess_manifest.json`
