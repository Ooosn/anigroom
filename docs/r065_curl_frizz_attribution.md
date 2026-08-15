# R065 Curl/Frizz Geometry Attribution

Date: 2026-08-16

## Scope

This is a read-only decomposition of the accepted R065 30k checkpoint. It
answers one question only: whether the visibly curled and periodically wavy fur
comes from the accepted brush backbone, curl, frizz, or the secondary local
shape residual.

This is not a crossing experiment. No crossing loss, lifecycle rule, training
schedule, checkpoint tensor, or baseline implementation was changed.

## Fixed protocol

- checkpoint:
  `/home/wangyy/anigroom-r065-local-crossing-residual-runtime-20260815/outputs/r065_local_crossing_residual_0_30k_h100_20260815/checkpoint_030000.pt`;
- checkpoint SHA-256:
  `2e498e17034e11c9f7061989be28903a6cbb6c3c6de4d5a217d260c5d2670c75`;
- deterministic subset: 100,000 of 471,073 render roots, seed 29;
- 32 samples per strand;
- identical roots, widths, colors, opacities, camera, mesh, and Blender
  protocol for every variant;
- only curl/frizz components and the secondary shape multiplier are changed.

The five reconstructed variants are:

1. `backbone`: brush backbone with curl and frizz set to zero;
2. `curl_only`: final learned curl on the same backbone, frizz zero;
3. `frizz_only`: final learned frizz on the same backbone, curl zero;
4. `primary_curl_frizz`: primary guide curl/frizz before the secondary local
   correction;
5. `final_curl_frizz`: complete accepted R065 geometry.

## Quantitative result

| variant | cumulative turn P50 | cumulative turn P95 | max local turn P95 | arc/chord P95 | backward strands |
| --- | ---: | ---: | ---: | ---: | ---: |
| backbone | 0.784 deg | 20.458 deg | 0.790 deg | 1.00543 | 0 |
| curl only | 24.896 deg | 108.594 deg | 5.814 deg | 1.03562 | 0 |
| frizz only | 15.432 deg | 59.671 deg | 7.189 deg | 1.01094 | 0 |
| primary curl + frizz | 35.766 deg | 127.774 deg | 10.622 deg | 1.03914 | 0 |
| final curl + frizz | 34.179 deg | 119.282 deg | 9.212 deg | 1.03666 | 0 |

Relative to the exact same backbone, curl adds `18.051 deg` median and
`107.958 deg` P95 cumulative turn. Frizz adds `10.677 deg` median and
`56.606 deg` P95. The final field adds `28.498 deg` median and `118.131 deg`
P95.

The secondary local residual is not the source of the regional curl. It reduces
primary curl/frizz P50 cumulative turn from `35.766 deg` to `34.179 deg` and
P95 from `127.774 deg` to `119.282 deg`, but cannot remove the learned periodic
field.

All five variants have zero segments that travel backward relative to their own
root-to-tip chord. The visible defect is therefore a curled/wavy centerline, not
a segment foldback and not the angle-crossing diagnostic.

## Visual result

The brush-only close-up is coherent. Enabling curl alone creates the broad,
repeated corrugation visible over the upper rump and back. Frizz alone mainly
adds smaller-scale roughness. Its rare high-amplitude outliers can still create
sharp local kinks, but it is not the main source of the large coherent wavy
patch. The final reconstruction retains the curl-only pattern.

Local renders:

- backbone:
  `D:/RTS/_tmp/r065_curl_frizz_component_20260816/closeups/r065_backbone_rump_closeup.png`;
- curl only:
  `D:/RTS/_tmp/r065_curl_frizz_component_20260816/closeups/r065_curl_only_rump_closeup.png`;
- frizz only:
  `D:/RTS/_tmp/r065_curl_frizz_component_20260816/closeups/r065_frizz_only_rump_closeup.png`;
- primary curl/frizz:
  `D:/RTS/_tmp/r065_curl_frizz_component_20260816/closeups/r065_primary_curl_frizz_rump_closeup.png`;
- final curl/frizz:
  `D:/RTS/_tmp/r065_curl_frizz_component_20260816/closeups/r065_final_curl_frizz_rump_closeup.png`.

## Representation audit

The accepted geometry initializes every strand's curl turns to `1.20` and curl
phase to `0`. In the active primary/secondary guide route, optimization includes
the primary and secondary curl-radius ratios and frizz-amplitude ratios, but not
curl turns or phase. The checkpoint confirms that all sampled roots retain
`curl_turns = 1.20` and `curl_phase = 0`.

The curl forward model evaluates

`angle(t) = 2 pi * curl_turns * t + curl_phase`.

Consequently, growing a previously near-zero curl radius does not introduce an
arbitrary gentle bend. It scales a fixed 1.2-revolution periodic template. A
smooth spatial patch of nonzero radii therefore becomes a smooth patch of the
wrong repeated wave. Radius smoothness cannot distinguish that field from a
physically supported curl.

The checkpoint statistics are:

- curl radius ratio: P50 `0.01083`, P95 `0.05848`, max `0.16555`;
- physical curl radius: P50 `0.000263`, P95 `0.001256`, max `0.004663`;
- frizz amplitude ratio: P50 `0.00449`, P95 `0.02041`, max `0.11384`;
- physical frizz amplitude: P50 `0.000108`, P95 `0.000488`, max `0.003430`.

## Gradient ownership audit

`RGB_FLOW_EXCLUDE_COLOR_GRADIENTS=1` only stops the RGB-derived flow loss from
updating color parameters. It still routes flow gradients to non-color
parameters, including curl/frizz. The ordinary RGB-and-regularization backward
also updates geometry. Shape detail and the Gaussian RGB residual are unlocked
over the same late part of training, so periodic geometry and appearance have
competing ways to absorb image detail.

The earlier fixed-checkpoint RGB ablation found only `+0.47657 dB` from shape
detail but `+2.32969 dB` from the Gaussian RGB residual. The visible GT in the
highlighted rump does not contain a dense 1.2-turn curl field. Together with
the component render, this supports the conclusion that curl is absorbing image
detail rather than reconstructing supported fur geometry.

## Conclusion and next controlled change

The main visible curl problem is a representation-and-gradient-ownership issue,
not crossing and not insufficient crossing regularization:

1. the brush backbone is already coherent;
2. learned curl radius scales a fixed 1.2-turn template;
3. RGB and noisy RGB-derived flow can grow that template;
4. spatial smoothness can make the wrong periodic field coherent but cannot
   decide whether the curl is supported;
5. secondary residuals repair part of the deformation but do not cause it.

The next experiment must remain a single controlled geometry change. Curl turns
must become a real primary-guide groom field with a neutral zero-curl state,
rather than a fixed 1.2-turn template whose amplitude alone is learned. Curl
radius and turns then remain editable, interpolated, and smooth fields. Raw RGB
appearance should not be the sole evidence that activates periodic geometry;
the separate appearance residual remains responsible for unsupported texture,
while curl/frizz require structural evidence. No white-tiger-specific region,
absolute curl clamp, or crossing threshold is justified by this diagnosis.

## Artifacts

- diagnostic tool: `tools/diagnose_curl_frizz_components.py`;
- exact report:
  `D:/RTS/_tmp/r065_curl_frizz_component_20260816/curl_frizz_component_report.json`;
- component strand archives:
  `D:/RTS/_tmp/r065_curl_frizz_component_20260816`;
- full 1920x1080 renders:
  `D:/RTS/_tmp/r065_curl_frizz_component_20260816/assets`;
- 1600x1200 rump close-ups:
  `D:/RTS/_tmp/r065_curl_frizz_component_20260816/closeups`.
