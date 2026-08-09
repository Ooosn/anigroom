# R055 Staged Primary/Secondary Shape Handoff

## Question

Can Gaussian RGB residual absorb image-specific appearance evidence while a
coarse-to-fine curl/frizz hierarchy adds legitimate strand detail without
returning to R053's simultaneous two-level ownership failure?

## Frozen Parent

R054 is immutable. R055 changes only curl/frizz ownership timing and their
secondary residual composition. Length, width, direction, brush stiffness,
interpolation, losses, render-root lifecycle, guide lifecycle, and learning
rates remain identical to R054.

## Method

The primary guide owns absolute semantic curl and frizz. Their common shape
multiplier is zero through iteration 20,000 and ramps linearly to one at
25,000. Gaussian RGB residual uses exactly the same 20,000-25,000 ramp.

The secondary guide owns only zero-centered positive relative residuals:

```text
effective = primary * exp(asinh(secondary_raw) * secondary_multiplier)
```

Its one shared multiplier is zero through 25,000 and ramps linearly to one at
30,000. A zero residual returns the primary field exactly; a zero primary value
cannot be turned into curl or frizz by the secondary field.

No new loss, absolute threshold, animal region, bend path, or attribute-specific
schedule is introduced.

## Acceptance

1. Formal full-resolution preflight proves primary curl, primary frizz,
   secondary curl residual, secondary frizz residual, and Gaussian RGB residual
   all receive finite nonzero optimizer state when active.
2. A from-zero 30k run completes under the existing 25 GB guard.
3. Fixed QA compares R055 with R050 and R054 using identical RGB and canonical
   strand render protocols.
4. R055 is accepted only if the late secondary handoff adds useful structure
   without recreating R053's curl-back, isolated long hair, or incoherent local
   turning.

## Result

### Execution

The formal from-zero H100 run completed with exit status zero at source commit
`16835d8a24de47b332e246a8c54613d739053baf`.

| Measurement | Result |
| --- | ---: |
| elapsed time | `9582.171 s` (`2 h 39 m 42 s`) |
| final train composite PSNR | `33.404186` |
| final test composite PSNR | `32.255074` |
| best test composite PSNR | `32.329842` at 29k |
| final render roots | `469771` |
| final training-metric Gaussians | `5645519` |
| peak allocated CUDA memory | `14650.57 MB` |
| peak `nvidia-smi` process memory | `17882 MB` |

The final render-root lifecycle event is still the 9k event, observed in the
population at 9020. Root count remains exactly `469771` afterward, and
`lifecycle_statistics_active=false` from 10k through 30k. The new shape
schedule therefore does not silently reactivate lifecycle work.

The full-resolution preflight also verified finite nonzero Adam first moments
for all intended owners when active: all 4500 primary curl/frizz rows, all
20000 secondary curl/frizz residual rows, and 13,548,096 generated-Gaussian
RGB-residual entries.

### Schedule trajectory

R055 and R054 are behavior-equivalent only through 10k. R054 ramps primary
shape and Gaussian RGB residual from 10k to 20k; R055 intentionally delays
that shared ramp to 20k-25k, then ramps the secondary residual from 25k to
30k. They must not be described as equal through 20k.

| Iteration | Primary shape | Gaussian RGB residual | Secondary shape residual | Test composite |
| ---: | ---: | ---: | ---: | ---: |
| 20000 | `0.0` | `0.0` | `0.0` | `31.412` |
| 21000 | `0.2` | `0.2` | `0.0` | `31.533` |
| 22000 | `0.4` | `0.4` | `0.0` | `31.643` |
| 23000 | `0.6` | `0.6` | `0.0` | `31.843` |
| 24000 | `0.8` | `0.8` | `0.0` | `32.003` |
| 25000 | `1.0` | `1.0` | `0.0` | `31.967` |
| 26000 | `1.0` | `1.0` | `0.2` | `32.150` |
| 27000 | `1.0` | `1.0` | `0.4` | `32.230` |
| 28000 | `1.0` | `1.0` | `0.6` | `32.138` |
| 29000 | `1.0` | `1.0` | `0.8` | `32.330` |
| 30000 | `1.0` | `1.0` | `1.0` | `32.255` |

### Fixed-view appearance audit

The identical eight-view 1920x1080 postprocess gives:

| Measurement | R054 | R055 | Delta |
| --- | ---: | ---: | ---: |
| mean composite PSNR | `33.47170` | `33.33922` | `-0.13248 dB` |
| mean PSNR without Gaussian RGB residual | `31.28880` | `31.69895` | `+0.41015 dB` |
| mean Gaussian RGB-residual contribution | `2.18290 dB` | `1.64027 dB` | `-0.54263 dB` |
| mean shape-detail contribution | `1.33385 dB` | `0.69488 dB` | `-0.63897 dB` |

R055 is lower than R054 by `0.092-0.214 dB` on every fixed view, rather than
being pulled down by one bad camera. That loss is paired with a substantially
smaller shape and appearance correction. The Gaussian residual parameter has
absolute mean `0.05410`, `70.88%` active entries, and only `2.02%` saturation;
its view09 rendered absolute image change is `0.00588` on `[0,1]`. It remains
an appearance outlet and does not overwrite the low-frequency hair color.

### Fixed 100k-strand structural audit

All runs below use the same 100,000 roots, 32 samples per strand, K4 root
neighbors, and canonical export code.

| Measurement | R050 | R054 | R055 |
| --- | ---: | ---: | ---: |
| local relative-length mean | `0.02047` | `0.02417` | `0.02226` |
| local relative-length P95 | `0.07741` | `0.09122` | `0.08422` |
| local 3D chord-direction mean | `3.828 deg` | `3.841 deg` | `3.861 deg` |
| local 3D chord-direction P95 | `11.296 deg` | `11.330 deg` | `11.391 deg` |
| strands with a backward segment | `0` | `375` | `159` |
| arc/chord P95 | `1.00673` | `1.15280` | `1.10128` |
| maximum-local-turn P95 | `0.955 deg` | `57.297 deg` | `50.027 deg` |
| maximum-local-turn P99 | `2.430 deg` | `74.317 deg` | `65.390 deg` |
| arc length above `0.12` | `5` | `0` | `4` |

R055 therefore improves every measured curvature/foldback statistic over R054
except the essentially unchanged neighbor chord direction, and it restores
about half of R054's loss of local length continuity. It does not return to
R050's nearly straight strict-structure field. Four mild long-tail strands
reappear (`max=0.12738`), so this is not evidence that the secondary residual
is free of all local outliers.

The view09 attribute audit supports the same conclusion. Compared with R054,
R055 lowers curl-radius mean/P95 from `0.000711/0.002272` to
`0.000462/0.001409`, curl amount from `0.000853/0.002726` to
`0.000555/0.001690`, and frizz from `0.000548/0.001633` to
`0.000431/0.001430`. Length mean/P95 stays nearly fixed
(`0.02403/0.04243` versus `0.02401/0.04266`), so the structural improvement is
not produced by globally shortening the coat.

### Visual decision

The identical three-camera Blender assets are coherent and contain no R053
style curl-back cluster, body spiral, or global collapse. R055 has fewer
extreme head spikes and visibly calmer local turning than R054, while retaining
the same overall groom. A few sparse head-fringe outliers remain and agree with
the four strands above `0.12` in the numeric audit.

R055 is accepted as the latest staged shape/appearance research checkpoint and
replaces R054 for subsequent controlled shape experiments. It does not replace
R050 as the strict structural reference, and it is not promoted as the default
Stage 1 baseline. The reusable finding is the ownership order: smooth primary
shape plus Gaussian appearance first, then a zero-centered secondary residual.
The remaining gap to R050 shows that secondary curl/frizz still needs a
stronger structure-preserving formulation before it becomes default behavior.

## Artifacts

- H100 output:
  `/home/wangyy/anigroom-r055-staged-shape-runtime-20260809/outputs/r055_staged_primary_secondary_shape_0_30k_h100_20260809`
- H100 postprocess:
  `/home/wangyy/anigroom-r055-staged-shape-runtime-20260809/postprocess/r055_staged_primary_secondary_shape`
- verified local postprocess:
  `D:/RTS/_tmp/r055_h100_postprocess_20260809/postprocess/r055_staged_primary_secondary_shape`
- canonical side asset:
  `D:/RTS/_tmp/r055_h100_postprocess_20260809/postprocess/r055_staged_primary_secondary_shape/assets/r055_030000_asset_side_y_v11_protocol.png`
- canonical opposite-side asset:
  `D:/RTS/_tmp/r055_h100_postprocess_20260809/postprocess/r055_staged_primary_secondary_shape/assets/r055_030000_asset_side_y_pos_v11_protocol.png`
- canonical top/front asset:
  `D:/RTS/_tmp/r055_h100_postprocess_20260809/postprocess/r055_staged_primary_secondary_shape/assets/r055_030000_asset_front_z_v11_protocol.png`
- fixed strand audit:
  `D:/RTS/_tmp/r055_h100_postprocess_20260809/postprocess/r050_r054_r055_strand_audit.json`
