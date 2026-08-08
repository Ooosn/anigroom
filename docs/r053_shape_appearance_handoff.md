# R053 Shape/Appearance Handoff

## Status

Completed and rejected as the next baseline. The Gaussian RGB residual is a
useful appearance outlet, but it does not by itself stop the secondary-guide
curl/frizz residual field from absorbing image texture. R050 remains the
accepted appearance checkpoint.

## Question

Gaussian RGB residual was introduced to absorb stripes, inter-fur shadow,
gloss, and image noise that would otherwise push editable geometry into false
length, direction, or curvature. R050 validates the residual while curl and
frizz are disabled. It therefore does not yet test the intended joint handoff.

R053 asks whether the Gaussian appearance outlet protects geometry when the
optional explicit curl/frizz controls are allowed to learn.

## Representation Contract

- R050 remains the parent; its flow, roots, interpolation, lifecycle, color
  ownership, losses, resolution, and fixed visualization protocol do not move.
- The retired legacy bend parameter is not restored. Guide-owned
  `brush_stiffness` remains the ordinary one-turn base centerline control.
- Guide and secondary-guide curl radius/frizz amplitude become optimizable.
- Curl and frizz initialize at their semantic neutral value instead of exposing
  the retired nonzero template when the gate opens. The secondary residual
  remains zero-centered and supplies a nonsaturated learning path.
- Effective curl and frizz remain exactly zero through 10k, then ramp from
  10k to 20k.
- The treatment Gaussian RGB residual uses that exact 10k-20k ramp.
- The matched control has the same residual tensor and optimizer group, but
  keeps its multiplier exactly zero through 30k.
- Curl frequency/phase retain the existing explicit template in this test.
  R053 evaluates noise isolation, not general curly-animal frequency recovery.

Configurations:

```text
configs/r053_shape_detail_no_gaussian_residual_0_30k.env
configs/r053_shape_detail_gaussian_residual_0_30k.env
configs/r053_shape_detail_gaussian_residual_fullres_preflight.env
```

The H100 execution entry is
`scripts/server/run_r053_shape_appearance_handoff.sh`. It verifies an exact
clean commit, runs the real full-resolution active-path preflight, waits at an
explicit authorization marker, then runs the control and treatment from zero
in sequence on one GPU. It also invokes the existing fixed checkpoint renderer
and deterministic 100k-strand exporter; it does not introduce another QA
protocol.

The active-path preflight also overrides the inherited guide and residual
freezes. It refuses authorization unless guide curl/frizz, secondary-guide
curl/frizz residuals, and the Gaussian RGB residual all have finite, nonzero
Adam first moments after the same real 1920x1080 step. The local RTX 4080
preflight passed this contract with 400k render roots and a 7.55 GB peak
allocation; this is a gradient-chain check, not a metric result.

## Required Comparison

The two from-zero runs differ only in whether the Gaussian residual becomes
active. They must match through 10k. At 30k report:

1. train/test composite PSNR and root/Gaussian counts;
2. fixed eight-view full RGB;
3. same-checkpoint residual-off, shape-detail-off, and both-off renders;
4. Gaussian residual magnitude/saturation;
5. guide/effective curl and frizz statistics;
6. deterministic 100k-strand fixed Blender views and structural statistics.

The treatment succeeds only if it reduces false curl/frizz, crossings, and
local turn tails relative to the no-residual control while preserving useful
RGB evidence. PSNR alone cannot accept it.

## Formal Result

Both full-resolution from-zero branches completed 30k and exited cleanly.

| Branch | Train composite | Test composite | Render roots | Gaussians | Peak allocated |
| --- | ---: | ---: | ---: | ---: | ---: |
| no Gaussian residual | 32.5532 | 31.8591 | 469,608 | 5,782,587 | 17,164.3 MB |
| Gaussian residual | 33.7561 | 32.5362 | 469,830 | 5,938,778 | 17,584.3 MB |

The Gaussian residual improves final train/test composite by
`+1.2029/+0.6771 dB` over the matched control and improves final test composite
by `+0.4151 dB` over R050. Across the fixed eight views, disabling only the
Gaussian residual from the treatment checkpoint costs `2.2522 dB` on average.
The learned residual is not dormant: its absolute mean/RMS are
`0.0752/0.1031`, `6.21%` of values reach the configured color-domain boundary,
and about `80.3%` of generated Gaussian samples are active.

## Structural Audit

The fixed deterministic 100k-strand audit does not satisfy the acceptance
condition.

| Metric | R050 | Control | Treatment |
| --- | ---: | ---: | ---: |
| 4NN relative-length mean | 0.02047 | 0.02823 | 0.02632 |
| 4NN relative-length P95 | 0.07741 | 0.10200 | 0.09376 |
| backward strands | 0 | 762 | 560 |
| arc/chord P95 | 1.00673 | 1.18163 | 1.19097 |
| maximum local-turn P95 | 0.9546 deg | 48.0566 deg | 58.8473 deg |

The treatment improves local length continuity and reduces backward strands by
about `26.5%` relative to the control, but its local-turn P95 is about `22.5%`
worse and its arc/chord tail is slightly worse. Both R053 branches are much
more curved than R050.

The attribute maps explain the failure. Control and treatment curl fields both
trace high-frequency tiger-stripe structure. The treatment lowers mean/P95/max
curl and the extreme frizz maximum, but raises mean frizz by about `71%` and
frizz P95 by about `40%`. Its frizz field becomes broad and patchy rather than
remaining a coherent groom property. The treatment also learns roughly
`9-10%` thinner strands, which makes its fixed wide-view Blender assets look
cleaner while partially hiding the larger local-turn tail.

## Decision

R053 proves that Gaussian-level RGB residual is effective and should remain in
the method. It does not prove a clean shape/appearance handoff while
curl/frizz have both primary-guide and secondary-guide degrees of freedom.
Therefore:

- do not promote either R053 branch;
- keep R050 as the accepted baseline;
- preserve R053 as evidence that PSNR-only acceptance would be wrong;
- next isolate attribute ownership: primary guides own smooth curl/frizz,
  while secondary-guide residuals no longer carry those two high-frequency
  groom attributes. The Gaussian RGB residual and the 10k-20k handoff remain
  unchanged.

## Artifacts

- H100 runtime:
  `/home/wangyy/anigroom-r053-shape-appearance-handoff-runtime-20260808`
- local fixed QA:
  `D:/RTS/_tmp/r053_h100_postprocess_20260809`
- deterministic strand audit:
  `D:/RTS/_tmp/r053_h100_postprocess_20260809/postprocess/r050_r053_strand_audit.json`
- treatment canonical asset:
  `D:/RTS/_tmp/r053_h100_postprocess_20260809/postprocess/r053_treatment/assets/r053_treatment_030000_asset_side_y_v11_protocol.png`
- control canonical asset:
  `D:/RTS/_tmp/r053_h100_postprocess_20260809/postprocess/r053_control/assets/r053_control_030000_asset_side_y_v11_protocol.png`
