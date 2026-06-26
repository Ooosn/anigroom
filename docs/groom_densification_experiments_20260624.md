# Groom Densification Experiments 2026-06-24

This file records controlled experiments around the current root
densification/prune module. These runs are diagnostic baselines and isolated
variants. They are not allowed to overwrite the baseline output directories.

## Fixed Baselines

| Run | Purpose | Final roots | Final PSNR | Decision |
| --- | --- | ---: | ---: | --- |
| `groom_density_no_densification_animal_v1` | sparse-root control | 504 | 20.7847 | keep as no-densify control |
| `groom_density_densification_animal_rootmove_gradual_v1` | animal-density gradual densification baseline | 4088 | 31.4992 | current animal-density baseline |
| `groom_density_densify_prune_islands_v2_effective` | combined densify + prune on island target | 1012 | 32.3562 | current prune validation baseline |

## Isolated Variants

| Run | Change | Final roots | Final PSNR | Result |
| --- | --- | ---: | ---: | --- |
| `groom_density_animal_gradual_detailbias_v1` | analytic detail-biased parent ranking | 4088 | 31.3651 | reject; worse than baseline |
| `groom_density_animal_gradual_smooth002_v1` | root-graph groom smoothness, weight 0.02 | 4088 | 31.5797 | positive but weaker |
| `groom_density_animal_gradual_smooth_v1` | root-graph groom smoothness, weight 0.04 | 4088 | 31.6058 | accept as current best variant |

## Current Conclusions

1. Densification is necessary on the animal-density target. The no-densify
   control remains far below the gradual densification baseline.
2. Pure analytic detail-biased parent selection is not useful in the current
   setup. It lowers PSNR and does not improve orientation losses.
3. Root-graph groom smoothness is useful when it is applied directly on groom
   parameters and protected near high-detail regions. Weight `0.04` currently
   gives the best tested tradeoff.
4. Accepted smoothness is a reconstruction stabilizer, not an asset-only
   trick. It fits the confirmed framework because groom parameters should be
   locally coherent on the mesh/root graph.

## Multi-Style Stress Sweep

All runs below keep the accepted modules active:

- projected curve initialization
- root movement
- gradual root densification
- orientation/detail losses
- flow coherence
- root-graph groom smoothness, weight `0.04`
- no analytic detail-bias

| Style | Density target | Steps | Final roots | Final PSNR | Observation |
| --- | --- | ---: | ---: | ---: | --- |
| `short_plush` | `animal_density` | 1360 | 4088 | 34.5744 | easy case; short plush fur is well covered |
| `dog_guard` | `animal_density` | 1360 | 4088 | 32.9672 | good animal guard-hair behavior |
| `wavy` | `animal_density` | 1360 | 4088 | 32.1355 | works, but orientation residual remains higher |
| `wet_matted` | `matted_density` | 1360 | 4088 | 32.0421 | matted/clumped fur is not the main failure mode |
| `cowlick_whorl` | `animal_density` | 1360 | 4088 | 31.7404 | local flow rotation is harder but still stable |
| `patchy_length` | `animal_density` | 1360 | 4088 | 30.8768 | long/short boundary needs longer convergence |
| `dirty_tangled` | `matted_density` | 1360 | 4088 | 28.7538 | current hardest case; high frizz and flow disagreement |

Longer optimization without changing root capacity:

| Style | Steps | Final roots | Final PSNR | Decision |
| --- | ---: | ---: | ---: | --- |
| `mixed_animal` | 2400 | 4088 | 32.4423 | accepted; representative baseline improves from 31.6058 |
| `dog_guard` | 2400 | 4088 | 33.6264 | accepted; animal guard hair improves from 32.9672 |
| `patchy_length` | 2400 | 4088 | 31.6272 | longer root-fixed convergence helps substantially |
| `dirty_tangled` | 2400 | 4088 | 29.2800 | longer training helps only mildly |

Additional hair-type coverage using the same V2 schedule:

| Style | Density target | Steps | Final roots | Initial PSNR | Final PSNR | Observation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `tiger_plush` | `animal_density` | 2400 | 4088 | 18.4393 | 34.7130 | target-relevant short plush/stripe coat is stable |
| `long_silky` | `animal_density` | 2400 | 4088 | 15.8957 | 38.4707 | long smooth sagging fur is not a bottleneck |
| `curly_ringlet` | `matted_density` | 2400 | 4088 | 18.6035 | 30.9226 | dense curled/ringlet fur remains difficult |
| `mane_ridge` | `animal_density` | 2400 | 4088 | 16.1306 | 34.7786 | localized long ridge/mane behavior is stable |
| `coarse_guard` | `animal_density` | 2400 | 4088 | 14.3211 | 35.6501 | coarse guard-hair width/stiffness variation is stable |

Additional 2400-step coverage with the same accepted V2 modules:

| Style | Density target | Steps | Final roots | Initial PSNR | Final PSNR | Observation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `cowlick_whorl` | `animal_density` | 2400 | 4088 | 17.8843 | 32.1250 | local whorl/rotating flow trains stably but remains harder than straight plush fur |
| `wavy` | `animal_density` | 2400 | 4088 | 17.6293 | 32.9207 | improves clearly over the 1360-step result; 2400 steps is a better minimum diagnostic budget |
| `wet_matted` | `matted_density` | 2400 | 4088 | 9.5272 | 32.9706 | very low projected initialization still recovers, so wet/clumped sag is not the main bottleneck |
| `clumped` | `matted_density` | 2400 | 4088 | 18.5166 | 33.7617 | child radius and clump strength are effective groom controls |
| `frizzy` | `matted_density` | 2400 | 4088 | 20.8929 | 32.7323 | high-frequency frizz is trainable but weaker than coherent clumps |
| `curled` | `matted_density` | 2400 | 4088 | 17.8557 | 29.5899 | classic curled hair is a hard case under the current curve parameterization |

Dirty/tangled targeted diagnostics:

| Variant | Change | Final roots | Final PSNR | Decision |
| --- | --- | ---: | ---: | --- |
| `smooth001_protect09` | weaker smoothness for high-frizz fur | 4088 | 29.3573 | small gain only; smoothness is not the main bottleneck |
| `morecap_smooth004` | more densification capacity | 5368 | 29.6953 | meaningful gain; dirty/tangled fur needs more local capacity |
| `local_projected_curve_4088` | local length/clump/frizz initialization | 4088 | 29.1983 | reject; local attribute init alone does not solve the failure |
| `orient_residual_bias2_4088` | screen orientation-residual parent ranking | 4088 | 28.9170 | reject; naive screen residual ranking hurts parent selection |

Hard-case capacity check using only existing modules:

| Style | Change | Steps | Final roots | Final PSNR | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `curled` | longer run + higher densification capacity | 3600 | 7800 | 30.6503 | capacity helps only mildly; do not solve by just adding roots |
| `curly_ringlet` | longer run + higher densification capacity | 3600 | 7800 | 32.5270 | capacity matters for ringlets, but still trails plush/guard cases |
| `dirty_tangled` | longer run + higher densification capacity | 3600 | 7800 | 30.7378 | more roots improve the case, but high-frequency flow/frizz remains the bottleneck |

Curled-hair optimization-only probes:

| Variant | Change | Steps | Final roots | Final PSNR | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `curled_curl_lr16_v2_2400` | raise curl learning-rate scale from `8` to `16` | 2400 | 4088 | 29.1132 | reject; worse than V2 |
| `curled_curl_lr16_orient075_v2_2400` | stronger orientation/detail supervision plus higher curl LR | 2400 | 4088 | 28.6066 | reject; hurts RGB/alpha without fixing orientation |
| `curled_curl_lr16_lowsmooth_v2_2400` | weaker groom smoothness and stronger detail protection | 2400 | 4088 | 29.1210 | reject; smoothness is not the main limitation |

Root-split phase interpolation probe:

| Style | Change | Steps | Final roots | Final PSNR | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `curled` | circular interpolation for `curl_phase` during split | 2400 | 4088 | 29.5795 | neutral; does not solve curled hair |
| `curly_ringlet` | circular interpolation for `curl_phase` during split | 2400 | 4088 | 30.8921 | neutral; does not solve ringlets |
| `tiger_plush` | circular interpolation for `curl_phase` during split | 2400 | 4088 | 34.9660 | positive for target-like plush/stripe fur |
| `mixed_animal` | circular interpolation for `curl_phase` during split | 2400 | 4088 | 32.3793 | essentially neutral, slightly below V2 |
| `dog_guard` | circular interpolation for `curl_phase` during split | 2400 | 4088 | 33.7695 | positive |
| `clumped` | circular interpolation for `curl_phase` during split | 2400 | 4088 | 33.6763 | essentially neutral, slightly below V2 |

Decision: keep circular `curl_phase` interpolation as root-split hygiene. It
is mathematically correct for a periodic groom parameter and helps target-like
plush/guard fur, but it should not be presented as the solution to curly hair.
When the root lifecycle is ported to the real white-tiger trainer, phase-like
groom parameters must use circular interpolation rather than plain arithmetic
averaging.

## Extended V2.1 Hair-Type Stress Suite

This sweep uses the accepted modules together, without changing the baseline
logic:

```text
projected_curve initialization
periodic curl-phase interpolation for split children
root movement enabled
gradual root densification
flow coherence = 1.0
root-graph groom smoothness = 0.04
detail protection = 0.65
prune disabled
```

Target-relevant and regular animal coats use 3200 steps with densification
until 1600. Harder boundary/frizz/curl cases use 3600 steps with densification
until 2000 and a higher per-event split cap. All runs use 1440 x 810 rendering,
not a low-resolution substitute.

| Style | Density target | Steps | Final roots | Initial PSNR | Final PSNR | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `tiger_plush` | `animal_density` | 3200 | 5368 | 18.4393 | 36.2670 | strong; target-relevant plush/stripe coat is ready for white-tiger Stage 1 |
| `short_plush` | `animal_density` | 3200 | 5368 | 18.0555 | 36.6880 | strong; short dense fur is not a bottleneck |
| `mixed_animal` | `animal_density` | 3200 | 5368 | 15.3712 | 33.2188 | acceptable; mixed length/color/flow remains harder but trains stably |
| `dog_guard` | `animal_density` | 3200 | 5368 | 16.0519 | 35.2360 | strong; guard-hair zones and ridge direction changes are covered |
| `coarse_guard` | `animal_density` | 3200 | 5368 | 14.3211 | 37.1711 | strong; width/taper/stiffness variation is effective |
| `long_silky` | `animal_density` | 3200 | 5368 | 15.8957 | 39.7933 | strong; long smooth sagging fur is not a bottleneck |
| `mane_ridge` | `animal_density` | 3200 | 5368 | 16.1306 | 36.0230 | strong; localized long ridge/mane behavior is stable |
| `wavy` | `animal_density` | 3200 | 5368 | 17.6293 | 33.7897 | acceptable; wave/flow variation improves with longer training |
| `cowlick_whorl` | `animal_density` | 3200 | 5368 | 17.8843 | 32.9344 | borderline; local rotating flow is a real harder case |
| `patchy_length` | `animal_density` | 3600 | 8184 | 19.2397 | 33.4372 | acceptable; higher capacity fixes most of the long/short boundary gap |
| `wet_matted` | `matted_density` | 3200 | 5368 | 9.5272 | 34.1682 | strong; wet/clumped sag recovers from very poor projection |
| `clumped` | `matted_density` | 3200 | 5368 | 18.5166 | 34.8185 | strong; child radius and clump strength are effective controls |
| `frizzy` | `matted_density` | 3600 | 8184 | 20.8929 | 34.2358 | good PSNR, but orientation residual remains high |
| `curly_ringlet` | `matted_density` | 3600 | 8184 | 18.6035 | 32.4782 | hard; ringlet curls need a stronger local curve representation |
| `curled` | `matted_density` | 3600 | 8184 | 17.8557 | 30.6060 | fail/hard; classic curled hair is not solved by more roots or longer training |
| `dirty_tangled` | `matted_density` | 3600 | 8184 | 17.1629 | 30.4944 | fail/hard; tangled high-frequency flow/frizz needs a new mechanism |

Conclusions:

1. The current V2.1 baseline is strong enough for the white-tiger-like coat
   family: plush, stripe, short dense fur, guard hair, mane/ridge, long smooth
   fur, wet matted fur, clumped fur, and frizzy fur all reach or exceed 34 PSNR
   under the extended schedule.
2. The remaining failure mode is specific rather than generic: high-frequency
   local curve/flow fields such as classic curls, ringlets, dirty tangles, and
   local whorls. These cases keep high orientation/detail residuals even after
   extra roots and longer training.
3. The next useful module should target local curve/frizz expressiveness or a
   stronger projected curl initialization. Generic smoothing, higher curl
   learning rate, stronger orientation weighting, naive detail-biased parent
   ranking, or simply adding more roots have already been tested and are not
   sufficient.

Artifacts:

- `D:\petsgaussianhair\_downloads\groom_density_v21_extended_suite_20260624\suite_summary.csv`
- `D:\petsgaussianhair\_downloads\groom_density_v21_extended_suite_20260624\extended_suite_contact_sheet.png`

## Extra Animal-Groom Stress Types

Output:

```text
D:\petsgaussianhair\_downloads\groom_density_v21_extra_types_20260624
```

These runs keep the accepted V2.1 modules enabled:

```text
projected_curve initialization
root movement
gradual root densification
periodic curl-phase interpolation for split children
flow coherence = 1.0
root-graph groom smoothness = 0.04
detail protection = 0.65
1440 x 810 rendering
```

During this sweep, the lifecycle code exposed a real bug: when densification
and pruning occurred in the same iteration, pruning still used the pre-split
pressure/contribution tensors. This has been fixed by reordering the signal
tensors to match the post-split root order, with child roots inheriting the
parent signal for the current prune event. The same iteration no longer
misaligns prune masks after split.

| Style | Density target | Steps | Final roots | Initial PSNR | Final PSNR | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `fine_undercoat` | `animal_density` | 3200 | 5368 | 38.7347 | 47.1557 | very strong; short dense undercoat is not a bottleneck |
| `side_parted` | `animal_density` | 3200 | 5368 | 20.7886 | 37.8562 | strong; opposite-flow parting lines are handled by current flow/smooth setup |
| `facial_short_to_long` | `animal_density` | 3600 | 8184 | 19.2449 | 35.6854 | strong; face-to-neck short/long transition is trainable |
| `spiky_guard` | `animal_density` | 3200 | 5368 | 17.5419 | 33.7089 | acceptable but harder; stiff guard/taper boundaries remain a useful stress case |
| `ringlet_plus_undercoat` | `matted_density` | 3600 | 8184 | 17.2925 | 33.0495 | borderline; dense ringlets over undercoat expose local curve/detail capacity limits |

Artifacts:

- `D:\petsgaussianhair\_downloads\groom_density_v21_extra_types_20260624\extra_types_training_contact_sheet.png`
- `D:\petsgaussianhair\_downloads\groom_density_v21_extra_types_20260624\extra_types_effective_roots_contact_sheet.png`

Conclusion:

The current baseline is not weak on ordinary animal grooming variation. It
handles short undercoat, directional parting, face-to-neck length transitions,
and many guard-hair cases. The remaining baseline-improvement target is more
specific: local curve capacity for ringlets/tangled curls and sharper stiff
guard boundaries. The next useful technical attempt should therefore improve
local curve representation or curl/detail supervision, not merely increase
root count.

## Fixed Baseline V2.2 Suite

Output:

```text
D:\petsgaussianhair\_downloads\groom_density_v22_baseline_suite_20260624
```

This folder merges the 16-case extended suite with the 5 extra animal-groom
stress cases into one fixed baseline summary. It is the current synthetic
groom/densification baseline to compare against before moving modules into
white-tiger Stage 1.

Common settings:

```text
projected_curve initialization
root movement
gradual root densification
periodic curl-phase interpolation for split children
flow coherence = 1.0
root-graph groom smoothness = 0.04
detail protection = 0.65
rendering = 1440 x 810
regular schedule = 3200 steps, densify until 1600, final roots 5368
hard schedule = 3600 steps, densify until 2000, final roots 8184
```

Summary:

| Group | Cases | Result |
| --- | --- | --- |
| Stable animal fur | `fine_undercoat`, `long_silky`, `coarse_guard`, `side_parted`, `short_plush`, `tiger_plush`, `facial_short_to_long`, `dog_guard`, `mane_ridge` | strong reconstruction and coherent root densification |
| Acceptable stress cases | `wet_matted`, `frizzy`, `clumped`, `spiky_guard`, `patchy_length`, `wavy`, `mixed_animal`, `cowlick_whorl` | trainable but exposes higher local detail/orientation pressure |
| Boundary cases | `curled`, `dirty_tangled`, `curly_ringlet`, `ringlet_plus_undercoat` | local curl/tangle/detail capacity remains the next target |

Artifacts:

- `D:\petsgaussianhair\_downloads\groom_density_v22_baseline_suite_20260624\suite_summary.csv`
- `D:\petsgaussianhair\_downloads\groom_density_v22_baseline_suite_20260624\suite_final_contact_sheet.png`
- `D:\petsgaussianhair\_downloads\groom_density_v22_baseline_suite_20260624\suite_effective_roots_contact_sheet.png`

## Child-Capacity Probe

Output:

```text
D:\petsgaussianhair\_downloads\groom_density_v22_child_capacity_probe_20260624
```

Question: for boundary curl cases, is the bottleneck guide-root count or
per-root child strand capacity?

Only `child_count` was changed from the V2.2 setting to 6. The accepted
modules remained enabled: projected-curve initialization, root movement,
gradual densification, flow coherence, groom smoothness, and detail
protection.

| Style | V2.2 PSNR | Child-count 6 PSNR | Final roots | Final Gaussians | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `ringlet_plus_undercoat` | 33.0495 | 33.5379 | 8184 | 2648456 | useful for layered ringlet/undercoat, but expensive |
| `curled` | 30.6060 | 30.7754 | 8184 | 2873805 | weak gain; pure curl remains a curve/phase-capacity issue |

Artifacts:

- `D:\petsgaussianhair\_downloads\groom_density_v22_child_capacity_probe_20260624\child_capacity_summary.csv`
- `D:\petsgaussianhair\_downloads\groom_density_v22_child_capacity_probe_20260624\child_capacity_probe_sheet.png`

Conclusion:

Increasing child strands under each guide root is not a clean default
baseline. It can help layered undercoat/ringlet cases, but it gives limited
gain for pure curled hair and substantially increases Gaussian count. The
next high-value module should target local curl/phase/detail representation
more directly, while keeping child capacity as a controlled option for
multi-layer animal fur.

## V2.2 Longer-Step Probe

Output:

```text
D:\petsgaussianhair\_downloads\groom_density_v22_step_probe_20260624
```

Question: are the weaker V2.2 cases merely under-trained, or do they expose a
representation/supervision bottleneck?

This probe keeps the accepted V2.2 modules enabled:

```text
projected_curve initialization
root movement
gradual root densification
periodic curl-phase interpolation for split children
flow coherence = 1.0
root-graph groom smoothness = 0.04
detail protection = 0.65
rendering = 1440 x 810
```

Only schedule length and final root budget were changed. Regular controls use
4200 steps and densify until 2000, while hard curl/tangle cases use 5200 steps
and densify until 2800.

| Style | V2.2 PSNR | Longer-step PSNR | Delta | Roots | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `long_silky` | 39.7933 | 41.3103 | +1.5170 | 6648 | longer schedule helps and remains stable |
| `spiky_guard` | 33.7089 | 34.7379 | +1.0290 | 6648 | stiff guard boundaries benefit from more coverage |
| `ringlet_plus_undercoat` | 33.0495 | 34.0505 | +1.0010 | 11384 | more roots plus post-densify stabilization outperforms the child-count-6 probe |
| `tiger_plush` | 36.2670 | 37.2175 | +0.9505 | 6648 | ordinary plush fur was not saturated at 3200 steps |
| `curly_ringlet` | 32.4782 | 33.3456 | +0.8674 | 11384 | improves, but orientation/detail error remains high |
| `dirty_tangled` | 30.4944 | 31.3384 | +0.8440 | 11384 | improves, but local tangled flow remains unresolved |
| `cowlick_whorl` | 32.9344 | 33.4285 | +0.4941 | 6648 | whorl flow remains a harder direction-field case |
| `curled` | 30.6060 | 31.0124 | +0.4064 | 11384 | classic curled hair is not solved by longer training/root count |

Artifacts:

- `D:\petsgaussianhair\_downloads\groom_density_v22_step_probe_20260624\step_probe_vs_v22.csv`
- `D:\petsgaussianhair\_downloads\groom_density_v22_step_probe_20260624\step_probe_final_contact_sheet.png`
- `D:\petsgaussianhair\_downloads\groom_density_v22_step_probe_20260624\suite_summary.csv`

Conclusion:

The longer schedule is a better default direction for animal-fur reconstruction
than the shorter V2.2 schedule, but it does not close the boundary cases.
Ordinary plush, silky, and guard-hair cases improve cleanly. Multi-layer
ringlet/undercoat also benefits from more roots and a longer post-densify
stabilization phase, so child-count growth should remain optional rather than
default. Classic curled hair and dirty/tangled flow remain the strongest
evidence that the next module should target local curve/phase/detail
representation or split-child initialization, not another generic smoothness
term or unbounded root growth.

## Fixed Baseline V2.3 Suite

Output:

```text
D:\petsgaussianhair\_downloads\groom_density_v23_baseline_suite_20260624
```

V2.3 is the second fixed synthetic baseline after V2.2. It uses the longer-step
schedule validated above and runs the full 21-case hair-type suite, not only the
8-case probe.

Fixed V2.3 settings:

```text
projected_curve initialization
root movement
gradual root densification
periodic curl-phase interpolation for split children
flow coherence = 1.0
root-graph groom smoothness = 0.04
detail protection = 0.65
rendering = 1440 x 810
regular schedule = 4200 steps, densify until 2000, final roots 6648
hard schedule = 5200 steps, densify until 2800, final roots 11384
```

V2.3 improves every V2.2 case in the full suite:

| Group | Cases | Result |
| --- | --- | --- |
| Strong stable cases | `fine_undercoat`, `long_silky`, `side_parted`, `coarse_guard`, `short_plush`, `tiger_plush`, `mane_ridge`, `facial_short_to_long`, `dog_guard` | final PSNR is 36.10-47.79 |
| Improved mid cases | `clumped`, `wet_matted`, `frizzy`, `spiky_guard`, `wavy`, `patchy_length`, `ringlet_plus_undercoat`, `mixed_animal` | final PSNR is 34.00-35.58 |
| Remaining boundary cases | `cowlick_whorl`, `curly_ringlet`, `dirty_tangled`, `curled` | all improve over V2.2, but local whorl/curl/tangle remains the bottleneck |

Lowest V2.3 cases:

| Style | V2.2 PSNR | V2.3 PSNR | Delta | Decision |
| --- | ---: | ---: | ---: | --- |
| `curled` | 30.6060 | 31.1840 | +0.5780 | classic curl remains unresolved by root growth |
| `dirty_tangled` | 30.4944 | 31.4046 | +0.9102 | tangled local flow remains unresolved |
| `curly_ringlet` | 32.4782 | 33.3893 | +0.9111 | improved but still detail-limited |
| `cowlick_whorl` | 32.9344 | 33.4112 | +0.4768 | whorl flow is a head-region stress case |

Artifacts:

- `D:\petsgaussianhair\_downloads\groom_density_v23_baseline_suite_20260624\suite_summary.csv`
- `D:\petsgaussianhair\_downloads\groom_density_v23_baseline_suite_20260624\v23_vs_v22.csv`
- `D:\petsgaussianhair\_downloads\groom_density_v23_baseline_suite_20260624\v23_final_contact_sheet.png`

Conclusion:

V2.3 is the current fixed baseline for moving to white-tiger Stage 1. It is
strictly better than V2.2 across the 21-case suite and does not introduce a
new failure mode in ordinary animal fur. The remaining unresolved cases are
curl/tangle/whorl cases, which should be treated as the next research module
rather than a reason to keep delaying the white-tiger recovery run.

## Historical Baseline V2 Definition

This was the previous second baseline before the V2.3 sweep. It is kept for
traceability only; it is not the current fixed baseline for white-tiger Stage 1.

The historical V2 baseline was:

```text
projected_curve initialization
periodic curl-phase interpolation for split children
root movement enabled
gradual root densification
densify warmup = 160
densify interval = 80
densify until = 1200
max splits per event = 256
root-graph groom smoothness = 0.04
detail protection = 0.65
flow coherence = 1.0
total iterations = 2400
prune disabled unless the target contains true empty/island regions
```

Compared with the 1360-step smooth baseline, V2 gives consistent gains on
representative animal-fur cases:

- `mixed_animal`: 31.6058 -> 32.4423
- `dog_guard`: 32.9672 -> 33.6264
- `patchy_length`: 30.8768 -> 31.6272

The current next technical target is not another generic smoothness term.
The evidence points to high-frequency curve/orientation capacity: target-like
plush, long smooth, mane-ridge, coarse guard, wet matted, and clumped coats are
already stable.  Dense ringlet curls, classic curled hair, and dirty/tangled
coats improve with more roots and longer training, but not enough to close the
gap.  The next useful module should therefore target local curve/frizz
expressiveness or projected curl initialization, not generic smoothing or
unbounded root growth. Naive screen orientation-residual parent ranking,
higher curl learning rate, stronger orientation weighting, and weaker
smoothness were tested and rejected.

Summary sheets:

- `D:\petsgaussianhair\_downloads\groom_density_more_hair_types_v2c_20260624\v2c_hair_type_summary_sheet.png`
- `D:\petsgaussianhair\_downloads\groom_density_hard_curly_morecap_20260624\hard_curly_morecap_summary_sheet.png`

## V2.3 Extended Stress Suite

Purpose:

This is not a new baseline. It keeps the fixed V2.3 modules and schedule, then
tests harder style-density combinations before moving to the white-tiger Stage
1 run.

Enabled modules:

```text
projected_curve initialization
root movement
gradient-driven densification
periodic segment refresh
flow coherence = 1.0
root-graph groom smoothness = 0.04
detail protection = 0.65
1440 x 810 validation renders
regular schedule: 4200 iterations, densify until 2000
hard schedule: 5200 iterations, densify until 2800
```

Result summary:

| Style | Density | Final PSNR | Decision |
| --- | --- | ---: | --- |
| `fine_undercoat` | `matted_density` | 49.4443 | pass |
| `tiger_plush` | `prune_islands` | 49.4248 | pass |
| `short_plush` | `prune_islands` | 48.1617 | pass |
| `coarse_guard` | `prune_islands` | 47.4227 | pass |
| `long_silky` | `matted_density` | 44.9283 | pass |
| `side_parted` | `matted_density` | 43.5043 | pass |
| `tiger_plush` | `matted_density` | 40.7500 | pass |
| `dog_guard` | `matted_density` | 40.4573 | pass |
| `mixed_animal` | `matted_density` | 40.1130 | pass |
| `facial_short_to_long` | `matted_density` | 38.8658 | pass, head-transition stress |
| `spiky_guard` | `matted_density` | 38.1395 | pass, spiky/guard stress |
| `cowlick_whorl` | `matted_density` | 37.5691 | pass, whorl boundary case |

Observation:

Most hard density/style combinations recover strongly after densification stops.
Several cases temporarily drop near the end of the densification window, then
recover during the fixed-root stabilization phase. This supports using a
dedicated stabilization tail in the white-tiger Stage 1 script instead of
evaluating immediately after root growth.

Boundary:

`cowlick_whorl + matted_density` remains the clearest unresolved boundary. Its
final PSNR is above 37, but its orientation and flow coherence errors remain
higher than the other stress cases. This points to local curve/phase detail and
whorl-specific flow capacity as the next research target if the white-tiger
head region shows similar artifacts.

Artifacts:

- `D:\petsgaussianhair\_downloads\groom_density_v23_extended_stress_suite_20260624\suite_summary.csv`
- `D:\petsgaussianhair\_downloads\groom_density_v23_extended_stress_suite_20260624\extended_stress_final_contact_sheet.png`

## Frozen V2.3 Baseline

Current baseline for the next white-tiger Stage 1 run:

```text
config: D:\petsgaussianhair\configs\groom_density_v23_baseline_suite.json
validation stress config: D:\petsgaussianhair\configs\groom_density_v23_extended_stress_suite.json
server package: D:\petsgaussianhair\server_pull\anigroom_code_<timestamp>.clean.tar.gz
server package hash: D:\petsgaussianhair\server_pull\anigroom_code_<timestamp>.clean.tar.gz.sha256
```

The clean package must contain only code, configs, docs, scripts, and repository
metadata. It excludes `_downloads`, `outputs`, old `_stage1_check*` folders,
external data, server cache folders, and Git internals. The next server run
must use this package or a newer package with an explicit updated hash.

The frozen baseline keeps:

```text
projected_curve initialization
root movement
gradient-driven densification
periodic segment refresh
flow coherence = 1.0
root-graph groom smoothness = 0.04
detail protection = 0.65
regular schedule = 4200 iterations, densify until 2000
hard schedule = 5200 iterations, densify until 2800
```

White-tiger Stage 1 should use the high-capacity native route in
`D:\petsgaussianhair\scripts\server\run_white_tiger_stage1_recovery_highcap.sh`.
That route enforces native `1920 x 1080` input through the preflight script and
must not be replaced by the old 10k-root native gate.

## Run Discipline

Future variants must use a new output directory under `_downloads/`.
Do not modify or overwrite the fixed baseline folders above.

Long training output should be redirected to a `.log` file. The conversation
should only read compact summaries from `summary.json`; do not print full JSON
histories into the chat.

Example:

```powershell
python tools/train_groom_density_densification.py ... *> run.log
```
