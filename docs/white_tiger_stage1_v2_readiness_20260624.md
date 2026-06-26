# White Tiger Stage 1 V2 Readiness 2026-06-24

## Current Verified Baseline

The second diagnostic baseline is fixed as `V2.1-long convergence baseline`.
It is defined in `docs/groom_densification_experiments_20260624.md`.

Core settings:

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
```

Representative evidence:

| Case | Previous | V2-long | Gain |
| --- | ---: | ---: | ---: |
| `mixed_animal` | 31.6058 | 32.4423 | +0.8365 |
| `dog_guard` | 32.9672 | 33.6264 | +0.6592 |
| `patchy_length` | 30.8768 | 31.6272 | +0.7504 |

Additional target-relevant coverage:

| Case | V2-long PSNR | Meaning |
| --- | ---: | --- |
| `tiger_plush` | 34.7130 | white-tiger-like short plush / stripe coat is stable |
| `long_silky` | 38.4707 | long smooth sagging fur is stable |
| `mane_ridge` | 34.7786 | localized long ridge / mane behavior is stable |
| `coarse_guard` | 35.6501 | coarse guard-hair width / stiffness variation is stable |
| `curly_ringlet` | 30.9226 | dense ringlet curls remain a known hard case |

V2.1 phase-interpolation regression:

| Case | V2-long | V2.1 phase-interp | Decision |
| --- | ---: | ---: | --- |
| `tiger_plush` | 34.7130 | 34.9660 | keep; target-like plush/stripe fur improves |
| `dog_guard` | 33.6264 | 33.7695 | keep; guard-hair case improves |
| `mixed_animal` | 32.4423 | 32.3793 | neutral |
| `clumped` | 33.7617 | 33.6763 | neutral |
| `curled` | 29.5899 | 29.5795 | neutral; does not solve curled hair |
| `curly_ringlet` | 30.9226 | 30.8921 | neutral; does not solve ringlets |

Extended V2.1 stress suite with existing modules enabled:

```text
output = D:\petsgaussianhair\_downloads\groom_density_v21_extended_suite_20260624
resolution = 1440 x 810
regular cases = 3200 steps, densify until 1600, final roots 5368
hard cases = 3600 steps, densify until 2000, final roots 8184
modules = projected_curve init, root move, gradual densify, flow coherence,
          root-graph groom smoothness, detail protection, circular phase split
```

| Case | Final PSNR | Meaning |
| --- | ---: | --- |
| `tiger_plush` | 36.2670 | white-tiger-like plush/stripe fur remains strong under longer training |
| `short_plush` | 36.6880 | short dense coat is stable |
| `dog_guard` | 35.2360 | guard-hair zones are stable |
| `coarse_guard` | 37.1711 | width/taper/stiffness variation is stable |
| `long_silky` | 39.7933 | long smooth sagging fur is stable |
| `mane_ridge` | 36.0230 | localized long ridge/mane behavior is stable |
| `wet_matted` | 34.1682 | wet/clumped sag recovers from poor projection |
| `clumped` | 34.8185 | clump controls are effective |
| `frizzy` | 34.2358 | frizz can fit image quality, though orientation residual remains high |
| `curly_ringlet` | 32.4782 | hard; ringlet curl needs stronger local curve capacity |
| `curled` | 30.6060 | hard; classic curled hair remains unsolved |
| `dirty_tangled` | 30.4944 | hard; high-frequency tangled flow remains unsolved |

This extended suite strengthens the white-tiger Stage 1 decision: the target
coat family is covered by the current baseline, while hard curl/tangle behavior
is a separate V3 research target.

Known limitation:

`dirty_tangled`, `curly_ringlet`, and classic `curled` hair remain hard. More
capacity helps these cases, but naive screen orientation-residual parent
ranking, local projected attribute initialization, higher curl learning rate,
stronger orientation weighting, and weaker smoothness were rejected. This is
recorded as a V3 research target around local curve/frizz expressiveness, not
a blocker for the white-tiger Stage 1 reconstruction baseline.

## White Tiger Data Preflight

Local white tiger input report:

```text
data_root = D:\petsgaussianhair\data\neuralfur_work\whiteTiger_processed\roaringwalk
mesh_path = D:\petsgaussianhair\data_sources\neuralfur_official_results\whiteTiger\furless_reshaped.obj
image_count = 36
mask_count = 36
orientation_angle_count = 36
orientation_conf_count = 36
resolution = 1920 x 1080
train views = 30
test views = 6
split = fixed stride 6
errors = []
```

The saved report is:

```text
D:\petsgaussianhair\_downloads\white_tiger_stage1_input_report_20260624.json
```

## Current Gap Before Server Training

The clean codebase now has a native-resolution white tiger Stage 1 entry:

```text
tools/train_white_tiger_stage1_v2.py
scripts/server/run_white_tiger_stage1_v2_native.sh
```

Verified local gate:

```text
output = D:\petsgaussianhair\_downloads\white_tiger_stage1_v2_native_entry_gate_20260624
resolution = 1920 x 1080
root_count = 512
iterations = 1
train views = 30
test views = 6
gsplat render/backward = ok
max_memory_mb = 600.88
```

This gate is only an entry validation, not a quality result.

The new entry already uses:

```text
anigroom.data.white_tiger for paths, split, and preflight
anigroom.mesh_roots for FPS mesh-root initialization
anigroom.grooming for explicit groom parameters and strand-to-Gaussian conversion
gsplat rasterization only
native 1920 x 1080 resolution requirement
RGB / mask / projected orientation / root-graph smoothness losses
train and test metric logging
```

Dynamic root lifecycle has now been ported into the native white-tiger Stage 1
entry:

```text
root/Gaussian gradient signal accumulation
thresholded parent selection
topology-local split candidates on mesh faces
parent-replace split
periodic curl-phase interpolation for child groom parameters
optimizer rebuild after root-count changes
checkpointed lifecycle history
```

Verified local native lifecycle gate:

```text
output = D:\petsgaussianhair\_downloads\white_tiger_stage1_v21_lifecycle_gate_fields_20260624
resolution = 1920 x 1080
root_count = 64
iterations = 2
forced densify at iteration 2
selected_parent_count = 8
inserted_child_count = 16
prune_count = 8
root_count_after = 72
checkpoint root tensors = 72 rows
```

Remaining gap before claiming it as the final V2.1 white-tiger baseline:

```text
run a realistic root-count gate on the Westlake development environment;
then run a long native-resolution white-tiger Stage 1 training job and compare
against the old 33-level result.
```

The old high-PSNR trainer still exists in git history as
`tools/train_white_tiger_uv_groom.py`, but it is not acceptable to restore it
blindly:

- it is large and mixes several old routes;
- it includes deleted modules such as `anigroom.mesh`, `anigroom.stage_a`, and
  `anigroom.preview`;
- it contains old UV texture / backing / diagnostic paths that should not
  silently define the new baseline;
- it previously ran at lower resolution in some checks and must not be used to
  claim the 1920 Stage 1 result without a clean audit.

## Required Next Step

1. Run a short native-resolution server gate with realistic root count.
2. Start the Westlake development-environment training through
   `scripts/server/run_white_tiger_stage1_v2_native.sh`.
3. Track PSNR against the old 33-level result and only freeze the baseline
   after the real white-tiger Stage 1 run proves it.

## 2026-06-24 Native Long-Run Result

The Westlake native-resolution V2.1 run was submitted as training task
`20260624083833` and stopped intentionally after the diagnostic signal was
clear.

Observed result:

```text
resolution: 1920 x 1080
train views: 30
test views: 6
root_count: 10000 -> 13072 by 12000 iterations
gaussian_count: about 472k at 13000 iterations
test PSNR at 13000: 21.5557
test SSIM at 13000: 0.8260
```

Decision:

```text
Do not freeze this as the white-tiger Stage 1 baseline.
Do not continue this configuration to 30000 iterations.
```

Reason:

```text
The run is numerically stable and the root lifecycle works, but the
configuration is still far below the old 33-level UV-groom result and even
below the 27-level global-calibration route. The missing pieces are not another
generic smoothness term; the route lacks the high-capacity/root-density,
projected initialization, adaptive strand sampling, head/detail weighting,
random mesh backing, and graph/flow regularization stack recorded in the
recovery audit.
```

The V2.1 synthetic hair-type suite remains useful as module evidence, but the
white-tiger reconstruction baseline must follow the recovery route rather than
this minimal native entry.
