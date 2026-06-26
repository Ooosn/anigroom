# Checkpoint A Status

Last updated: 2026-06-23.

## Current Server Environment

Development environment:

```text
name: 20260620113448
wpId: cc0055eb-3c3d-42d5-904f-eb378cf66442
gpu: 1 x NVIDIA-A100-SXM4-80GB
image: liuhaohan_w:deps20260602
```

Environment rules followed:

```text
image name remains liuhaohan_w
image tag remains deps20260602
memory was not changed
no new image name or project-named image was created
```

## Code State

The server project path is:

```text
/ssdwork/liuhaohan/petsgaussianhair
```

The active code directories were replaced with the clean local package:

```text
anigroom/
docs/
scripts/
tools/
README.md
.gitignore
.gitattributes
```

Preserved server directories:

```text
data/
data_sources/
external/
logs/
outputs/
```

The old smoke-test training and hand-written rendering files are not present in
the active server code path.

Code path audit:

```text
rg debug_torch_splat/splat_render/darkness/hairness/density:
  only appears in documentation sections that explicitly mark these items as
  removed or forbidden.

formal training path:
  anigroom/stage1/training.py imports gsplat.rendering.rasterization
  tools/render_anigroom_stage1_views.py calls render_stage1
  render_stage1 calls rasterize_stage1
  rasterize_stage1 calls gsplat.rendering.rasterization

baseline path:
  anigroom/baselines/plain_gsplat.py imports gsplat.rendering.rasterization
  tools/render_plain_gsplat_baseline_views.py imports gsplat.rendering.rasterization
```

Interpretation:

```text
The current Checkpoint A code path is a formal gsplat path. The removed
smoke-test renderer and unconfirmed channels are documented only as rejected
items and are not part of training, rendering, evaluation, or visualization.
```

## Data Preflight

Script:

```text
scripts/server/checkpoint_a_preflight.sh
```

Result:

```text
white_tiger_server_env_ok=1
preflight_ok=1
```

Data:

```text
data_root: /ssdwork/liuhaohan/petsgaussianhair/data/neuralfur_work/whiteTiger_processed/roaringwalk
mesh: /ssdwork/liuhaohan/petsgaussianhair/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj
images: 36
masks: 36
orientation angles: 36
orientation confidence: 36
resolution: 1920 x 1080
train views: 30
test views: 6
test_stride: 6
```

Fixed test indices:

```text
0, 6, 12, 18, 24, 30
```

## gsplat Backend

Problem found:

```text
The installed gsplat wheel imported, but CUDA was disabled because nvcc was not
on PATH. The installed backend set _C=None.
```

Resolution:

```text
Installed the local gsplat-1.1.1 wheel.
Patched gsplat/cuda/_backend.py with the existing cached-backend loader from
the RTS gsplat copy.
Loaded /root/.cache/torch_extensions/py310_cu118/gsplat_cuda/gsplat_cuda.so.
Committed the modified environment back to liuhaohan_w:deps20260602.
```

Reproducible script:

```text
scripts/server/install_gsplat_cached_backend.sh
```

Probe result:

```text
render_shape: (1, 64, 64, 3)
alpha_shape: (1, 64, 64, 1)
mean_grad_norm: 0.02280793897807598
gsplat_cached_backend_ok=1
```

## Plain gsplat Baseline

Purpose:

```text
Validate the data, camera, metric, and gsplat training path before AniGroom
roots/strands are introduced.
```

Script:

```text
scripts/server/run_plain_gsplat_baseline_short.sh
```

Run:

```text
output: /ssdwork/liuhaohan/petsgaussianhair/outputs/plain_gsplat_baseline_5000_1000
num_gaussians: 5000
iterations: 1000
eval_every: 100
save_every: 500
resolution: 1920 x 1080
```

Final metrics at iteration 1000:

```text
elapsed_sec: 76.013
peak_memory_mb: 244.39
train_psnr: 20.322509765625
train_mse: 0.009284299798309803
train_mask_l1: 0.04106789827346802
test_psnr: 20.43193817138672
test_mse: 0.009053287096321583
test_mask_l1: 0.0394776314496994
```

LPIPS(VGG) re-evaluation at iteration 1000:

```text
train_psnr: 20.591959126790364
train_ssim: 0.8565517882506053
train_mask_l1: 0.04106789343059063
train_lpips_vgg: 0.16928897524873415

test_psnr: 20.931286493937176
test_ssim: 0.8621561229228973
test_mask_l1: 0.03947762819007039
test_lpips_vgg: 0.16352875282367071
```

Baseline render export:

```text
output: outputs/plain_gsplat_baseline_5000_1000/render_test_001000
result: test-view contact sheet generated. The baseline obtains much higher
PSNR than current AniGroom, but visual inspection shows a blurry volumetric
body rather than editable fur parameters.
```

Baseline all-view export:

```text
server: outputs/plain_gsplat_baseline_5000_1000_render_all_001000/contact_sheet.jpg
local: D:\petsgaussianhair\_baseline_plain_gsplat_5000_1000\render_all_001000\contact_sheet.jpg
local LPIPS: D:\petsgaussianhair\_baseline_plain_gsplat_5000_1000\lpips_vgg_001000.json
```

Comparison to current AniGroom checkpoint:

```text
Plain gsplat baseline 001000:
  test_psnr: 20.931286493937176
  test_ssim: 0.8621561229228973
  test_mask_l1: 0.03947762819007039
  test_lpips_vgg: 0.16352875282367071

AniGroom Stage 1 globalcalib 005000:
  test_psnr: 27.725433349609375
  test_ssim: 0.9013419151306152
  test_mask_l1: 0.004981714766472578
  test_lpips_vgg: 0.1211653637389342

Interpretation:
  The plain gsplat baseline validates that the data/camera/metric path is
  functional, but its 36-view render is a blurry volumetric animal. AniGroom
  gives a stronger reconstruction while keeping explicit root/groom
  parameters.
```

## AniGroom Stage 1 Short Runs

The formal Stage 1 code path now runs through:

```text
mesh-rooted barycentric roots
explicit groom parameters
strand-to-Gaussian generation
gsplat rasterization
RGB / mask / orientation / graph-smooth losses
fixed train/test metrics
checkpoint and root-parameter export
test-view render export
```

Probe:

```text
output: outputs/anigroom_stage1_probe
roots: 300
strand_samples: 2
iterations: 5
result: forward/backward, metric logging, checkpoint export all pass
```

Short run v1, before graph/orientation:

```text
output: outputs/anigroom_stage1_10000_500_v1
roots: 10000
gaussians: 40000
iterations: 500
test_psnr: 16.227001190185547
test_ssim: 0.8442583084106445
test_mask_l1: 0.15999963879585266
peak_memory_mb: 543.15
visual: continuous but still gray/white fog-like fur; no clear stripes
```

Short run v2, graph smooth + orientation:

```text
output: outputs/anigroom_stage1_10000_500_v2_graph_orient
roots: 10000
gaussians: 40000
iterations: 500
test_psnr: 16.20789909362793
test_ssim: 0.8450592756271362
test_mask_l1: 0.16149356961250305
orientation_loss: 0.3758739233016968 -> 0.3132280111312866
peak_memory_mb: 575.76
visual: similar coverage to v1; orientation signal has numeric effect but is
not yet visually dominant at 500 iterations
```

Longer default run:

```text
output: outputs/anigroom_stage1_20000_2000_default
roots: 20000
gaussians: 80000
iterations: 2000
test_psnr: 16.394187927246094
test_ssim: 0.8338176012039185
test_mask_l1: 0.13558626174926758
visual: stripes begin to appear, but the result is still gray/fog-like and
legs/edges contain broken sparse samples. Width, length, and opacity shrink
during training, so long training improves mask but hurts structure clarity.
```

Foreground/boundary weighted reconstruction loss:

```text
output: outputs/anigroom_stage1_10000_500_v5_weighted
roots: 10000
gaussians: 40000
iterations: 500
test_psnr: 16.243881225585938
test_ssim: 0.8458697199821472
test_mask_l1: 0.16205370426177979
effect: keeps length/width/opacity from collapsing as quickly; slightly
improves 10k/500 metrics over v2.

output: outputs/anigroom_stage1_20000_2000_weighted
roots: 20000
gaussians: 80000
iterations: 2000
test_psnr: 16.39181137084961
test_ssim: 0.8334776759147644
test_mask_l1: 0.14138294756412506
effect: similar final PSNR to the unweighted 20k/2000 run, better parameter
collapse behavior, but still drifts toward silhouette fitting after 500-1000
iterations.
```

Decision:

```text
foreground/boundary weighted losses are kept. They are directly tied to the
white tiger foreground and confirmed by metrics/parameter behavior. Longer
training still needs additional structure constraints or early stopping.
```

Root evidence accounting:

```text
output: outputs/anigroom_stage1_evidence_probe
roots: 1000
iterations: 5
logged fields:
  visible_root_fraction
  visible_samples_per_step_mean
  grad_abs_mean
  grad_abs_p95_visible
  grad_abs_max
result: evidence logging works with gsplat radii and retained per-Gaussian
mean gradients. Densification/pruning should be built on this, not on random
root resampling.
```

Color initialization experiment:

```text
full multi-view color write:
  output: outputs/anigroom_stage1_10000_500_v3_color
  test_psnr: 15.800822257995605
  test_ssim: 0.8384369611740112
  result: visible stripes, but darker/less accurate reconstruction

0.35 blended color hint:
  output: outputs/anigroom_stage1_10000_500_v4_colorblend
  test_psnr: 16.03670310974121
  test_ssim: 0.8426181674003601
  result: better than full write, still worse than v2
```

Decision:

```text
color initialization from projected views is kept as an optional experiment,
but is disabled by default. It needs view/visibility filtering before it can be
used as a reconstruction default. The current default is v2-style training
without color initialization.
```

Evidence-based root densification:

```text
output: outputs/stage1_20260623_10000r_2000i_densify_v2
roots: 10000 -> 10216
iterations: 2000
structure interval: 500
densify threshold: 0.001 evidence per visible Gaussian sample
added roots:
  iter 500: 20
  iter 1000: 86
  iter 1500: 76
  iter 2000: 34
test_psnr: 16.67959213256836
test_ssim: 0.8299156427383423
test_mask_l1: 0.1293644905090332
local render: D:\petsgaussianhair\_stage1_check_10000_2000_densify_v2\contact_sheet.jpg
local parameters: D:\petsgaussianhair\_stage1_check_10000_2000_densify_v2\parameter_contact_sheet.jpg
```

Decision:

```text
The evidence-based mechanism is kept. It is not a fixed-percent sampler:
parents are selected only when normalized per-root Gaussian mean-gradient
evidence exceeds a threshold. It improves PSNR and mask fit, but the current
visual output is still gray/foggy and cannot be used as the display asset.
The next bottleneck is opacity/width/color structure, not root-count plumbing.
```

Rejected curve experiment:

```text
output: outputs/stage1_20260623_10000r_2000i_densify_hermite_v1
change: migrated the old smoke-test Hermite strand curve into the formal
gsplat route
result at iter 1000:
  test_psnr: 16.193784713745117
  test_ssim: 0.8361680507659912
  test_mask_l1: 0.15335969626903534
comparison: worse than densify_v2 at the same iteration
decision: do not keep this change in the mainline
```

Longer mainline run:

```text
output: outputs/stage1_20260623_10000r_5000i_densify_main_v1
roots: 10000 -> 10267
gaussians: 40000 -> 41068
iterations: 5000
mesh backing: disabled
densify threshold: 0.001 evidence per visible Gaussian sample
test_psnr:
  iter 2000: 16.670642852783203
  iter 3000: 16.86288070678711
  iter 4000: 16.961257934570312
  iter 5000: 17.01410484313965
test_ssim:
  iter 5000: 0.8456823229789734
test_mask_l1:
  iter 5000: 0.10904263705015182
local render: D:\petsgaussianhair\_stage1_check_10000_5000_densify_main_v1\contact_fur_only.jpg
local parameters: D:\petsgaussianhair\_stage1_check_10000_5000_densify_main_v1\parameter_contact_sheet.jpg
```

Conclusion:

```text
The formal gsplat route is trainable beyond 2000 iterations and continues to
improve reconstruction metrics. Visual inspection shows clearer stripe/detail
than 2000 iterations, but also reveals transparent/empty regions around the
belly and legs. The current bottleneck is therefore not root evidence logging
or training stability; it is the coverage/opacity/strand-structure behavior of
the reconstruction parameterization.
```

Mesh backing and width probes:

```text
backing_v1:
  output: outputs/stage1_20260623_10000r_2000i_densify_backing_v1
  mesh backing: 40000 training-only surface Gaussians with random colors
  test_psnr at 2000: 16.539405822753906
  test_ssim at 2000: 0.8366179466247559
  result: useful diagnostic for white-on-white transparency, but not better
  than the no-backing mainline in reconstruction metrics.

thin init:
  output: outputs/stage1_20260623_10000r_1000i_backing_thin_v1
  init_root_width: 0.004
  init_tip_width: 0.0012
  test_psnr at 1000: 16.365596771240234
  result: too thin; coverage becomes broken and transparent.

mid-width init:
  output: outputs/stage1_20260623_10000r_1000i_backing_midwidth_v1
  init_root_width: 0.008
  init_tip_width: 0.003
  test_psnr at 1000: 16.5085506439209
  result: slightly better than thin, but still visually gray/foggy. Width
  tuning alone is not the core fix.
```

Opacity-floor probe:

```text
output: outputs/stage1_20260623_10000r_3000i_densify_opfloor065_v1
opacity_floor: 0.65
opacity_floor_weight: 0.35
test_psnr at 3000: 16.72258949279785
test_ssim at 3000: 0.8232991099357605
test_mask_l1 at 3000: 0.12317433208227158
comparison: worse than the no-floor mainline at 3000
result: keeping opacity high globally reduces the empty-region artifact only
partly and introduces speckled/noisy fur. It is not a default solution.
```

Head/detail weighted reconstruction probe:

```text
output: outputs/stage1_20260623_10000r_5000i_headdetail_v1
detail_rgb_weight: 2.0
head_rgb_weight: 2.0
head_mask_weight: 1.0
head_axis_start: 0.78
head_projection_radius: 41
weight_debug_every: 1000
roots: 10000 -> 10279
test_psnr:
  iter 2000: 16.678970336914062
  iter 3000: 16.870336532592773
  iter 4000: 16.96803855895996
  iter 5000: 17.02278709411621
test_ssim at 5000: 0.8468822240829468
test_mask_l1 at 5000: 0.10852858424186707
local render: D:\petsgaussianhair\_stage1_check_10000_5000_headdetail_v1\contact_fur_only.jpg
local parameters: D:\petsgaussianhair\_stage1_check_10000_5000_headdetail_v1\parameter_contact_sheet.jpg
local weight debug: D:\petsgaussianhair\_stage1_check_10000_5000_headdetail_v1\weight_debug_005000.jpg
```

Conclusion:

```text
The detail map correctly emphasizes stripe/edge pixels, and the projected
head map lands on the head/front-leg region rather than the tail or
background. The module gives a small positive metric change over the mainline
5000-step run, but visual inspection is still very close to the mainline and
does not solve the belly/leg transparency problem. Keep it as formal logging
and analysis support, but do not treat it as the main quality fix.
```

Mesh-to-camera calibration probe:

```text
problem found:
  The current furless mesh projection is not aligned with the GT silhouette.
  A mesh-surface alpha probe using both cameras_extr.npy and GaussianHaircut's
  cameras.npz decomposition gives the same result: the mesh projects too low
  and too small, especially in side views.

camera convention result:
  cameras_extr.npy and cameras.npz decomposition are equivalent for this data.
  Transposed view matrices are wrong.

diagnostic:
  before calibration, side test views have more than 50% of foreground pixels
  with alpha < 0.5.
  after calibration, the same views drop to about 0.15% foreground pixels with
  alpha < 0.5.

formal fix:
  Add a learnable global mesh-to-camera similarity calibration to Stage 1:
  scalar scale + 3D translation before strand-to-Gaussian generation.
  This keeps roots on the mesh and keeps groom parameters interpretable, but
  lets the asset align to the provided camera/mask coordinate frame.

output:
  outputs/stage1_20260623_10000r_2000i_globalcalib_v1

metrics:
  iter 500 test_psnr: 21.39668846130371
  iter 1000 test_psnr: 23.824508666992188
  iter 1500 test_psnr: 25.43743896484375
  iter 2000 test_psnr: 26.472793579101562
  iter 2000 test_ssim: 0.8919357657432556
  iter 2000 test_mask_l1: 0.005958186462521553

learned calibration at 2000:
  global_scale: 1.2556742429733276
  global_translation_norm: 0.32384106516838074

local render:
  D:\petsgaussianhair\_stage1_check_10000_2000_globalcalib_v1\contact_sheet.jpg
  D:\petsgaussianhair\_stage1_check_10000_2000_globalcalib_v1\alpha_diagnostics.jpg
```

Conclusion:

```text
The main failure in the previous Stage 1 runs was mesh/camera silhouette
misalignment, not merely low root density or white-fur transparency. The
global calibration module is now part of the formal Stage 1 route. Coverage,
densification, and parameter smoothing should be evaluated after this
calibration is active.
```

## Next Required Work

Global-calibration 5000-step follow-up:

```text
output: outputs/stage1_20260623_10000r_5000i_globalcalib_v1
roots: 10000 -> 10055
gaussians: 40000 -> 40220
iterations: 5000
mesh backing: disabled
global transform: learnable scalar scale + translation
densify threshold: 0.001 evidence per visible Gaussian sample
```

Metrics:

```text
iter 1000 test_psnr: 23.824752807617188
iter 2000 test_psnr: 26.423702239990234
iter 3000 test_psnr: 27.34708023071289
iter 4000 test_psnr: 27.750234603881836
iter 5000 test_psnr: 27.725433349609375

iter 5000 test_ssim: 0.9013419151306152
iter 5000 test_mask_l1: 0.004981714766472578
iter 5000 train_psnr: 27.725561141967773
iter 5000 train_lpips_vgg: 0.12506301825245222
iter 5000 test_lpips_vgg: 0.1211653637389342
```

Learned calibration at 5000:

```text
global_scale: 1.2570874691009521
global_translation_norm: 0.32284078001976013
```

Alpha diagnostics:

```text
checkpoint 004000 foreground alpha<0.5 fractions:
  view 0000: 0.0013966870582977178
  view 0006: 0.0017349543270612428
  view 0012: 0.0028257790531907275
  view 0018: 0.0005158361704322707
  view 0024: 0.0020307013764133703
  view 0030: 0.0021085533926755054

checkpoint 005000 foreground alpha<0.5 fractions:
  view 0000: 0.0011061761501717925
  view 0006: 0.0014645994810038468
  view 0012: 0.0021025829422760115
  view 0018: 0.0004561077717506394
  view 0024: 0.0016661031967526456
  view 0030: 0.001890276643951333
```

Local renders:

```text
D:\petsgaussianhair\_stage1_check_10000_5000_globalcalib_v1\render_test_004000\contact_sheet.jpg
D:\petsgaussianhair\_stage1_check_10000_5000_globalcalib_v1\render_test_004000\alpha_diagnostics.jpg
D:\petsgaussianhair\_stage1_check_10000_5000_globalcalib_v1\render_test_005000\contact_sheet.jpg
D:\petsgaussianhair\_stage1_check_10000_5000_globalcalib_v1\render_test_005000\alpha_diagnostics.jpg
D:\petsgaussianhair\_stage1_check_10000_5000_globalcalib_v1\render_all_005000\contact_sheet.jpg
D:\petsgaussianhair\_stage1_check_10000_5000_globalcalib_v1\render_all_005000\stage1_005000_all_views.mp4
D:\petsgaussianhair\_stage1_check_10000_5000_globalcalib_v1\render_all_005000\stage1_005000_all_views.gif
```

Parameter visualizations:

```text
D:\petsgaussianhair\_stage1_check_10000_5000_globalcalib_v1\params_005000\parameter_contact_sheet.jpg
D:\petsgaussianhair\_stage1_check_10000_5000_globalcalib_v1\params_005000\parameter_overview_readable.jpg
D:\petsgaussianhair\_stage1_check_10000_5000_globalcalib_v1\params_005000\parameter_manifest.json
```

LPIPS record:

```text
server: outputs/stage1_20260623_10000r_5000i_globalcalib_v1/lpips_vgg_005000.json
local: D:\petsgaussianhair\_stage1_check_10000_5000_globalcalib_v1\lpips_vgg_005000.json
note: LPIPS is computed with the VGG trunk because the server already has the
VGG weights available. The AlexNet cache contains only a partial file.
```

Current checkpoint decision:

```text
Use checkpoint_005000.pt as the current Stage 1 checkpoint. Checkpoint 004000
has slightly higher PSNR by about 0.025 dB, but 005000 has cleaner alpha
coverage and visually similar fur/stripe quality.
```

Updated conclusion:

```text
The current formal route should continue. The global calibration turns the
previous 17 PSNR headdetail run into a stable 27+ PSNR reconstruction with
almost no foreground alpha holes. The remaining problems are now real grooming
quality issues: head/face clarity, strand-level fur feel, color sharpness, and
asset-mode smoothness, not gross silhouette alignment.

The 36-view render is stable enough for Checkpoint A inspection. It shows
continuous body, back, tail, and leg coverage, but the head is still softer
than desired. The parameter overview confirms that the formal channels are
present and interpretable, but also shows high-frequency point noise in the
learned parameter maps. That noise should be addressed in the next module pass
with stronger mesh-graph/detail-aware smoothing, not with unconfirmed
hairness/density/darkness channels.
```

Core mechanism evidence:

```text
mesh-graph smooth:
  active in the formal loss as graph_smooth.
  005000 regularizer graph_smooth: 0.16920463740825653.
  It keeps the mechanism connected to mesh-root neighborhoods, but the
  parameter overview still shows visible point noise; this needs stronger
  asset/reconstruction smoothing in the next module pass.

orientation / flow loss:
  active in the formal loss as orientation_loss.
  Earlier graph/orientation probe decreased orientation_loss from about
  0.3759 to 0.3132 at 500 iterations.
  005000 orientation_loss: 0.30497846007347107.
  This proves the signal is wired and trainable; visual flow quality still
  needs a dedicated overlay/debug view in the next pass.

head / boundary / detail weighting:
  active in 005000 training.
  005000 loss weight stats:
    boundary_mean: 0.03117847628891468
    detail_mean: 0.09063071757555008
    detail_p95: 0.575912594795227
    head_mean: 0.05649064481258392
    head_max: 1.0
    rgb_weight_mean: 2.325953960418701
    mask_weight_mean: 2.178832530975342
  Weight debug images are generated during the headdetail/globalcalib runs.
  The head is still visually soft, so this mechanism is necessary but not
  sufficient.

root densification:
  active in 005000 training through root evidence.
  root count: 10000 -> 10055.
  added roots by structure update:
    iter 500: 10
    iter 1000: 5
    iter 1500: 3
    iter 2000: 7
    iter 2500: 1
    iter 3000: 11
    iter 3500: 7
    iter 4000: 8
    iter 4500: 3
    iter 5000: 0
  final evidence_per_visible_sample_max: 0.0009867316111922264, below the
  0.001 threshold, so densification naturally stops at the end.

root pruning:
  main 005000 run triggered prune checks but removed no roots because roots
  remained visible/high-opacity.
  mechanism probe:
    server: outputs/stage1_prune_mechanism_probe_20260623/prune_probe.json
    local: D:\petsgaussianhair\_stage1_prune_mechanism_probe_20260623\server_output\prune_probe.json
    root_before: 300
    root_after: 150
    graph_edges_before: 1061
    graph_edges_after: 550
  This confirms tensor slicing, graph rebuild, root-parameter export, and
  post-prune gsplat rendering work. It is not a quality result.
```

1. Improve head/face and stripe detail after calibration. Use the current
   calibrated checkpoint as the baseline for these changes.
2. Revisit densification after calibration. Current threshold-based
   densification adds only a few roots once the silhouette is aligned, so the
   next evidence should focus on detail/stripe/head regions rather than fixing
   gross mask holes.
3. Improve coverage/opacity/width behavior only after mesh calibration is on.
   Do not use a global opacity floor.
4. Decide whether mesh backing remains a training diagnostic or becomes a
   controlled regularizer; current backing is not better than the no-backing
   mainline.
5. Improve parameter visualizations for presentation, but keep them secondary
   to the actual reconstruction quality.
