# White Tiger Stage 1 Long-Streak Diagnosis

Date: 2026-06-26

This note records the current diagnosis for the view09 white-tiger Stage 1
baseline. It does not change the baseline implementation.

## Symptom

The current single-view baseline reaches a reasonable composite PSNR, but the
rendered fur contains long dark streaks. The artifact looks like black tiger
stripes are being explained by long, curved, high-opacity fur strands rather
than by local color variation on short fur.

Reference run:

`D:\petsgaussianhair\server_pull\white_tiger_stage1_server_20260626070424\metrics.jsonl`

At iteration 3000:

- raw PSNR: 31.505
- composite PSNR: 32.618
- RGB L1: 0.01956
- orientation loss: 0.4541
- orientation detail loss: 0.8234
- root count: 18,341
- Gaussian count: 2,162,315

The metric is not the main issue. The image is visually wrong because some
strands become too long, soft, and high-opacity.

## Evidence From Current Baseline

Current dense groom range in
`D:\petsgaussianhair\tools\train_white_tiger_stage1.py`:

```text
length=(0.040, 0.220)
curl_radius=(0.0, 0.026)
curl_frequency=(0.0, 5.5)
frizz=(0.0, 0.010)
child_radius=(0.0, 0.012)
clump_strength=(0.0, 1.0)
opacity=(0.05, 0.98)
```

At iteration 3000, the learned parameter tails are:

```text
length mean 0.0691, p95 0.1057, max 0.2177
stiffness mean 0.4543, min 0.0675
curl_radius p95 0.00939, max 0.02587
frizz p95 0.00203, max 0.00813
child_radius p95 0.00583, max 0.01089
clump_strength mean 0.0779
opacity p95 0.9781
tip_opacity p95 0.8751
```

This combination is exactly the failure mode seen in the render:

- a few roots can grow very long strands;
- low stiffness and nonzero curl/frizz make them curve or wobble;
- low clump strength lets child strands spread instead of staying tight;
- high opacity makes dark strands occlude nearby white fur;
- RGB reconstruction then rewards this because black stripes can be painted by
  long strands.

The formal config also makes RGB the dominant term:

```text
RGB_WEIGHT=1.0
ORIENTATION_WEIGHT=0.02
ORIENTATION_DETAIL_WEIGHT=0.02
SMOOTH_WEIGHT=0.025
STRAND_SHAPE_SMOOTH_WEIGHT=0.012
SHAPE_PRIOR_WEIGHT=0.015
```

So the current objective lets RGB pull geometry more strongly than orientation
and shape priors can keep it hair-like.

## NeuralFur / GaussianHaircut Contrast

NeuralFur's released GaussianHaircut path avoids this failure with several
strong constraints:

1. It does not mainly optimize RGB for fur shape.
   `simple_run_panda.sh` sets orientation, mask, chamfer, SDF, shape, and
   gravity losses, while `lambda_dl1` defaults to 0 in the argument parser.

2. Length is region-controlled.
   `metrical_panda_furless_15k_small.yaml` has a `mapping_length` table for
   body, face, legs, tail, paws, etc. The strand generator normalizes decoded
   arc length and multiplies by region length. Length does not freely grow to
   explain stripes.

3. Shape comes from a pretrained strand prior.
   `optimizable_textured_fur.py` loads `strand_ckpt.pth`; the decoder is frozen
   and only texture/latent parameters are optimized. This restricts shape to a
   hair-like manifold.

4. Each Gaussian is a thin strand segment.
   `gaussian_model_latent_strands.py` places Gaussians at adjacent strand
   segment midpoints, sets the long axis from segment length, uses a small
   global transverse `strand_scale`, and returns opacity 1. This produces
   sharp line-like fur instead of soft free blobs.

5. It uses SDF/chamfer/shape/gravity constraints:
   - SDF: prevents penetration into the body.
   - Chamfer: attracts strands toward outer surface.
   - Shape consistency: penalizes inconsistent curvature profiles.
   - Gravity/region directions: keeps body-region flow plausible.

## Root Cause

The long dark streaks are not primarily a mesh alignment or depth-clipping bug.
They are an objective/parameterization failure:

```text
RGB-dominant optimization
  + too-large length range
  + high-opacity strand color shared along the whole strand
  + weak orientation/shape constraints
  + low clump strength and wide child spread
  -> dark stripes become long geometry strokes
```

The current baseline can raise PSNR while hurting visual hair quality because
RGB loss accepts "black paint strokes" as a cheap explanation for tiger stripes.

## Repair Direction Without Polluting Baseline

The next experiment should be a separate config/branch/run, not a baseline code
mutation. The purpose is to prove the root cause by making a
NeuralFur-style sharp-fur variant.

Recommended controlled changes:

1. Restrict long-tail geometry.
   - Reduce `length` upper bound or add a stronger prior above 0.075-0.10.
   - Increase stiffness prior or lower the allowed low-stiffness tail.

2. Reduce high-frequency curve freedom early.
   - Lower LR for bend/curl/frizz.
   - Freeze or heavily damp curl/frizz for the first training phase.
   - Raise `strand_shape_smooth_weight` to actually affect the objective.

3. Keep child strands tighter.
   - Lower `child_radius`.
   - Raise initial/minimum `clump_strength`.

4. Make shape supervision stronger than stripe painting.
   - Lower RGB weight temporarily, or delay strong RGB until alpha/mask/orient
     are stable.
   - Increase orientation and orientation-detail weights for the geometry phase.

5. Keep color local.
   - Avoid letting one dark root color dominate a long visible strand.
   - If needed, separate geometry fitting from color fitting: first shape,
     then color.

Verification gate:

- The same view09 crop should show fewer long black streaks.
- `length max` and `length p95` should stay bounded.
- `stiffness` should not collapse.
- `clump_strength` should not collapse toward zero.
- composite PSNR should not be the only pass criterion; strand-line visualization
  must look shorter, sharper, and less chaotic.

