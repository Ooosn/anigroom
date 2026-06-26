# AniGroom Stage 1 Module Decisions

This document records which mechanisms are kept, removed, or still undecided
for Checkpoint A. It complements `stage1_confirmed_framework.md`.

The previous implementation is treated as a smoke test. A mechanism can be
kept only when it has a clear grooming or reconstruction role and can be
rebuilt cleanly on top of `gsplat`.

## Decision Categories

```text
Keep:
  confirmed for the formal Stage 1 implementation.

Remove:
  must not appear in the formal code path.

Rebuild:
  the idea is useful, but the old implementation is not trusted.

Defer:
  potentially useful later, but not part of Checkpoint A.
```

## 1. Rendering

Decision: Keep `gsplat`; remove the hand-written renderer.

Reference:

```text
gsplat
GaussianHaircut / 3DGS-style splatting, used only as renderer context
```

Reason:

```text
AniGroom needs standard differentiable Gaussian rasterization, not a custom
debug renderer. Root and groom parameters should generate Gaussian means,
scales, rotations, colors, and opacities in PyTorch; `gsplat` should handle
rasterization and return gradients.
```

Remove:

```text
splat_render
debug_torch_splat
any training path that does alpha compositing by hand
```

Verification:

```text
plain gsplat baseline runs
AniGroom Stage 1 uses the same gsplat wrapper
no training script imports old hand-written renderer symbols
```

## 2. Data, Camera, and Metrics

Decision: Keep one unified data/evaluation path.

Reason:

```text
The baseline and AniGroom must use the same images, masks, cameras,
train/test split, resolution, and metric code. Otherwise the comparison is not
meaningful.
```

Formal requirements:

```text
fixed train/test split
no silent image resizing during metric computation
training-only randomization disabled for evaluation
PSNR / SSIM / LPIPS logged with image count and resolution
```

Verification:

```text
camera projection debug image
single-view overfit
plain gsplat baseline metric table
AniGroom metric table using the same evaluator
```

## 3. Mesh-to-Camera Calibration

Decision: Keep a learnable global mesh-to-camera similarity calibration.

Reason:

```text
AniGroom roots are constrained to the body mesh. Unlike a free 3DGS baseline,
they cannot independently move the whole animal to compensate for an input
mesh/camera mismatch. The current white-tiger `furless_reshaped.obj` projects
too low and too small under the provided camera/mask frame. If this mismatch
is left unresolved, the model wastes fur opacity, width, and densification on
gross silhouette alignment instead of learning grooming parameters.
```

Keep:

```text
global scalar scale
global 3D translation
regularization toward identity
logging and checkpoint/export of the learned transform
```

Remove:

```text
screen-space shifts
per-view camera hacks
silent camera convention changes
full nonrigid body warp for Checkpoint A
```

Verification:

```text
camera convention probe shows cameras_extr.npy and cameras.npz decomposition
are equivalent; transposed view matrices are wrong.

Before calibration, side test views have more than 50% foreground pixels with
alpha < 0.5 in a mesh-surface alpha probe.

After calibration, the same foreground-hole fraction drops to about 0.15%,
and the 2000-step Stage 1 run improves to test PSNR 26.47, SSIM 0.8919, and
mask L1 0.00596.
```

## 4. Mesh Roots

Decision: Keep face-local barycentric roots for Checkpoint A.

Reference:

```text
standard mesh surface parameterization
3DGS mean optimization path through generated Gaussian means
```

Reason:

```text
Each root must stay on the mesh. Optimizing barycentric logits inside the
current face gives a differentiable and stable first version. Cross-face root
motion is useful but not required when initial roots are dense and
densification can add local capacity.
```

Keep:

```text
face_id
barycentric logits
x_root = barycentric interpolation of the face vertices
```

Defer:

```text
cross-face differentiable root sliding
latent-controlled root xyz
```

Verification:

```text
all barycentric coordinates are valid
root positions remain on the globally calibrated mesh
root gradients are nonzero in short training
```

## 5. Root Initialization

Decision: Rebuild root initialization as dense mesh sampling plus uniform
selection.

Reason:

```text
The formal system should not depend on raw mesh vertices or fragmented UV
layout. Candidate roots should be sampled on mesh faces, then selected to give
roughly uniform surface coverage.
```

Keep:

```text
dense face sampling
approximately uniform selection, e.g. FPS or a fast approximation
head/detail-biased allocation only as a controlled option
```

Remove:

```text
UV-only root placement
global residual-only root sampling as the main initializer
```

Verification:

```text
root projection visualization
root surface coverage statistics
head/body/tail coverage report
```

## 6. Groom Parameters

Decision: Keep only parameters with direct grooming meaning.

Confirmed parameters:

```text
length
root width
tip width
flow direction
bend / sag
stiffness
color
opacity
```

Reason:

```text
These parameters map to common grooming controls and can be visualized,
edited, and regularized. They are enough for the first white-tiger Stage 1
prototype.
```

Remove:

```text
density / hairness as an independent parameter
darkness
carrier/base-fur decomposition
unexplained texture channels
```

Why density/hairness is removed:

```text
In Checkpoint A one root generates one strand. Root count is controlled by
initialization, densification, and pruning. Visibility is controlled by
opacity. A separate density/hairness channel would duplicate opacity or root
existence and make pruning/editing ambiguous.
```

Defer:

```text
clump
curl
guide-root to multiple child-strands
strand template library
```

These are useful for richer assets, but adding them before the base Stage 1
system is stable would make the first prototype harder to evaluate.

Verification:

```text
parameter maps are readable
local edits visibly affect only the intended control
no removed parameter name appears in the formal code path
```

## 7. Strand and Gaussian Generation

Decision: Keep one-root-one-strand in Checkpoint A.

Reason:

```text
This keeps the geometry path explainable. More complex guide/child strand
systems can be added after the base reconstruction works.
```

Keep:

```text
root surface point
tangent flow direction
normal lift
static sag / gravity-like bend
width profile
sampled anisotropic Gaussian segments
```

Important:

```text
static sag is part of the learned groom shape. If exported to an engine,
simulation should add secondary motion, not duplicate the full learned sag.
```

Verification:

```text
strand debug render
Gaussian count per strand
length/width/sag statistics
```

## 8. Orientation and Flow

Decision: Keep orientation-supervised flow.

Reference:

```text
NeuralFur / GaussianHaircut-style orientation supervision
Gabor-filter orientation maps
```

Reason:

```text
Fur direction is not reliably recovered from RGB loss alone. Projecting
strand tangents and supervising them with confidence-masked orientation maps
is directly aligned with the grooming goal.
```

Keep:

```text
DoG + Gabor orientation maps
orientation confidence
flow initialization from orientation maps
projected flow orientation loss
flow hint prior
flow coherence on root graph
```

Remove:

```text
Canny-like edge direction as the main flow source
orientation loss without masks/confidence
```

Verification:

```text
orientation target visualization
projected strand-flow overlay
orientation loss curve
flow map before/after training
```

## 9. Head, Boundary, and Detail Handling

Decision: Keep these as weights and evidence, not as unrelated objectives.

Reason:

```text
The white tiger head, silhouette, stripes, and high-detail regions are where
full-body averages hide failures. They should increase the importance of
existing RGB, mask, orientation, and densification signals.
```

Keep:

```text
head-region metrics
head-weighted RGB/mask/orientation losses
boundary-aware mask weights
detail-weighted RGB loss
boundary/detail/head evidence for densification parent selection
```

Remove:

```text
new appearance stories unrelated to grooming
head-only tricks that are not reported separately
```

Verification:

```text
head metric table
boundary/detail/head weight visualizations
comparison of full-image metrics and head metrics
```

## 10. Random Mesh / Background Backing

Decision: Keep only as a training stabilization trick.

Reason:

```text
White fur against a white background can be optimized as semi-transparent
alpha. Random colored backing makes that shortcut less attractive during
training.
```

Rules:

```text
training only
disabled in evaluation
not a contribution
not a base/fur decomposition
not used to decide fur/no-fur identity
```

Verification:

```text
same checkpoint evaluated with randomization disabled
alpha histogram before/after enabling the trick
```

## 11. Root Neighborhood Graph and Smoothness

Decision: Rebuild as a deterministic root graph.

Reason:

```text
The parameter maps looked noisy in the smoke test. Smoothness must be defined
on nearby roots on the mesh, not fragmented UV adjacency.
```

Keep:

```text
root graph parameter smoothness
flow coherence
range priors for length / width / sag / stiffness / opacity
```

Remove:

```text
random same-face / adjacent-face edge sampling as the formal graph
UV TV as the core smoothness mechanism
```

Formal graph requirement:

```text
deterministic
reproducible
fast enough for each structure update
usable for smoothness, densification placement, pruning, and edits
```

Verification:

```text
graph degree statistics
smoothness loss curve
parameter map before/after smoothness
```

## 12. Root Densification

Decision: Rebuild. Keep the goal, not the old implementation.

Reason:

```text
AniGroom needs more local root capacity where reconstruction, orientation, or
boundary evidence says the current roots are insufficient. The old global
face-residual random resampling is too coarse and not aligned with root-level
grooming.
```

Formal split:

```text
parent-root selection
local new-root placement
```

Parent selection signals:

```text
per-Gaussian visibility / contribution aggregated to root_id
root movement gradient evidence without cancellation
image residual evidence
orientation failure evidence
boundary/detail/head evidence
```

Important gradient rule:

```text
Normal training gradients are left untouched. For densification evidence,
record per-Gaussian gradient contribution toward root motion and accumulate
absolute values per root. This avoids opposite strand samples cancelling each
other when deciding whether a root needs more capacity.
```

New-root placement:

```text
selected parent
inspect local root neighborhood
place new root on nearby under-covered surface position
inherit/interpolate groom parameters from parent and neighbors
```

Remove:

```text
fixed-percent densification
global random root resampling
old residual-face sampler as the formal algorithm
```

Verification:

```text
densification log with parent ids and evidence values
root count over time
new root placement visualization
before/after local reconstruction or orientation error
```

## 13. Root Pruning

Decision: Rebuild prune around sustained low contribution.

Reason:

```text
Pruning should remove roots that do not help training, without deleting roots
before they have had a fair chance to be observed.
```

Formal rule:

```text
do not prune before training views have been observed at least once
prune_start = max(one_train_pass_iterations, configured_prune_warmup)
run prune on the post-densification root set
keep all tensors, masks, optimizer state, and root_id mappings synchronized
```

Prune signals:

```text
consistently invisible
low accumulated contribution
low effective opacity
optional persistently low gradient/evidence
```

Verification:

```text
prune log with root ids and reasons
root count before/after
no index/mask mismatch after structure update
render quality does not collapse after prune
```

## 14. Adaptive Sampling

Decision: Defer until fixed sampling is stable, but preserve the design path.

Reason:

```text
Long or highly curved strands need more Gaussian samples than short/simple
strands. This matters for speed and quality, but it should not be introduced
before the gsplat Stage 1 path is correct.
```

Checkpoint A order:

```text
first fixed sample count
then record sample-count and quality bottlenecks
then enable adaptive sampling only if the fixed version is stable
```

Verification:

```text
same checkpoint rendered with fixed and adaptive sampling
speed / memory / quality table
```

## 15. New Module Candidates

The following modules may be added only if the baseline Stage 1 path is stable
and their effect can be measured.

### 14.1 Root-Evidence Densification

Status: likely useful, part of Checkpoint A once basic training runs.

Reason:

```text
It is directly tied to AniGroom's mesh-rooted representation and improves
capacity where the current root set is insufficient.
```

### 14.2 Asset-Cleaning Objective

Status: defer.

Reason:

```text
Asset mode is important for the paper, but Checkpoint A is Stage 1
reconstruction. Asset objectives should not contaminate reconstruction
metrics before the reconstruction path is proven.
```

### 14.3 Clump / Curl / Child-Strand Templates

Status: defer.

Reason:

```text
They can improve richer grooming, but the white tiger first prototype does
not need them to prove the main route. They should be introduced only when
there is a clear artifact that the current parameter set cannot handle.
```

### 14.4 Multi-View Root Color Initialization

Status: optional experiment, disabled by default.

Reason:

```text
Projecting roots into training views and averaging sampled color can make
stripe patterns appear earlier, but the naive version mixes back-facing,
occluded, and misaligned evidence. On the 10k-root / 500-iteration white-tiger
short run, both full color write and 0.35 blended color hint reduced PSNR/SSIM
relative to the default graph+orientation run.
```

Decision:

```text
keep the code path as an explicit experiment switch
do not enable it by default for reconstruction
only reconsider after adding visibility/front-facing filtering
```

### 14.5 Mesh Backing / Random Mesh Color

Status: diagnostic, not a default reconstruction module yet.

Reason:

```text
Random mesh backing is useful for testing the white-fur-on-white-background
transparency problem. It verifies that a surface support layer can be rendered
through the formal gsplat path without affecting root-evidence accounting.
However, the 10k-root / 2000-iteration backing run did not beat the no-backing
mainline on reconstruction metrics, and visual inspection still showed
gray/fog-like fur. Keep it as an explicit switch while the coverage problem is
being diagnosed.
```

Decision:

```text
keep --mesh-backing as an explicit experiment switch
do not enable it by default
do not describe it as the Stage 1 contribution
```

### 14.6 Global Opacity Floor

Status: negative result, not default.

Reason:

```text
The 5000-iteration mainline improves PSNR but reduces mean opacity, producing
empty/transparent regions around the belly and legs. A global opacity floor
was tested to check whether this was the main failure mode.
```

Result:

```text
output: outputs/stage1_20260623_10000r_3000i_densify_opfloor065_v1
opacity_floor: 0.65
opacity_floor_weight: 0.35
test_psnr at 3000: 16.72258949279785
comparison: no-floor mainline at 3000 is 16.86288070678711
visual: opacity is preserved, but the result becomes more speckled and does
not solve the coverage problem cleanly.
```

Decision:

```text
keep opacity_floor only as a diagnostic switch
do not enable it by default
solve coverage locally through root distribution, head/detail weighting, and
better structure constraints rather than globally forcing opacity
```

### 14.7 Head / Detail Loss Weighting

Status: keep as a formal analysis and weighting mechanism.

Reason:

```text
Checkpoint A requires head/detail-specific evidence, not only global
foreground and boundary weights. The formal implementation adds two
image-space maps:

1. a target-image detail map from masked Sobel gradients, which emphasizes
   stripes, local texture, and silhouette detail;
2. a projected head-region map from high-Z mesh roots, which emphasizes the
   head/front-body region without using a separate hand mask.

Both maps affect only loss weights. They do not change the representation or
renderer, and they are saved as weight_debug images for verification.
```

Result:

```text
output: outputs/stage1_20260623_10000r_5000i_headdetail_v1
test_psnr at 5000: 17.02278709411621
test_ssim at 5000: 0.8468822240829468
test_mask_l1 at 5000: 0.10852858424186707
comparison: slightly better than the no-head/detail mainline at 5000
visual: very close to the mainline; it does not solve belly/leg empty regions
by itself.
```

Decision:

```text
keep detail/head maps and logs
keep weight_debug visualization
do not overstate this as the main quality fix
use it as supporting evidence while solving the remaining local coverage
problem
```

## 16. Checkpoint A Minimum Evidence

Checkpoint A is not complete until the following exist:

```text
plain gsplat baseline metrics and renders
AniGroom Stage 1 metrics and renders
360 turntable
root distribution visualization
length / width / flow / bend-sag / stiffness / color / opacity visualizations
orientation overlay
head/boundary/detail evidence visualization
densification/prune logs if enabled
server training log with time and memory
```
