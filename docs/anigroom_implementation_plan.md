# AniGroom Implementation Plan

This document defines the next execution plan after the Stage 1 technical
framework. It is a planning document only. It does not start implementation.

## 1. Current Decision

AniGroom will be reimplemented as a clean formal project.

The previous version is treated as a smoke test:

```text
useful mechanisms may be recovered
old code structure is not kept
old hand-written renderer is not kept
old rough densification implementation is not kept
```

The new implementation must follow:

```text
formal Stage 1 technical framework
gsplat renderer
server-first training and validation
strict train/test protocol
no closed-door reinvention
```

## 2. Compute Policy

Most work should run on the server, not the local workstation.

Reason:

```text
local machine may already be running other training jobs
local GPU memory should not be risked
formal tests must be reproducible on server hardware
```

Local machine responsibilities:

```text
code editing
document writing
small CPU-only checks
log parsing
pulling results from server
visualization / figure / video assembly
very small GPU sanity checks only when memory risk is clearly below 32 GB
```

Local GPU use is allowed only for minimal tests such as:

```text
import checks
single forward pass on tiny image
tiny synthetic scene
one or two debug iterations
```

Local GPU use is not allowed for:

```text
full white tiger training
large multi-view experiments
long baseline training
large ablations
```

Server responsibilities:

```text
environment setup
plain gsplat baseline
official 3DGS baseline
AniGroom Stage 1 training
ablation training
full metric evaluation
long-running experiments
```

## 3. Server Execution Policy

Development environment is used for setup and short verification.

Long training must use server training tasks when available, not a temporary
development environment.

Development environment checks:

```text
repository sync
dependency check
dataset path check
camera/mask/image sanity check
plain gsplat import and forward/backward check
short baseline run
short AniGroom run after implementation
```

Training task checks:

```text
script path
data path
output path
GPU assignment
environment image
log writing
checkpoint writing
metric writing
failure visibility
```

The development environment has limited lifetime. If it is used for setup, it
must be refreshed/exported according to platform rules before timeout.

## 4. Project Structure

The clean implementation should use a modular layout:

```text
anigroom/
  data/
  mesh/
  roots/
  groom/
  rendering/
  losses/
  densification/
  evaluation/
  visualization/

scripts/
  server/
  local/

docs/
  stage1_confirmed_framework.md
  stage1_module_decisions.md
  anigroom_implementation_plan.md
```

Each module must define:

```text
purpose
inputs
outputs
reference implementation or paper
unit or sanity test
failure conditions
```

## 5. Baseline Plan

Two baselines are required.

### 5.1 Official 3DGS Baseline

Purpose:

```text
external reconstruction baseline
standard 3DGS comparison
sanity check for camera/data quality
```

Execution:

```text
clone official 3DGS repository
run white tiger training on server
evaluate train/test metrics
save renders and logs
```

### 5.2 Plain gsplat Baseline

Purpose:

```text
verify our data loader
verify our camera convention
verify gsplat backend
verify metric pipeline
provide internal baseline before AniGroom
```

Execution:

```text
implement minimal free-Gaussian gsplat training in the new project
run short verification first
then run full baseline on server
```

This baseline must not use AniGroom roots or strands.

## 6. AniGroom Stage 1 Implementation Order

Implementation begins only after this plan and the Stage 1 technical framework
are accepted.

Order:

```text
1. data loader and train/test split
2. plain gsplat baseline
3. mesh loading and root initialization
4. face-local barycentric root coordinates
5. groom parameter container
6. strand to Gaussian generation
7. gsplat rendering path
8. RGB and mask losses
9. orientation and flow supervision
10. head/boundary/detail weighting
11. root graph smoothness and flow coherence
12. random mesh/background backing
13. root densification and pruning
14. adaptive strand/Gaussian sampling
15. full evaluation and visualization
```

Each step must run a server-side short verification before moving to the next
step.

## 7. Recovered Mechanisms to Preserve

Useful mechanisms from the previous smoke-test version must be recovered in the
formal version:

```text
head-region metrics and weighting
orientation-based flow initialization
flow hint prior
projected flow orientation loss
flow coherence
boundary/detail weighting
random mesh/background backing
face-local root movement
root-graph smoothness
root evidence for densification parent selection
adaptive strand/Gaussian sampling
debug metrics and visualizations
```

They must be rebuilt on top of:

```text
formal root representation
gsplat renderer
clean train/test protocol
server-reproducible scripts
```

## 8. Evaluation Tracks

The project has two evaluation tracks.

### 8.1 Reconstruction Track

Purpose:

```text
measure image reconstruction and held-out view quality
```

Compare against:

```text
official 3DGS baseline
plain gsplat baseline
GaussianHaircut / NeuralFur when applicable
```

Metrics:

```text
PSNR
SSIM
LPIPS
mask / alpha error
head-region metrics
orientation metrics
training time
memory
```

### 8.2 Asset Extraction Track

Purpose:

```text
measure whether the output is a usable editable fur asset
```

Compare against:

```text
GaussianHaircut
NeuralFur
other animal/hair asset methods if available
```

Evidence:

```text
360 render
root distribution visualization
parameter maps
flow visualization
local edit examples
densification visualization
before/after asset-cleaning results
```

This track is not only PSNR-driven.

## 9. Verification Gates

Do not start full training until these gates pass:

```text
plain gsplat baseline runs
camera projections visually match images
mask loading is correct
train/test split is fixed
single-view overfit works
small multi-view overfit works
root gradients are nonzero and stable
barycentric roots remain inside their faces
gsplat render is stable
logs and checkpoints are written
evaluation script runs without training-time randomization
```

Full server training starts only after short server verification succeeds.

## 10. Failure Rules

A run must be stopped and inspected if:

```text
loss becomes NaN
render becomes blank
alpha becomes all transparent or all opaque
root count changes without matching mask/state update
train/test resolution mismatch appears
metric script silently resizes predictions incorrectly
GPU memory grows unexpectedly
training is much slower than expected
logs or checkpoints are missing
```

No silent fallback is allowed.

## 11. Immediate Next Steps

The first non-training implementation step is now available:

```text
tools/report_white_tiger_stage1_inputs.py
scripts/server/install_gsplat_cached_backend.sh
scripts/server/checkpoint_a_preflight.sh
```

This verifies the current white-tiger processed input layout and writes a
fixed train/test split report. The default local result is:

```text
36 views total
30 training views
6 held-out test views
test_stride = 6
image resolution = 1920 x 1080
```

This report must be reused by both the plain `gsplat` baseline and AniGroom
Stage 1. Do not let each training script invent its own split.

Next planning tasks:

```text
1. run scripts/server/install_gsplat_cached_backend.sh if gsplat CUDA is missing
2. run scripts/server/checkpoint_a_preflight.sh on the server development environment
3. confirm gsplat import, CUDA visibility, and gsplat backward on server
4. define plain gsplat baseline script
5. define official 3DGS baseline script
6. implement shared evaluator using the reported split
7. implement AniGroom Stage 1 modules in the clean package structure
```

Only after these are accepted should code implementation begin.

## 12. Current Server Target

The current development environment requested for Checkpoint A is:

```text
20260620113448
```

Use it for dependency checks, data-path checks, short baseline verification,
and short AniGroom verification. If the development environment expires,
restart or refresh it according to the established server workflow before
continuing.

Do not start full training until the local repository state and server scripts
match the formal documents above.
