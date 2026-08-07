# R044 Parent-Conditioned Secondary Guide Residual Field

## Status

R044 is an isolated candidate built from the immutable R043 baseline. Its
from-zero full-resolution H100 forward/backward gate and strict checkpoint
resume gate pass. The formal 30k result is pending; R043 remains the active
accepted baseline until that result is measured and structurally inspected.

## Problem

R043 has 4500 primary guides and about 470k independent render roots after its
finite lifecycle. The primary field is appropriately smooth but too sparse to
represent controlled local geometry deviations. Giving every render root an
independent geometry residual provides enough capacity, but it also makes the
residual field expensive to smooth and allows isolated strands to depart from
their neighborhood.

R043's density-matched K32 render graph contains about 15 million directed
edges at the final population. Several per-iteration geometry losses traverse
that graph. Reducing K, using a stale graph, or smoothing generated strand
samples would change the accepted physical support or optimize the wrong
representation.

## Representation

R044 introduces a fixed secondary guide layer `G1` between primary guides
`G0` and render roots `R`:

```text
G0: direct low-frequency groom from clean-flow initialization
G1: zero-centered local geometry residuals only
R:  direct G0 base + interpolated G1 residual + render appearance
```

The base groom is never routed through `G1`. A render root samples its G0 base
exactly as in R043 and independently samples the G1 residual field. Therefore
an all-zero G1 field returns the same render geometry as direct G0
interpolation; there is no G0-to-G1-to-render double interpolation.

The G1 residual field contains the existing zero-centered geometry
coordinates:

- positive relative length;
- root width, tip/root ratio, and width taper;
- normalized local 3D direction perturbation;
- optional curl and frizz coordinates;
- child spread and clump strength.

Brush stiffness remains guide-owned. Curl and frizz retain zero effective
scale in this short-fur configuration. R044 does not add an animal-region rule,
length endpoint, split budget, or density schedule.

## Secondary Placement

The formal population is 4500 G0 roots and 20k G1 roots. Placement is balanced
and parent-conditioned rather than a second unrelated global FPS:

1. Retain every G0 point as an exact G1 anchor.
2. Draw a dense area-uniform mesh candidate pool.
3. Assign every candidate to its nearest topology-valid G0 surface cell.
4. Run seeded local FPS inside each cell with the G0 point as the fixed seed.
5. Give every G0 cell the same base child count.
6. Assign the indivisible remainder to cells with the largest uncovered local
   radius, with deterministic tie handling.

For 4500 to 20k roots, every G0 owns exactly four or five G1 roots. This keeps
the secondary capacity uniform with respect to the primary control field while
allowing the actual points to spread over each local surface cell.

## Surface Support

Each render root first obtains its accepted G0 surface support. Its G1
candidates are then restricted to children of those G0 roots. The nearest G1
support is selected only inside that topology-valid union. This prevents an
ambient-space neighbor from crossing folded or nearby disconnected surface
regions.

The formal configuration uses:

```text
G0 -> render support: K=8
G1 -> render residual support: K=8
G1 smoothing graph: K=32
render appearance graph: K=32
```

G1 local 3D direction residuals are parallel-transported into the query root's
surface frame before interpolation. They are vectors rather than axes, so the
transport path explicitly preserves residual magnitude.

## Optimization And Lifecycle

The accepted G0 base schedule is unchanged. The historical render geometry
residual parameters are replaced by the 20k-row G1 residual parameters and use
the same gradual geometry unlock. Optimizer parameter names and state follow
the active G1 field; no unused render geometry residual remains in the model.

Geometry residual smoothness, effective-groom smoothness, and clean-flow 3D
consistency operate on the fixed 20k G1 graph. The dense render graph retains
appearance smoothing for root/tip color and opacity. Sample-level strand
smoothing remains removed.

Render densification changes R only. It rebuilds G1-to-render support while G1
parameter rows and optimizer state remain fixed. Primary-guide densification
is disabled for this candidate and fails fast if requested, because a changing
G0 topology requires a separately designed G1 lifecycle rather than silent
reparenting.

## Isolated Configuration Change

Compared with `configs/r043_density_matched_render_support_0_30k.env`, the only
additional resolved assignments are:

```text
GEOMETRY_RESIDUAL_DOMAIN=secondary_guide
SECONDARY_GUIDE_ROOT_COUNT=20000
SECONDARY_GUIDE_CANDIDATE_MULTIPLIER=16
SECONDARY_GUIDE_INTERPOLATION_K=8
SECONDARY_GUIDE_SMOOTH_K=32
```

All R043 data, renderer, losses, schedules, lifecycle thresholds, resolution,
and population settings remain unchanged.

## Verification

The complete local suite passes (`96 passed`). Focused coverage includes:

- balanced deterministic local FPS;
- topology-restricted G1 support;
- zero-residual direct-G0 geometry equivalence;
- differentiable interpolation into G1 residual parameters;
- 3D residual magnitude preservation under normal rotation;
- optimizer inclusion of G1 and exclusion of render geometry residuals;
- strict model-state roundtrip;
- render lifecycle support rebuild with unchanged G1 rows.

Static checks pass for Python compilation, shell syntax, diff whitespace, and
the new module/test Ruff scope. R043's configuration and behavior source are
unchanged.

## Full-Resolution H100 Gate

The from-zero gate used the real white-tiger data, official mesh, gsplat
renderer, mesh-depth clipping, complete losses, 400k render roots, 4500 G0
roots, 20k G1 roots, and 1920x1080 images. Only the target iteration was set to
one; no model, renderer, resolution, root count, or loss was reduced.

```text
commit:       2a76723e54b2571d2f8f307400e1bb961fcb81a5
host/GPU:     pcg02 / one H100 80GB
start/end:    2026-08-07 18:11:46 / 18:17:02 +09:00
exit code:    0
G0/G1/R:      4500 / 20000 / 400000
G1 per G0:    4 or 5
support fallbacks: 0
G1 graph:     640000 directed edges
render graph: 12800000 directed edges
iteration 1:  30.93 s including traced full iteration work
peak allocated CUDA: 7673.50 MB
nvidia-smi process: 8810 MB
```

The candidate pool contained 320k points. The smallest/largest G0 cells held
17/140 candidates. Every render query had at least 32 topology-valid G1
candidates before selecting K8.

Artifacts:

```text
source: /home/wangyy/anigroom-r044-secondary-guide-20260807
gate:   /home/wangyy/anigroom-r044-secondary-guide-runtime-20260807/outputs/r044_secondary_guide_formal_gate_h100_20260807
log:    /home/wangyy/logs/anigroom_r044_secondary_guide_formal_gate_h100.log
```

## Strict Resume Gate

`checkpoint_000001.pt` was resumed with optimizer and RNG state enabled and
advanced through a second full forward/backward iteration.

```text
checkpoint SHA-256: 98fd6f93ffb617b9d31d300b1a51bc1b414c264195a8ad6278a3475ac61df7ac
optimizer state entries restored: 21
start/target iteration: 1 / 2
exit code: 0
iteration 2: 32.61 s
peak allocated CUDA: 7774.36 MB
```

The resumed checkpoint restored the exact 20k G1 face IDs, barycentric
coordinates, parent IDs, and nine residual tensors. It rebuilt the same
640k-edge G1 graph and 12.8M-edge render graph without topology migration.

Resume artifacts:

```text
/home/wangyy/anigroom-r044-secondary-guide-runtime-20260807/outputs/r044_secondary_guide_resume_gate_h100_20260807
```

## Acceptance Gate

R044 is not accepted from the one-step gates. The next required evidence is a
from-zero 30k run followed by the same fixed postprocess used for R043:

1. train/test composite PSNR and full lifecycle history;
2. wall time and peak memory against R043;
3. canonical single-image 100k-strand asset renders;
4. centerline length/turn audit, especially tail tip and head fringe;
5. G1 residual distributions and local continuity after the residual unlock;
6. proof that render lifecycle changed support but not G1 row identity.

R044 replaces R043 only if it preserves competitive reconstruction while
improving local groom continuity or execution cost without introducing a new
artifact class.
