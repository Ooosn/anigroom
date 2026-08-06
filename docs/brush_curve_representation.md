# Brush Centerline Representation

Status: R039 strict-schema candidate. The geometry and schema are implemented;
formal asset visualization and from-zero training are not yet accepted.

## Purpose

Ordinary brushed fur needs one natural transition from the surface normal to a
learned 3D endpoint. A straight strand is already a valid baseline. The brush
control may curve that same strand once, but it must not add a second interior
deformation, change the endpoint, or hide image-driven geometry in an
uninterpretable field.

The executable base centerline therefore contains only:

- straight root-to-tip length;
- normalized local 3D endpoint direction;
- guide-owned brush stiffness.

Curl and frizz remain separate optional controls and are disabled in the R039
white-tiger candidate. There is no additional low-frequency interior shape
field in the model, render residual, optimizer, lifecycle state, checkpoint,
CLI, or configuration.

## Geometric Contract

For root `P0`, outward unit normal `n`, normalized learned direction `d`, and
positive straight length `L`:

```text
P2 = P0 + L d
```

`P2` is fixed for every stiffness value. The conceptual one-corner polyline is

```text
Q = P0 + dot(P2 - P0, n) n
P0 -> Q -> P2
```

where `Q` is the point on the root-normal ray at the tip's normal height. The
actual centerline is one quadratic Bezier, not that hard corner.

The normal/direction disagreement is an explicit continuous coefficient:

```text
d_tangent = d - dot(d, n) n
direction_difference = ||d_tangent||
effective_stiffness = brush_stiffness * direction_difference
```

No angular threshold or special aligned-direction branch is used. A direction
close to the normal continuously reduces the effective stiffness. An exactly
normal-aligned direction gives zero effective stiffness through the same
formula used by every other strand.

Let the straight quadratic control point be

```text
M = (P0 + P2) / 2
```

and define

```text
C = M + effective_stiffness (Q - M)
B(t) = (1-t)^2 P0 + 2(1-t)t C + t^2 P2,  t in [0,1]
```

This has the required behavior:

- `brush_stiffness = 0` is the exact straight segment;
- increasing stiffness approaches the smooth version of `P0 -> Q -> P2`;
- root and tip never move;
- the centerline is one quadratic turn and has no second bend or inflection;
- normal/direction agreement suppresses curvature continuously;
- length, direction, and stiffness remain differentiable.

## Ownership And Interpolation

`brush_stiffness` is a guide-root field in the semantic interval `[0,1]`.
Render roots obtain it through the same intrinsic surface interpolation used by
the other guide controls. It has no render-root residual. Its graph smoothness,
guide lifecycle interpolation, optimizer ownership, checkpoint identity, and
diagnostic output all use the same field name and schema.

This ownership prevents local RGB evidence from independently changing the
material curve at every render root. Render-root length and direction residuals
remain separate visible-geometry controls under their existing schedule.

## Gaussian Sampling

The final enabled centerline is built before discrete segment allocation.
Absolute straight length provides the base count; final arc length and turning
complexity add samples where the curve requires them. There is a representation
minimum but no animal-specific maximum segment count.

Segment counts are detached integer topology decisions. Generated Gaussian
positions remain differentiable with respect to length, 3D direction, and
brush stiffness.

## Strict Schema

R039 intentionally cannot load an R038 checkpoint. The removed field is not
ignored, zeroed, migrated, or retained for compatibility. Frozen R036/R038
configs and documents remain only as immutable experimental evidence; the R039
launcher config is `configs/r039_brush_centerline_0_30k.env`.

## Acceptance Checks

Before any from-zero training, the representation must pass:

1. exact straight behavior at zero stiffness;
2. fixed root and tip for every stiffness;
3. explicit direction-difference scaling;
4. natural straight behavior for normal-aligned directions without a branch;
5. one quadratic turn with no second curvature event;
6. finite nonzero gradients to length, direction, and stiffness;
7. strict absence of the removed field from executable schemas;
8. one canonical large centerline visualization using this exact function.

Only after those checks pass may R039 start a strict from-zero Stage 1 run.
