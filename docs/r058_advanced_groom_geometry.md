# R058 Advanced Groom Geometry

Status date: 2026-08-11.

Status: accepted strand-geometry implementation. R057 remains the last trained
staged-shape checkpoint, but its retired geometry schema is not executable in
the R058 source. No Stage 1 training or schedule change is part of R058.

## Scope

R058 defines how optional curl and frizz deform the already accepted ordinary
strand. The ordinary strand remains responsible for:

- a surface-attached root;
- positive nominal straight length;
- normalized 3D groom direction;
- the single normal-to-direction brush transition controlled by stiffness.

Curl and frizz are detail layers around that backbone. They do not replace the
3D direction field, brush stiffness, interpolation, or root hierarchy.

## Why The Old Definition Was Invalid

The previous implementation predated the 3D-direction brush backbone and had
four structural defects:

1. Its second curl axis was approximately the mesh normal instead of a vector
   perpendicular to the actual sampled 3D backbone.
2. Curl was introduced with a one-sided envelope and then visually forced into
   shapes that could resemble a second brush bend or an S-curve.
3. Frizz frequency and phase were derived from curl frequency and phase, so the
   two editor controls were not independent.
4. Frizz was a regular analytic wave. It produced repeated, aliased patterns
   instead of irregular small-scale hair variation.

These defects explain why an isolated exported strand could look abnormal even
before any RGB loss or training schedule acted on it. A loss cannot repair an
incorrect forward geometry definition.

## Reference Semantics

The control semantics follow established grooming systems rather than an
animal-specific rule:

- Blender Curl Hair Curves exposes curl radius, frequency, start, end factors,
  random offset, and guide-based deformation:
  https://docs.blender.org/manual/en/4.3/modeling/geometry_nodes/hair/guides/curl_hair_curves.html
- Blender Frizz Hair Curves uses per-curve random variation, cumulative smooth
  offsets, amplitude controls, and optional length preservation:
  https://docs.blender.org/manual/pt/dev/modeling/geometry_nodes/hair/deformation/frizz_hair_curves.html
- SideFX Hair Clump treats curl amplitude/radius and frequency as independent
  controls along a guide:
  https://www.sidefx.com/docs/houdini/nodes/sop/hairclump.html
- SideFX grooming interpolates generated hair from guide curves:
  https://www.sidefx.com/docs/houdini/nodes/obj/hairgen.html

R058 adopts the common geometry semantics, not the complete UI or every
vendor-specific option.

## Geometry Contract

Let `B(t)` be the accepted brush backbone for `t` in `[0,1]`. Its root, nominal
tip, length, normal-to-direction transition, and 3D direction are fixed before
advanced detail is evaluated.

For every sampled point, R058 computes:

- `tau(t)`: local unit tangent of `B`;
- `s(t)`: transported transverse side axis;
- `o(t)`: transverse outward axis;
- `E(t) = t^2 (3 - 2t)`: root detail envelope.

The frame follows the sampled brush curve without adding twist around the
root-to-tip chord. The envelope has `E(0)=0` and `E'(0)=0`, so every detail
layer preserves both root position and root tangent.

### Curl

For physical radius `r`, turns-per-strand `f`, and phase `phi`:

```text
theta(t) = phi + 2 pi f t

C(t) = r E(t) [
    (sin(theta(t)) - sin(phi)) s(t)
  + (cos(theta(t)) - cos(phi)) o(t)
]
```

This definition provides:

- exact identity when `r=0`;
- exact identity when `f=0`, even if radius is nonzero;
- smooth root attachment;
- a real transverse 3D coil for large radius/turn values;
- no artificial requirement to return to the nominal tip.

Allowing the detailed tip to move is intentional. Pinning both ends of a curl
creates the forced-return S-curve seen in the retired implementation.

### Frizz

Frizz uses two deterministic, band-limited noise signals `eta_s(t)` and
`eta_o(t)` generated from a fixed per-strand seed:

```text
F(t) = a E(t) [eta_s(t) s(t) + eta_o(t) o(t)]
```

`a` is physical amplitude. The seed is persistent root state but is not a
trainable groom coordinate. It is independent of curl turns and phase. Cubic
interpolation between noise knots makes the shape stable when strand sampling
density changes.

### Composition

Both offsets are evaluated around the same undeformed brush backbone:

```text
P(t) = B(t) + C(t) + F(t)
```

This makes curl and frizz additive and order-independent. Frizz does not ride
on a separately twisted curl frame, and changing curl controls cannot silently
change the frizz realization.

There is one geometry path: both transverse axes are retained. R058 has no
outward-only, tangent-only, legacy, or fallback deformation mode.

The generic groom field initializes curl radius and frizz amplitude at their
neutral lower bounds. Advanced geometry therefore changes no strand until its
controls are explicitly enabled; the visualizer sets physical test values
directly instead of relying on hidden nonzero defaults.

## Sampling And Differentiability

Advanced deformation is evaluated before adaptive strand resampling and
strand-to-Gaussian conversion. Final arc length and turning complexity therefore
increase Gaussian count when the actual detailed curve requires it.

The integer segment decision remains detached. Strand positions remain
differentiable with respect to the brush backbone, curl radius, curl turns,
curl phase, and frizz amplitude.

The persistent frizz seed is a registered buffer. It is stored in state and
migrated through root lifecycle updates, but cannot enter an optimizer.

## Verification

The local test suite passes with `132 passed` and covers:

- orthonormal and no-twist frames;
- exact root and root-tangent preservation;
- zero-control and zero-turn identity;
- transverse offsets;
- curl/frizz independence and additive composition;
- signed displacement on both transverse axes;
- no axial foldback for moderate controls;
- finite real coils for deliberately extreme controls;
- physical scale equivariance;
- frizz stability across sample densities;
- finite nonzero gradients;
- persistent, non-trainable frizz seed state.

Canonical moderate sweeps report zero backward strands for curl radius, curl
turns, frizz amplitude, and combined controls. The stress sweep produces four
backward cases only when radius is deliberately near half or all of a very
short strand length with `2.5-4` turns. Those cases are continuous 3D coils,
not polygon zigzags or projection artifacts.

The exact geometry outputs and report are under:

```text
D:/RTS/_tmp/r058_groom_geometry_v3
```

The formal `strand -> adaptive segments -> Gaussians -> gsplat` validation is:

```text
D:/RTS/_tmp/r058_groom_geometry_v3/controls/groom_parameter_controls_sheet.png
D:/RTS/_tmp/r058_groom_geometry_v3/controls/report.json
```

All intended trainable controls in that render path receive finite nonzero
gradients. The visualizer uses the existing canonical rendering path; R058 did
not add a second fake or private renderer.

After removing the retired schema and mode switch, all five canonical NPZ
exports were compared against the accepted R058 geometry output. Every strand,
width, color, opacity, and root-id array had maximum absolute difference
`0.0`.

## Implementation

- advanced deformation: `anigroom/grooming/strand_deformations.py`
- integration before sampling: `anigroom/grooming/strand_gaussians.py`
- unified geometry export and gsplat QA:
  `tools/visualize_groom_parameter_controls.py`
- lifecycle propagation of the fixed frizz seed:
  `tools/train_white_tiger_stage1.py`
- tests: `tests/test_strand_deformations.py`

The training entry uses the same physical names and propagates the fixed seed
through root lifecycle updates. The retired geometry mode is absent from the
model, CLI, launcher, and configs. No optimizer, loss, or schedule was changed,
and no training was executed.

## Checkpoint Boundary

R058 renames the physical curl coordinate to `curl_turns` and adds persistent
`frizz_seed_phase` state. Existing R057 checkpoints use a different schema and
must fail strict loading. There is no alias, key conversion, non-strict load,
or one-time checkpoint migration; the first R058 training experiment starts
from zero.

R058 also does not accept or revise the existing curl/frizz decoder ranges,
ownership hierarchy, regularization, or unlock schedule. Those are training
questions and require separate controlled experiments. R057 remains historical
training evidence, not a runtime compatibility path.
