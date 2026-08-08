# R053 Shape/Appearance Handoff

## Status

Implementation and matched validation in progress. R050 remains the accepted
appearance checkpoint until both 30k branches complete fixed RGB and structural
QA.

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
