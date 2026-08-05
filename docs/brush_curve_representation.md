# Brush Curve Representation

Status: design accepted; implementation is the next representation focus after
the active hard-range audit is closed. This document is a design contract, not
evidence that the current trainer already implements the representation.

## Purpose

The current straight-strand plus legacy lateral-bend construction does not
explicitly represent the normal-to-groom transition seen in brushed fur. The
replacement must preserve a simple, editable length meaning while producing a
smooth root-to-tip curve and removing the remaining animal-scale decoder
ranges.

The representation separates:

- the root-to-tip displacement and low-frequency brushed profile;
- optional non-periodic bend, periodic curl, and high-frequency frizz;
- guide-owned material/groom controls from image-driven local appearance;
- continuous differentiable strand geometry from discrete Gaussian sampling.

## Geometric Contract

For root position `P0`, outward unit surface normal `n`, normalized learned 3D
groom direction `d`, and learned straight length `L`:

`P1 = P0 + L d`

`L` is the straight root-to-tip distance, not the final curve arc length. This
keeps the control and its optimization simple. The endpoint displacement is
decomposed into a normal and tangent component:

`delta = P1 - P0`

`h = dot(delta, n)`

`t = delta - h n`

The base strand is then

`P(s) = P0 + h F_n(s, c) n + F_t(s, c) t`, for `s in [0, 1]`.

The transition functions must satisfy:

- `F_n(0, c) = F_t(0, c) = 0`;
- `F_n(1, c) = F_t(1, c) = 1`;
- position and tangent are continuous for every valid `c`;
- both functions equal `s` when `c = 0`, giving the exact straight segment;
- as `c` approaches `1`, normal height is accumulated earlier and tangential
  displacement later, giving a smooth normal-emergence/laydown curve;
- the root and endpoint remain unchanged for every value of `c`.

The canonical internal control is `brush_curve_strength c in [0, 1]`:

- `c = 0`: straight root-to-tip line;
- `c = 1`: strongest smooth normal-to-groom transition supported by the base
  profile.

If the public editor exposes the conventional material term `stiffness`, use
`brush_curve_strength = 1 - stiffness`, so high stiffness means straighter
hair. The implementation must not use the name `stiffness` with the opposite
meaning.

The normal height `h` is derived from the learned endpoint. It is not an
independent lift parameter and has no separate physical range.

## Parameter Ownership

`brush_curve_strength` is a guide-root material/groom field. Render roots
obtain it through the same intrinsic surface interpolation and smoothing used
for other guide fields. It has no render-root residual: local RGB evidence
must not independently change material stiffness at every render root.

The base guide representation is therefore:

- straight length;
- normalized local 3D direction;
- brush curve strength (or conventional stiffness through its complement).

Visible geometric fields such as length, direction, and width may retain the
already accepted guide/render hierarchy. Gaussian RGB residuals remain the
outlet for shadows, highlights, and other high-frequency photometric evidence.

## Bend, Curl, And Frizz

These controls are complementary only when their geometric roles are
identifiable:

- brush curve strength: the base normal-to-groom transition;
- bend: optional low-frequency, non-periodic centerline deformation;
- curl: coherent periodic or helical deformation;
- frizz: zero-mean high-frequency irregular deformation.

The legacy one-axis bend must not be reused as the base brushed profile. A
future bend must alter the curve interior without duplicating the endpoint 3D
direction. Curl and frizz must not use fixed animal-scale amplitude or
frequency endpoints.

Brush curve strength belongs to the base guide geometry. Optional bend, curl,
and frizz shape detail may unlock after 20k through the single shared shape
detail ramp. They must not introduce separate per-attribute schedules. For the
white-tiger checkpoint they should be allowed to remain near zero; their
representation must nevertheless support genuinely bent, curly, and frizzy
fur on other subjects.

## Gaussian Sampling

The final centerline is constructed first, including every enabled shape
component. Gaussian allocation then uses the resulting curve, not only the raw
straight length:

- straight length contributes the base representation count;
- final arc length and local turning/approximation error add samples where the
  curve needs them;
- no maximum segment count or animal-specific physical threshold is applied;
- a minimum segment count and numerical tolerances are representation
  requirements, not learned-attribute clamps.

The current renderer deterministically rebuilds strands, segment budgets, and
derived Gaussian tensors on every render call. Segment counts are detached
integer topology decisions; Gaussian positions and attributes remain
differentiable with respect to the groom fields. The brush curve adds only
vectorized curve evaluation, so the initial implementation keeps this
per-iteration rebuild. Caching is justified only by profiling, not by changing
the model contract.

Any future persistent Gaussian RGB residual must use stable semantic identity,
such as render-root identity plus normalized strand coordinate, rather than a
transient flattened Gaussian array index.

## Hard-Range Replacement

Implementing this design closes the physical decoder-range audit only if the
old definitions are removed rather than disabled:

- remove the legacy `tanh` bend interval;
- remove fixed curl radius and frequency intervals;
- remove the fixed frizz amplitude interval;
- remove any remaining independent lift field or lift range;
- retain the already accepted positive-unbounded length, width, taper, and
  child-spread representations;
- retain only semantic domains such as normalized direction, opacity/color,
  tip ratio, clump weight, and brush curve strength.

Positive shape amplitudes may use positive unbounded coordinates and relative
soft priors. Stability must come from guide ownership, intrinsic interpolation,
smoothness, shared staged optimization, and image evidence, not post-decode
physical clipping.

### Current audit state

The completed R-series has already removed active physical endpoints from
length, root width, width taper, and child spread. Root/tip color and opacity,
tip/root width ratio, clump weight, normalized direction, and the future brush
curve strength retain semantic domains and are not hard-coded animal scales.

The remaining active source definitions to replace are exactly:

- legacy bend decoded with `tanh` into `[-1, 1]`;
- curl radius decoded into `[0, 0.026]` in the formal trainer;
- curl frequency decoded into `[0, 5.5]`;
- frizz amplitude decoded into `[0, 0.010]`.

Curl and frizz have zero effective scale in the current white-tiger config, so
they did not affect R036. They still matter to the general representation and
must be redesigned rather than merely left disabled. There is no active
independent lift or stiffness field in the current model; the brush-curve
implementation introduces one guide-owned control and must not revive the old
lift path.

## Acceptance Criteria

Before this representation replaces the current baseline:

1. `c = 0` reproduces the exact straight strand and the same endpoint.
2. Increasing `c` preserves both endpoints and produces a continuous,
   monotonic normal-to-groom transition without a kink or loop.
3. Direction, length, and brush-curve gradients are finite and nonzero in
   differentiable tests.
4. Guide interpolation remains smooth across mesh faces and lifecycle changes.
5. Final-curve Gaussian allocation increases for real arc-length/turning
   complexity and has no maximum cap.
6. The old bend, lift, curl, and frizz physical ranges are absent from source,
   config, checkpoints, tests, and active documentation.
7. Fixed-view visualizations show straight, brushed, bent, curly, and frizzy
   cases with one canonical renderer and camera protocol.
8. A strict from-zero white-tiger run preserves the accepted composite metric
   range while improving the pure-fur structural visualization.
