# R055 Staged Primary/Secondary Shape Handoff

## Question

Can Gaussian RGB residual absorb image-specific appearance evidence while a
coarse-to-fine curl/frizz hierarchy adds legitimate strand detail without
returning to R053's simultaneous two-level ownership failure?

## Frozen Parent

R054 is immutable. R055 changes only curl/frizz ownership timing and their
secondary residual composition. Length, width, direction, brush stiffness,
interpolation, losses, render-root lifecycle, guide lifecycle, and learning
rates remain identical to R054.

## Method

The primary guide owns absolute semantic curl and frizz. Their common shape
multiplier is zero through iteration 20,000 and ramps linearly to one at
25,000. Gaussian RGB residual uses exactly the same 20,000-25,000 ramp.

The secondary guide owns only zero-centered positive relative residuals:

```text
effective = primary * exp(asinh(secondary_raw) * secondary_multiplier)
```

Its one shared multiplier is zero through 25,000 and ramps linearly to one at
30,000. A zero residual returns the primary field exactly; a zero primary value
cannot be turned into curl or frizz by the secondary field.

No new loss, absolute threshold, animal region, bend path, or attribute-specific
schedule is introduced.

## Acceptance

1. Formal full-resolution preflight proves primary curl, primary frizz,
   secondary curl residual, secondary frizz residual, and Gaussian RGB residual
   all receive finite nonzero optimizer state when active.
2. A from-zero 30k run completes under the existing 25 GB guard.
3. Fixed QA compares R055 with R050 and R054 using identical RGB and canonical
   strand render protocols.
4. R055 is accepted only if the late secondary handoff adds useful structure
   without recreating R053's curl-back, isolated long hair, or incoherent local
   turning.

## Result

Pending formal execution.
