# R062 Mesh No-Penetration

Status date: 2026-08-14.

Status: formal candidate prepared on top of accepted `stage1-r061`; native
H100 preflight and the single from-zero 30k comparison are pending.

## Single Variable

R062 inherits every R061 setting and enables one geometric validity term:

```text
mean(ReLU(-SDF(x)) / diagonal(SDF bounds))
```

The mean covers non-root samples from a deterministic cyclic block of 16,384
current render roots per step. The reviewed global weight is 256. There is no
body-part mask, length condition, penetration tolerance, or separate schedule.

The frozen SDF SHA256 is:

```text
766e177fbeeb89fc779292f56662c7c6b256f7d4365415baa366cef04af10530
```

## Formal Gate

1. Run the exact 1920x1080 renderer and backward path for two active-path
   iterations on H100.
2. Verify collision gradients reach guide geometry, secondary geometry, and
   root barycentric coordinates, but not global translation or scale.
3. Record time and peak allocated-memory delta against the frozen R061
   full-resolution preflight.
4. Run exactly one uninterrupted from-zero 30k comparison.
5. Compare R061/R062 fixed-view RGB, canonical pure-strand assets, lifecycle,
   foldbacks, crossings, curl/frizz structure, and all-root penetration.

R062 is not accepted merely because the run finishes. Missing or mismatched
SDF data, disabled collision with nonzero inputs, malformed sampled depth, or
lost gradient ownership is a hard error and has no fallback.
