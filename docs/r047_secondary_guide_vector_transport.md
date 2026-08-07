# R047 Secondary-Guide Residual Vector Transport

## Status

Implementation candidate built from the R046 diagnostic branch. R043 remains
the accepted baseline until a formal run validates this fix.

## Root Cause

Secondary direction residuals initialize at the zero vector. Their interpolation
path used the unit-direction transport operation:

```text
normalize(v) * norm(v)
```

Although this equals `v` away from zero, its autograd derivative at `v = 0` is
zero because both the restored magnitude and the normalized vector evaluate to
zero. Consequently RGB/render losses could not start a secondary direction
residual from its exact zero initialization. R044/R045 direction values were
started indirectly by effective-groom and clean-flow geometry losses, masking
the broken RGB gradient path.

## Fix

Add a linear Rodrigues transport for arbitrary vector fields. It rotates the
input without normalizing it, preserves magnitude, and has a valid nonzero
gradient at the zero vector. Keep the existing normalized transport for true
unit directions.

Use the vector-field operation consistently for:

- secondary-guide direction-residual interpolation;
- direction-residual graph smoothness;
- render-root lifecycle inheritance of direction residuals.

No root count, interpolation K, graph K, loss weight, learning rate, renderer,
or schedule changes are part of R047.

The formal run uses
`configs/r047_secondary_guide_vector_transport_resume10k_16k.env`, which
inherits the R046 diagnostic configuration without changing any value. This
keeps the code-level transport fix as the only experimental variable.

## Gate

1. Unit tests must prove nonzero residual-direction gradient at exact zero.
2. Resume the same R044 10k checkpoint and compare R047 at 12k/14k/16k against
   R043, R044, R045, and R046.
3. Direction residual must leave zero through RGB supervision even when the
   explicit secondary geometry losses remain disabled.
4. Inspect PSNR and asset structure together before considering promotion.
