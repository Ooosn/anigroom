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

A float64 numerical audit over 10,000 random nonzero vectors measured maximum
absolute differences of `2.22e-15` for both old/new output equivalence and norm
preservation. At exact zero, the old path produced gradient sum `0.0`; the new
linear transport produced a finite nonzero gradient. Thus the fix preserves
the old operation away from its singular initialization point.

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

## Formal Result

The one-H100 continuation completed from the same R044 10k checkpoint with the
R046 no-geometry-regularizer configuration:

```text
/home/wangyy/anigroom-r047-vector-transport-runtime-20260808/outputs/r047_vector_transport_resume10k_16k_h100_20260808
```

| Iteration | R046 test | R047 test | R047 - R046 | R046 direction P95 | R047 direction P95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 11000 | 30.5196 | 30.5216 | +0.0020 | 0.0000 | 0.0216 |
| 12000 | 30.6025 | 30.6070 | +0.0044 | 0.0000 | 0.0391 |
| 14000 | 30.8687 | 30.8951 | +0.0264 | 0.0000 | 0.0987 |
| 16000 | 31.0552 | 31.1146 | +0.0594 | 0.0000 | 0.4359 |

Peak allocated CUDA memory was 10.32 GB. The direction residual immediately
left zero and the PSNR gain over R046 grew with training, proving the repaired
RGB gradient path. With all explicit geometry regularization disabled,
direction P95 then grew excessively. R047 is therefore a successful causal
fix but not a promotable training configuration.

## Disposition

Keep the vector-field transport fix. Do not promote the R047 no-regularizer
configuration and do not increase the 20k population. The next controlled run
must restore R045's local K4 geometry regularization while retaining the fixed
transport, testing whether the field can be both learnable and structurally
controlled.
