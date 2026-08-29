# R070: Width-Only Slenderness Ownership

Status: cancelled before commit or training. The proposed ownership correction
addressed R069's long-hair escape, but subsequent exact-crop visual review
showed that guide length was not the user's reported defect. R068 remains the
accepted single-sample baseline.

## Parent Failure

R069's guide-support gauge successfully prevents Panda's near-zero guide
length and reduces coarse root width without transferring the failure into
opacity. Its slenderness term is

```text
relu(log_width_ratio - log_length_ratio).
```

The scalar is correct, but its backward ownership is not. Gradient descent can
reduce it by narrowing width or increasing length. At the matched 12k gate the
slenderness component is `0.413904`, while the explicit length-collapse
component is only `0.004859`. R069 therefore creates `78` effective roots above
length `0.12` and `11` above `0.15`; the R068 parent has none.

## Single Change

R070 keeps the complete R069 forward value and changes only backward ownership:

```text
ell_i   = asinh(guide_length_raw_i)
omega_i = asinh(guide_root_width_raw_i)

collapse_i    = relu(-ell_i)
slenderness_i = relu(omega_i - stop_gradient(ell_i))
```

All confidence, intrinsic area weighting, fourth-moment reduction, weight
`0.001`, schedules, rendering, lifecycle, flow, SDF, appearance, and schema-10
state remain unchanged.

The forward gauge metric remains directly comparable to R069. In backward:

- collapse can only increase a guide that fell below its stored length
  reference;
- slenderness can only narrow guide root width;
- slenderness cannot create a long-hair escape through guide length.

This is a gradient-ownership correction, not an upper length penalty. It adds
no maximum length, body rule, species value, percentile, camera condition, or
new hyperparameter.

## Gates

Focused tests must prove forward equality with R069, zero length gradient from
isolated slenderness, finite corrective width gradient, retained corrective
collapse gradient, strict config lineage, and complete-suite compatibility.

The first runtime gate is the same strict from-zero Panda 12k protocol. It is
rejected if any of the following occurs:

- near-zero length or coarse width returns;
- opacity becomes the replacement escape;
- any effective length exceeds `0.12` when the matched parent has none;
- canonical view-09/view-27 coverage or noise regresses;
- PSNR is used to overrule structural failure.

Only a passing Panda gate authorizes the unchanged R070 source/weight on the
white-tiger 12k protocol. A 30k run remains out of scope until both pass.

## Cancellation

The user's exact view-09 crop is dominated by gray/black salt-and-pepper noise,
not by the scalar length pattern. In the matched R069 12k crop, hair-only alpha
is spatially continuous while raw-fur RGB remains noisy. A root-color luma
overlay places large numbers of dark render-root colors throughout nominally
white coat regions. The visible failure is therefore dense root/tip color-field
contamination (with possible geometry interaction), not an alpha hole that can
be accepted by repairing guide length.

R070 partial code and configs were removed before commit. No R070 H100 run was
launched. This document remains only as rejected ownership reasoning so the
same wrong-target branch is not silently repeated.
