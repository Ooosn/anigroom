# Groom Parameterization Validation

## Purpose

This module validates whether explicit, editable groom parameters can be optimized through the differentiable strand-to-Gaussian rendering path.

It is not a white-tiger training recipe, and its iteration counts are not production defaults. The stress suite is only used to answer:

- Can the parameterization represent common animal-fur appearances?
- Can projected initialization make optimization stable?
- Which fur cases expose current weaknesses?

## Confirmed Pipeline

The validated chain is:

```text
surface roots
-> explicit groom parameters
-> differentiable strands
-> adaptive strand Gaussian segments
-> gsplat rendering
-> RGB / alpha / orientation losses
-> groom-parameter gradients
```

The current explicit parameters are:

- length
- root width, tip width ratio, width taper
- flow direction and flow strength
- lift, bend, sag, stiffness
- curl radius, curl frequency, curl phase
- frizz
- child radius and clump strength
- root color, tip color, opacity

Removed concepts such as hairness gates, darkness shortcuts, and carrier/background explanations are not part of this module.

## Stress Suite

The validation script is:

```text
D:\petsgaussianhair\tools\train_groom_style_stress_suite.py
```

The default mode is now a diagnostic validation mode:

```text
init mode: projected_curve
styles: mixed_animal, dog_guard, cowlick_whorl, patchy_length, wet_matted, dirty_tangled, curled
```

The oracle-style diagnostic initialization has been removed from the formal script. It was only used to check whether strong curl failed because of poor initialization or because of the representation/loss itself.

## Current Findings

The parameterization and differentiable optimization are sufficient for most common animal-fur fields:

| Case | What it stresses | Result |
|---|---|---|
| mixed_animal | mixed length, color, and flow | stable |
| dog_guard | plush undercoat plus longer guard-hair zones | stable |
| cowlick_whorl | local whorl / rotating flow | stable |
| patchy_length | abrupt short/long transitions | stable |
| wavy / clumped / short_plush | common groom shapes | stable |
| curled | strong curl phase detail | partially solved |
| wet_matted | high clump, sag, dark wet hair | hard |
| dirty_tangled | noisy flow, frizz, clump, color variation | hard |

Representative diagnostic outputs:

```text
D:\petsgaussianhair\_downloads\groom_style_stress_animal_projection_v1
D:\petsgaussianhair\_downloads\groom_style_stress_animal_hard_840_v1
D:\petsgaussianhair\_downloads\groom_style_stress_curled_projected_curve_840_v1
```

## Interpretation

Projected initialization is a necessary part of the method. Generic initialization can fit simple coats, but it is not robust enough for curled, frizzy, clumped, or mixed animal fur.

The goal of this module is not to exactly copy every target strand. A case is considered successful when the learned hair has:

- correct coarse flow
- plausible length and width distribution
- continuous and comfortable strand appearance
- stable clump / curl / frizz behavior
- interpretable parameters
- no dotted-chain artifacts, broken strands, or white transparent haze

## Remaining Weak Points

Strong curl, wet/matted fur, and dirty/tangled fur still expose limitations:

- curl phase detail is not fully recovered by the current sinusoidal curl representation;
- wet/matted hair is limited by alpha, width, clump, and occlusion behavior more than flow direction;
- dirty/tangled hair improves with longer diagnostic optimization but still retains orientation-detail error.

These are research directions for the next stage, not blockers for accepting the core differentiable groom parameterization.

## Next Use

This module is now a validation tool. It should be used after changing groom parameters, strand generation, adaptive Gaussian segmentation, or losses.

It should not be used as a default iteration schedule for real reconstruction. The white-tiger pipeline will run long-stage optimization with projection initialization, root lifecycle, real image losses, and additional reconstruction/asset objectives.
