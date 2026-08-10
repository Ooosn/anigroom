# R056 Decoupled RGB/Flow Ownership

## Question

Can RGB-derived pseudo-flow teach legitimate direction, brush stiffness,
curl, and frizz while generated-Gaussian RGB residual absorbs image-specific
appearance, without allowing either branch to explain the other branch's
evidence?

PSNR is a secondary reconstruction diagnostic. The primary target is a clean
factorization between editable groom geometry and disposable appearance
residual.

## Frozen Parent

R055 is immutable. R056 inherits every R055 value and schedule. It does not
change flow weights, residual capacity, curl/frizz timing, root lifecycle,
interpolation, learning rates, confidence thresholds, or regularizers.

## Method

One gsplat call renders six channels with shared geometry and alpha:

```text
[final RGB with Gaussian residual, base-fur RGB without Gaussian residual]
```

The existing RGB-flow extractor sees only base-fur RGB. Its weighted loss may
update 3D direction, brush stiffness, primary curl/frizz, and secondary
direction/curl/frizz residuals. It cannot update root/tip color or Gaussian RGB
residual.

Full RGB reconstruction always updates appearance parameters. Its gradient to
all non-appearance parameters is multiplied by:

```text
1 - gaussian_rgb_residual_multiplier
```

This reuses the existing R055 handoff: geometry receives the original full RGB
gradient before appearance residual activates, then gives that ownership up
smoothly as the residual becomes active. Mask, clean-flow anchor, 3D flow
smoothness, guide/effective smoothness, and other structural losses keep their
original owners and weights.

No pseudo-flow is called ground truth. It remains noisy image evidence; the
experiment tests ownership before changing its estimator or confidence model.

## Acceptance

1. With the feature disabled, code remains behavior-identical to R055.
2. Unit tests prove RGB-flow cannot update Gaussian residual or base colors and
   full RGB cannot update geometry after the handoff reaches one.
3. Full-resolution H100 preflight proves the six-channel path and all intended
   optimizer states are finite and active.
4. A from-zero 30k run is evaluated primarily with residual-on/off RGB,
   residual images, identical 100k-strand structural audits, and canonical
   assets. PSNR is reported but is not the acceptance target.
5. R056 is accepted only if appearance residual becomes visually interpretable
   without worsening R055's backward strands, local turning, length continuity,
   or isolated curl-back artifacts.

## Status

Implementation under verification. R055 remains the accepted parent until all
R056 checks complete.
