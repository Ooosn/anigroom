# R057 RGB-Flow Gradient Ownership

## Question

Can RGB-derived flow supervise geometry without directly changing root/tip
color or generated-Gaussian RGB residual?

## Frozen Parent

R055 is immutable. R057 inherits its forward render, final-RGB flow source,
losses, weights, schedules, interpolation, lifecycle, capacity, and learning
rates exactly.

R057 does not inherit R056's residual-free flow image or its RGB-to-geometry
gradient attenuation. R056 demonstrated that removing RGB geometry evidence as
appearance residual activates causes a sharp reconstruction collapse.

## Single Change

Each iteration still uses one render and one optimizer step. Backward is routed
in two parts:

1. RGB and all non-flow regularizers backpropagate normally.
2. Weighted RGB-flow loss backpropagates only to optimized non-color
   parameters.

The excluded color family is exactly:

- root color
- tip color
- optional child color delta
- generated-Gaussian RGB residual

All optimized non-color parameters retain both their original RGB gradient and
their RGB-flow gradient. No gradient multiplier, projection, extra render,
base-fur flow source, or optimizer step is introduced.

## Acceptance

1. A focused gradient test proves geometry receives `RGB + flow`, while base
   color and Gaussian RGB residual receive RGB only.
2. One Adam step advances every active tested owner exactly once.
3. The full-resolution H100 preflight completes with the flag recorded in its
   checkpoint and finite nonzero active optimizer states.
4. The from-zero 30k run is compared against synchronized R055 metrics and the
   same fixed RGB, residual, attribute, strand, and canonical asset protocol.
5. R057 is accepted only after both reconstruction and strand structure are
   inspected. PSNR alone is not sufficient.

## Status

Implementation verified locally; formal H100 preflight and 30k run pending.
