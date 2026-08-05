# R003 Surface Attribute Interpolation

R003 established the interpolation contract now used by Stage 1. This file
describes the current contract only; superseded parameterizations are not
retained.

## Ownership

Guide roots own the coherent grooming field. Render roots sample that field and
may add zero-centered local residuals after their scheduled unlock. Lifecycle
children use exactly the same sampler as ordinary render roots.

## Surface Support

Interpolation is built on mesh-surface support rather than UV distance or
unrestricted Euclidean nearest neighbors. Each query receives neighboring
guide roots, intrinsic paths, and normalized weights. The support is cached
while topology and root positions are unchanged and invalidated after a guide
lifecycle update.

## Attribute Rules

- Scalars use normalized weighted interpolation.
- Positive scale-relative length uses guide interpolation followed by its
  zero-centered ratio residual.
- Colors and opacity use normalized weighted interpolation.
- Periodic phase values use circular interpolation.
- A 3D direction is parallel-transported into the query surface frame before
  weighted averaging and normalization.

Direction initialization, render-root sampling, graph smoothing, and
densification therefore share one normalized 3D vector representation. There
is no second per-root directional state.

## Lifecycle Contract

Guide lifecycle changes rebuild the interpolation support. Render-root split
children are placed on nearby mesh faces, sample the current guide field at
their new surface positions, and receive zero-centered local residual rows.
Deleting a parent does not copy its absolute endpoint into a child.

## Verification

The surface tests verify:

- no cross-sheet interpolation on nearby but disconnected mesh layers;
- direction transport across changing surface normals;
- exact preservation of a coherent direction field;
- finite gradients through interpolation and residual composition;
- stable attribute transfer after root lifecycle changes.
