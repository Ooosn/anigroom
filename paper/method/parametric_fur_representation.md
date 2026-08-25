# Differentiable Parametric Fur Representation

## Paper-ready draft

We represent animal fur as a set of explicit, surface-rooted strands whose
geometry and appearance are controlled by interpretable grooming parameters.
Unlike an unconstrained collection of independent Gaussian primitives, every
strand retains a semantic root, a continuous centerline, and a root-to-tip
attribute profile. The representation is differentiable up to the discrete
choice of the number of Gaussian segments, allowing image evidence to optimize
the groom while preserving an editable asset structure.

### Surface-rooted local frame and 3D flow

For a root attached to the body mesh at position $\mathbf{x}_r$, we construct
an orthonormal local frame $\mathbf{F}_r=[\mathbf{t}_r,\mathbf{b}_r,\mathbf{n}_r]$
from the surface tangent, bitangent, and outward normal. Each primary guide
stores a normalized local direction $\mathbf{d}_r^{\mathrm{loc}}\in\mathbb{R}^3$.
Each effective render root receives this field by surface interpolation,
followed by any active local geometry residual, and the result is lifted to a
world-space grooming direction by

$$
\mathbf{d}_r =
\frac{\mathbf{F}_r\mathbf{d}_r^{\mathrm{loc}}}
{\|\mathbf{F}_r\mathbf{d}_r^{\mathrm{loc}}\|_2}.
$$

This is a genuine three-dimensional direction field rather than an
image-plane flow. Consequently, interpolation and regularization operate on
surface-aware 3D directions and remain meaningful around the belly, limbs,
tail, and other regions where a shared 2D direction would be geometrically
ambiguous.

In the active R067 hierarchy, primary guides own the semantic geometry fields
and surface-interpolate them to render roots. Secondary guides carry only
zero-centered local geometry residuals; they do not own curl turns. Render
roots own root/tip color and opacity, while the generated-Gaussian RGB
residual carries high-frequency appearance. Effective render values are
composed at render roots and are not independent copies of every semantic
geometry field.

### Normal-to-flow brush backbone

The nominal strand tip is

$$
\mathbf{x}_{tip}=\mathbf{x}_r+L_r\mathbf{d}_r,
$$

where $L_r>0$ is the editable straight endpoint length. A straight segment
is insufficient for brushed fur: real fibers leave the skin along the normal
and turn smoothly toward the local grooming direction. We therefore construct
a quadratic Bezier backbone

$$
\mathbf{B}_r(u)=(1-u)^2\mathbf{x}_r
+2(1-u)u\mathbf{c}_r+u^2\mathbf{x}_{tip}, \qquad u\in[0,1],
$$

with

$$
\begin{aligned}
\mathbf{c}^{straight}_r &= \tfrac{1}{2}(\mathbf{x}_r+\mathbf{x}_{tip}),\\
\mathbf{c}^{corner}_r &= \mathbf{x}_r+
\langle L_r\mathbf{d}_r,\mathbf{n}_r\rangle\mathbf{n}_r,\\
\mathbf{c}_r &= \mathbf{c}^{straight}_r+
s_r\,\|\mathbf{d}_r-\langle\mathbf{d}_r,\mathbf{n}_r\rangle\mathbf{n}_r\|_2
(\mathbf{c}^{corner}_r-\mathbf{c}^{straight}_r).
\end{aligned}
$$

Here $s_r\in[0,1]$ is the brush stiffness. The tangential-difference factor
continuously suppresses unnecessary curvature when the target direction is
already close to the surface normal. Thus $s_r=0$ recovers a straight strand,
while larger values produce a stronger but single, smooth normal-to-flow turn
without a threshold or a second bend field.

### Independent curl and root-to-tip opacity

We augment the low-frequency backbone with signed transverse curl. Let
$\boldsymbol{\tau}_r(u)$ be the local backbone tangent and let
$\mathbf{s}_r(u),\mathbf{o}_r(u)$ be transported orthonormal transverse axes.
With the root-preserving envelope $E(u)=u^2(3-2u)$, curl is defined as

$$
\mathbf{C}_r(u)=r_rE(u)\left[
(\sin\theta_r(u)-\sin\phi_r)\mathbf{s}_r(u)
+(\cos\theta_r(u)-\cos\phi_r)\mathbf{o}_r(u)
\right],
$$

where $\theta_r(u)=\phi_r+2\pi f_ru$, $f_r$ is the signed number of turns per
strand, and the active R067 reconstruction fixes $\phi_r=0$. Primary guides
learn the dimensionless curl ratio $\rho_r$ and signed turns $f_r$, and decode
the physical radius as $r_r=L_r\rho_r$. No secondary residual owns turns.
This keeps the learned controls geometrically comparable across short and
long fur. A nonzero phase may appear only as a fixed synthetic display
convention in the figure; it is not an editable or optimized reconstruction
control.
The differentiable centerline is the backbone plus this curl,

$$
\mathbf{P}_r(u)=\mathbf{B}_r(u)+\mathbf{C}_r(u),
$$

and the envelope preserves the root position and root tangent. The
root-to-tip opacity profile is an appearance control. For curve presentation,
the renderer uses the curve's root-to-tip Hair Info `Intercept` coordinate to interpolate
opacity in a shader mix between Transparent BSDF and the existing Principled
BSDF, preserving color and BRDF response. Gaussian presentation uses the
transported Gaussian opacities in the corresponding transparent/Principled mix.

### Root-to-tip appearance and width profile

After hierarchy composition, each effective strand has root and tip widths
$(w_r^0,w_r^1)$ and a taper exponent $\gamma_r>0$; semantic geometry values
are guide-owned and local residuals are composed at the render root. Its
continuous width profile is

$$
w_r(u)=w_r^0(1-u^{\gamma_r})+w_r^1u^{\gamma_r}.
$$

Root-to-tip color and opacity use continuous linear profiles,

$$
\mathbf{c}_r(u)=(1-u)\mathbf{c}_r^0+u\mathbf{c}_r^1,\qquad
\alpha_r(u)=(1-u)\alpha_r^0+u\alpha_r^1.
$$

These explicit profiles separate the editable fur appearance from later
Gaussian-level residual appearance used to absorb non-grooming image effects.

### Strand-to-Gaussian conversion

We adaptively resample each continuous strand according to its physical length
and geometric complexity. For two adjacent samples $\mathbf{p}_{r,j}$ and
$\mathbf{p}_{r,j+1}$, one anisotropic Gaussian is placed at

$$
\boldsymbol{\mu}_{r,j}=\tfrac{1}{2}
(\mathbf{p}_{r,j}+\mathbf{p}_{r,j+1}),
$$

oriented along $\Delta\mathbf{p}_{r,j}$, with principal scales

$$
\mathbf{s}_{r,j}=\left(
\tfrac{\kappa}{2}\|\Delta\mathbf{p}_{r,j}\|_2,
\bar w_{r,j},
\bar w_{r,j}
\right),
$$

where $\kappa$ provides a small longitudinal overlap and $\bar w_{r,j}$ is the
mid-segment width. Longer or more strongly curved strands therefore receive
more Gaussian segments automatically, while short straight fur remains
compact. Means, rotations, scales, colors, and opacities are differentiable
functions of the underlying groom parameters.

## Figure caption

**Figure X. Interpretable differentiable groom controls.** Single-variable
sweeps show the effects of 3D direction, length, brush stiffness, width
profile, curl radius, signed curl turns, root-to-tip opacity, and root-to-tip
color. The displayed geometric sweeps use $\hat L=L/L_{\mathrm{ref}}$ and
$\hat r=r/L$. Each strand is rendered over a shallow convex receiver to expose its
three-dimensional shape and contact shadow. The composed examples combine
multiple controls; the aligned lower row shows the corresponding adaptive
anisotropic Gaussian initialization after all geometric deformation.

## Implementation boundary

- Implemented active geometric controls shown in the figure: length, local 3D
  direction, brush stiffness, root/tip width and taper, and signed curl
  radius/turns. Active reconstruction fixes curl phase to zero; phase is not an
  editable or optimized control.
- Active ownership: primary guides own semantic geometry and surface-
  interpolate it to render roots; secondary guides carry zero-centered local
  geometry residuals without a turns field; render roots own root/tip color and
  opacity, and generated-Gaussian RGB residuals carry high-frequency
  appearance.
- Implemented appearance controls described in text: root/tip color and opacity.
- Not claimed as an implemented groom control: material BRDF roughness.
- The discrete adaptive segment count is detached; the generated strand and
  Gaussian attributes remain differentiable with respect to the continuous
  groom parameters.
