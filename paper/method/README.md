# Parametric Groom Method

This directory is the single source of truth for the parametric groom method
text and its paper figure.

## Frozen figure contract

`fig_parametric_groom_controls.{pdf,svg,png}` is the accepted paper figure.
Its geometry, camera, lighting, material, typography, spacing, and panel layout
are frozen as of 2026-08-13. Do not change the figure or its two rendering
scripts unless a later paper revision explicitly reopens this artifact.

## Retained artifacts

- `parametric_fur_representation.tex`: manuscript-ready method section.
- `parametric_fur_representation.md`: readable method draft.
- `fig_parametric_groom_controls.pdf`: paper figure used by LaTeX.
- `fig_parametric_groom_controls.svg`: editable vector export.
- `fig_parametric_groom_controls.png`: 600 DPI review export.
- `fig_parametric_groom_controls.tex`: standalone LaTeX figure environment.
- `render_parametric_fur_figure.py`: the only figure assembly entry point.
- `render_parametric_groom_blender.py`: figure-specific Blender renderer.

The figure builder calls the formal groom geometry in `anigroom/grooming/`
and delegates presentation-only material, lighting, receiver, and Gaussian
outline rendering to the adjacent Blender script. Neither script contains an
alternative strand geometry implementation. Intermediate NPZ, Blender renders,
and render reports are disposable files under the selected `--work-dir`; they
are not paper assets.

The accepted figure's detail appearance control is an honest root-to-tip alpha
profile. Its differentiable centerline is the brush backbone plus signed curl.
Curve presentation maps the curve's root-to-tip `Parametric` coordinate to a
Transparent/Principled shader mix; Gaussian presentation uses the transported
Gaussian opacities in the corresponding transparent mix.

## Reproduce

Run from the repository root:

```powershell
conda activate mygs
python paper/method/render_parametric_fur_figure.py
```

The canonical paper export uses the low-saturation smoked-champagne fur
palette. Pass `--palette copper` to reproduce the previous copper rendering;
geometry, lighting, camera, and layout are identical between the two.

The command regenerates the formal PDF, SVG, and PNG in this directory.
