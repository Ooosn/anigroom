# Parametric Groom Method

This directory is the single source of truth for the parametric groom method
text and its paper figure.

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

## Reproduce

Run from the repository root:

```powershell
conda activate mygs
python paper/method/render_parametric_fur_figure.py
```

The command regenerates the formal PDF, SVG, and PNG in this directory.
