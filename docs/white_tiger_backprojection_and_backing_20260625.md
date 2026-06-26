# White Tiger Backprojection and Backing Decision

## Fixed Alignment

All white-tiger projection, backprojection, and visualization must use:

```text
scale = 1.28
translation = [0.0, 0.32, 0.02]
```

Config: `configs/white_tiger_mesh_alignment.json`.

The nvdiffrast mesh-depth path is vertically flipped back to the dataset image convention before visibility testing. The accepted alignment visualizations are in:

```text
_downloads/white_tiger_alignment_final128_z002_allviews_20260625/
```

## External Texture-Baking Rules Used

Mature multi-view texturing systems do not average all projections blindly. MVS-Texturing / texrecon exposes geometric visibility, data-cost/view selection, seam leveling, outlier removal, and hole filling controls. OpenMVS also treats mesh texturing as a separate `TextureMesh` stage after dense mesh generation. Recent texture-generation/backprojection work also reports seams and artifacts from ad-hoc averaging.

For AniGroom Stage 1 initialization, we do not need final photo-texture atlas generation. We use the robust subset relevant to groom-parameter initialization:

- mesh-depth visibility is mandatory;
- foreground mask is mandatory;
- mask-edge confidence downweights silhouette/boundary samples;
- orientation confidence weights flow samples;
- view-angle weight favors front-facing observations;
- unobserved roots/texels stay explicitly unobserved;
- UV baking is diagnostic unless it outperforms root-direct projection.

References:

- MVS-Texturing: https://github.com/nmoehrle/mvs-texturing
- Texrecon arguments: https://github.com/nmoehrle/mvs-texturing/blob/master/apps/texrecon/arguments.cpp
- Texrecon pipeline: https://github.com/nmoehrle/mvs-texturing/blob/master/apps/texrecon/texrecon.cpp
- OpenMVS texturing stage: https://openmvg.readthedocs.io/en/latest/software/MVS/OpenMVS/
- Im2SurfTex discussion of ad-hoc backprojection artifacts: https://arxiv.org/html/2502.14006v1

## Backprojection Routes

### Route A: Root-Direct

```text
root -> camera projection -> mesh-depth visibility -> mask/confidence/view-angle weighted sample
```

Outputs:

```text
root color
root flow
root weight
visible-view count
```

This is the selected Stage 1 initialization route.

Weighted diagnostic:

```text
_downloads/white_tiger_backprojection_weighted_20260625/
```

Key result:

```text
root-direct observed fraction: 0.9671
```

The projected diagnostics show continuous color and flow over the animal surface without obvious back-side contamination.

### Route B: UV / Texture Bake

```text
texel -> mesh surface -> camera projection -> mesh-depth visibility -> weighted fusion -> sample back to root
```

Outputs:

```text
baked color
baked flow
baked weight
observed count
root sampled color/flow/weight
```

Weighted diagnostic:

```text
_downloads/white_tiger_backprojection_weighted_20260625/
```

Key result:

```text
root-from-UV observed fraction: 0.6210
direct/UV overlap fraction: 0.6010
direct-vs-UV mean color L1 on overlap: 0.1689
direct-vs-UV absolute flow dot on overlap: 0.7323
```

The atlas is highly fragmented. Even though most valid texels receive some observation, sampling the baked UV map back to FPS roots leaves many roots unobserved and visibly breaks flow near seams. This route stays as diagnostic/visualization for now and must not be forced into Stage 1.

## Correct Mesh-Backing Composition

The intended composition is:

```text
C = C_fur + T_fur * C_mesh_rand
```

where `C_fur` and `T_fur` must be produced after clipping fur samples against the visible mesh depth. This is different from passing a flat background color to gsplat.

Current gsplat version:

```text
gsplat 1.1.1
```

Its `rasterization()` supports `render_mode="RGB+D"` and `render_mode="RGB+ED"` for output depth, but it does not accept a per-pixel mesh-depth map for occlusion/clipping. Therefore `backgrounds=` alone is not a correct mesh backing implementation for the animal body.

Required rasterizer extension:

- pass visible mesh depth map, or packed per-camera depth image, into the rasterizer;
- in forward, ignore or clip Gaussian contributions behind mesh depth at the covered pixel;
- in backward, apply the same visibility/clipping mask so gradients match the rendered image;
- return fur color and transmittance/alpha so the Python side can composite with a random mesh-color render, or accept the mesh-color image as the rasterizer background after depth clipping.

Until this is implemented, random backing can only be treated as a temporary training background, not as the final physically correct mesh-body backing.

## Stage 1 Decision

Use root-direct projected initialization for Stage 1:

```text
depth visibility + mask edge confidence + orientation confidence + view-angle weight
```

Do not use UV bake as the training initialization route in the current white-tiger run.

Before multi-view training, run a minimal single-view optimization with the root-direct initialization. If single-view PSNR is below 30, treat it as a bug or missing renderer constraint and debug before proceeding.
