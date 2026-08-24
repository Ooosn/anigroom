# R067 No Frizz

Status: contract only. No R067 training, metric, visual, or acceptance result
is claimed by this document.

## Objective

R067 is a clean reconstruction removal of frizz from the accepted R066
learned-curl-turns route. It asks whether the R066 mainline can retain curl,
radius, crossing, lifecycle, renderer, and appearance behavior while removing
all differentiable frizz state and ownership. This is not a scale-zero mode.

## Ownership Contract

- `DecodedGroom` and `GroomParameterField` contain no frizz field, parameter, or
  persistent seed buffer.
- `RenderGeometryResidualField`, primary-guide controls, secondary-guide
  interpolation, effective composition, `build_strands`, and `deform_backbone`
  contain no frizz value or argument.
- The Adam optimizer, lifecycle row migration, smoothness, prior, loss,
  finite-state, crossing ownership, checkpoint state, and optimizer-name list
  contain no frizz key.
- `frizz_backbone` remains only as a standalone procedural post-edit utility;
  it is deterministic and differentiable but disconnected from the mainline.
- Crossing owns only local direction and curl-radius residuals.

## Config And Schema

The R067 config inherits R066 and unsets exactly these two effective keys:

```json
{
  "GUIDE_FRIZZ_RESIDUAL_SCALE": {"r066": "1.0", "r067": null},
  "SHAPE_FRIZZ_SCALE": {"r066": "1.0", "r067": null}
}
```

No value is replaced by zero. `CURRENT_CHECKPOINT_VERSION` is 9. Schema 8,
old config mappings containing frizz keys, frizz model-state keys, frizz
optimizer names, migration aliases, and resume compatibility are rejected
before model/config/optimizer reconstruction.

## Formal Contract

`scripts/server/run_r067_no_frizz.sh` must require a clean exact checkout,
fresh runtime/output, the `mygs` interpreter, frozen data/mesh/SDF, schema 9,
unlimited virtual memory, full pytest, full 1920x1080 active-path preflight,
and an uninterrupted from-zero 0-30k run. It must not allow resume, fallback,
reduced resolution, reduced preflight, or hidden frizz arguments.

## Postprocess Contract

The R067 component report contains backbone, learned curl-only, fixed-1.2 curl,
primary curl, and final curl variants. Foldback, RGB, structure, crossing,
signed guide-turn, and Blender protocols retain their R066 contracts with no
frizz fields in output metadata. Historical R066 docs, configs, run-specific
scripts, tools, tests, and artifacts remain evidence and are not rewritten.

## Required Tests

The suite must prove no frizz keys in model state, effective groom, residual,
config, checkpoint, optimizer names, or postprocess metadata; exact config
delta; schema-8 rejection; strict schema-9 loading; curl-only build behavior;
standalone procedural utility isolation; direction-plus-curl crossing
ownership; and lifecycle/Adam state transport without frizz.
