# R036: Hierarchical Positive Child Spread

Status: formal from-zero run complete; frozen measured Stage 1 baseline.

## Base And Scope

R036 keeps R034's accepted absolute uncapped strand-segment allocation and
R035's hierarchical width profile. It removes the last active animal-scale
endpoint in the ordinary short-fur route: the fixed physical child-radius
interval.

This is a complete ownership change rather than a decoder-only edit:

- guide roots own a positive child-spread field around a local reference;
- render roots own a zero-centered relative child-spread residual;
- zero render residual exactly reproduces guide interpolation;
- guide and render lifecycle updates interpolate physical values and local
  references, then re-encode the child coordinates;
- guide, render-residual, and final-effective smoothness use logarithmic
  coordinates on the intrinsic surface graph;
- checkpoint schema and optimizer ownership reject the retired direct bounded
  render field.

No body part, image region, percentile, absolute child radius, or
white-tiger-specific threshold is introduced.

## Parameterization

For local guide reference `c_ref`, guide raw coordinate `g`, render raw
coordinate `r`, and the existing coverage unlock `s`:

`c_guide = c_ref * exp(asinh(g))`

`c_effective = c_guide * exp(s * child_scale * asinh(r))`

Both values are strictly positive and have no physical upper endpoint. The
reference only defines the zero coordinate and is propagated through surface
interpolation; it is not a clamp. A zero render coordinate means exact guide
following.

## Training Ownership

Child spread controls coverage, not late strand shape. It therefore retains
the existing coverage schedule: the render residual ramps from 1k to 7k.
Moving it into the lifecycle-aware residual container must not accidentally
freeze it until the 10k geometry-residual phase. Guide child spread remains a
guide field and follows the existing guide-freeze and guide-lifecycle rules.

No new schedule or loss weight is added. Existing weights act on dimensionless
log-ratio coordinates rather than a normalized absolute physical interval.

## Lifecycle Contract

Guide densification interpolates physical child spread and its local reference,
then encodes the new guide raw coordinate. Render densification interpolates
the zero-centered render coordinate and migrates surviving optimizer rows while
initializing only new rows. This preserves the same effective field before and
after topology changes when no new evidence has been optimized.

## Verification

- complete repository suite: `58 passed`;
- Python compilation: passed;
- positive, finite, unbounded guide child spread: passed;
- zero render residual equals guide child spread: passed;
- render child residual receives gradients during the coverage phase: passed;
- direct bounded render child field is excluded from the formal optimizer:
  passed;
- guide lifecycle preserves child spread and reference encoding: passed;
- strict checkpoint state includes the hierarchical child-spread schema:
  passed.

The strict H100 run started from zero and completed all 30k iterations with no
resume or fallback. Its final checkpoint, optimizer state, config snapshot,
metrics, initialization reports, and intermediate checkpoints are present.

| Run | Final train composite | Final test composite | Best test composite |
| --- | ---: | ---: | ---: |
| R032 reference | 33.28816 | 32.47268 | 32.66519 at 29k |
| R035 parent | 33.44020 | 32.65968 | 32.84404 at 29k |
| R036 | 33.42397 | 32.66322 | 32.83977 at 29k |

R036 ends with `317245` render roots and `14,215,421` Gaussians, slightly below
R035's `318942` roots and `14,278,095` Gaussians. Peak live PyTorch allocation
was `24096.82 MiB`, with `27166 MiB` reserved. There is no population or memory
explosion.

The final child-spread distribution is positive and finite:

- min / p05 / p50 / p95 / max:
  `0.002152 / 0.003054 / 0.004171 / 0.006623 / 0.010108`;
- mean / standard deviation: `0.004332 / 0.001080`;
- the typical render-root log-ratio residual remains centered at zero:
  p05 / p50 / p95 is `-0.001185 / 0.000002 / 0.001256`.

The parent R035 direct field reached `0.011903`, nearly the retired `0.012`
endpoint, and also contained values close to zero. R036 has no such endpoint,
yet learns a narrower coherent distribution without collapse or expansion.
This is the intended result: the hierarchy, interpolation, and evidence choose
the physical spread instead of a decoder interval choosing it.

The fixed V11-protocol pure-fur render uses four child strands, deterministic
100k sampling, 32 points per strand, and the canonical Blender camera/material:

- `D:/RTS/_tmp/r036_30k_final/r036_030000_asset_side_y_v11_protocol.png`
- `D:/RTS/_tmp/r036_30k_final/r036_030000_asset_side_y_pos_v11_protocol.png`
- `D:/RTS/_tmp/r036_30k_final/r036_030000_asset_front_z_v11_protocol.png`

The matched parent render is:

`D:/RTS/_tmp/r035_30k_final/r035_030000_asset_side_y_v11_protocol.png`

It shows no new curl, isolated child-spread burst, or width collapse relative
to R035 or the accepted reference. Both sides and the orthogonal view retain a
continuous coat with no hidden one-sided spread failure. Broad shoulder,
belly, and tail-root length regions remain visible, but they are continuous
fields and pre-exist the child-spread change; they are not repaired by
reintroducing a radius cap.

The retired formal training interval was `[0, 0.012]`; the generic field
default also carried a separate `0.018` endpoint. R036 removes both. Neither
number is used by the new model or its acceptance logic.

## Schedule Boundary

The measured R036 snapshot and frozen local baseline retain the historical
child-spread coverage ramp from 1k to 7k. Width profile, length, direction, and
the other render geometry residuals retain the shared 10k-to-20k geometry
ramp. R037's broader proposal to classify all coverage fields under the early
ramp was never formally run and remains deferred.
