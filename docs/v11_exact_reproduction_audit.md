# V11 Exact Reproduction Audit

## Historical Lineage

The accepted V11 result is not a from-zero V11-config run.  Its exact lineage
is:

1. `20260707020500`: run the parent `30k_rgbflow` route from iteration 0.
2. Save `checkpoint_009000.pt` at the historical stage boundary.
3. `20260707161530`: construct the V11 model with bounded child-strand RGB
   residuals, load the 9k model and RNG state, reset the optimizer, and continue
   through iteration 30000.

The exact parent upload archive is preserved at:

`D:/RTS/_tmp_upload/petsgaussianhair_sync_20260707_020044/petsgaussianhair_code_sync.tar.gz`

Its SHA-256 is:

`d495a7db6852b4167a66a899b6075c07fd55c5254a811cb28eeb53de703aa554`

The archive overlays Git commit:

`881245abcf989b71b7c20048aaa52ca6130118ef`

The parent and V11 use byte-identical trainer and strand-to-Gaussian code.
Only the V11 config and resume behavior change after 9k.

## V11 Delta From Parent

The effective configs differ only in these algorithmic fields:

- enable bounded child-strand RGB residual, scale `0.15`;
- disable child opacity residual;
- reduce bend residual scale from `0.08` to `0.04`;
- disable curl and frizz in both guide and render-root effective shape;
- delay shape detail from 9k to 14k and use bend scale `0.5`;
- reset the optimizer at the 9k handoff;
- change checkpoint frequency only.

This is child-strand appearance residual, not a true per-Gaussian RGB residual.

## Active Fixed Choices And Heuristics

### Sample-specific inputs

- white-tiger mesh alignment is fixed to scale `1.28` and translation
  `[0.0, 0.32, 0.02]`;
- native resolution is fixed to `1920x1080`;
- every sixth camera is test, producing 30 train and 6 test views;
- the accepted clean-flow target contains 4000 body guides and 500 head guides;
- initial clean-flow length is clipped to `[0.010, 0.040]`, with confidence
  threshold `0.50` and scale `0.30`;
- clean-flow lift is clipped to `[0.008, 0.040]`.

### Render-root densification

- enabled from 600 through 10000 every 100 iterations;
- uses `target_direct`: the top 32768 residual pixels are depth-unprojected to
  mesh faces, filtered by residual `>=0.010` and minimum spacing `0.0012`;
- the event budget is `512 * 2 = 1024` inserted roots;
- parents are retained, so this route is clone/insertion rather than split;
- in this route, `DENSIFY_SCORE_THRESHOLD` and
  `DENSIFY_MIN_CONTRIBUTION` do not select parents;
- historically 83 of 95 events hit the full 1024-root cap, inserting 87993
  roots total.  Growth is therefore strongly budget-limited.

### Guide-root densification

- enabled from 11000 through 16000 every 200 iterations;
- render-root need is interpolated to guide roots using 8 neighbors;
- threshold is `2.5e-5`, with an absolute cap of 32 new guides per event;
- parents are retained;
- all 26 historical events hit the 32-guide cap, adding exactly 832 guides.

### Other active constants

- 100000 initial render roots, 4500 initial guide roots, 4 child strands;
- 64 strand samples, adaptive 10-36 segments, Gaussian overlap `1.45`;
- root and guide smooth graphs use hard-coded KNN `k=8`;
- guide child placement uses a hard-coded barycentric step `0.05`;
- random mesh backing texture is enabled with strength `0.30`, 5 octaves;
- mesh-depth clipping uses absolute tolerance `0.018`, relative tolerance
  `0.004`, and local kernel `1`;
- render roots remain restricted to their original mesh face through
  barycentric optimization; cross-face root movement is not implemented.

## Present But Disabled In This Line

These code paths exist but have zero gates in both the parent and V11 runs:

- overpaint capacity;
- dark-stroke, screen-stroke, neutral-screen, and color-contrast capacity;
- early-capacity and effective-geometry budgets;
- overlong split and screen-footprint split;
- pruning;
- strand-splat orientation-map loss;
- local child opacity residual.

They are historical code surface, not active V11 behavior.

## Reproduction Gates

The wrapper must reproduce the historical metric trajectory, not only finish:

| iteration | historical test composite PSNR |
| ---: | ---: |
| 1000 | 21.1935 |
| 5000 | 23.2514 |
| 9000 | 24.0695 |
| 10000 | 28.4462 (V11 continuation) |
| 12000 | 30.0485 |
| 16000 | 31.2587 |
| 20000 | 31.9110 |
| 30000 | 32.1814 |

The final acceptance also compares the original full-resolution RGB render,
pure-fur Blender render, effective direction field, and effective length map.

## Exact Local Reproduction Result

The two-stage route was reproduced from iteration 0 on 2026-07-12 with run ID
`20260712162626`.  Phase A produced the 9k parent checkpoint; phase B loaded
that model and RNG state, reset the optimizer exactly as the historical V11
launcher did, and continued to 30k.

| result | historical V11 | local exact reproduction | delta |
| --- | ---: | ---: | ---: |
| train composite PSNR | 33.1317 | 33.1472 | +0.0155 |
| test composite PSNR | 32.1814 | 32.1938 | +0.0123 |
| render roots | 187993 | 187991 | -2 |
| generated Gaussians | 8499094 | 8498494 | -600 |

The trajectory also agrees before the handoff (`17.5149` at iteration 1,
`23.2510` at 5k, and `24.0696` at 9k) and after it (`28.4424` at 10k,
`30.0307` at 12k, `31.2619` at 16k, and `31.9180` at 20k).  The remaining
differences are consistent with CUDA reduction/order differences between the
local RTX 4080 SUPER and the historical A100 run.

Artifacts:

- final checkpoint:
  `outputs/20260712162626/phase_b_v11_9k_30k/checkpoint_030000.pt`;
- phase-A log: `logs/20260712162626/phase_a.log`;
- phase-B log: `logs/20260712162626/phase_b.log`;
- 100k-strand canonical pure-fur diagnostic:
  `outputs/20260712162626/phase_b_v11_9k_30k/visualization_exact_v11/asset_pure_fur_side_y_100000.png`.

## Speed Audit

The exact local run required about 5 h 59 min of model-training time.  The
historical A100 logs required about 3 h 38 min.  This hardware comparison is
not an algorithm ablation, so the controlled tests below all resume the same
local 30k checkpoint and time iterations 30020-30100.

| timing route | sec/iter | iter/s | change from exact |
| --- | ---: | ---: | ---: |
| exact V11 (`child_count=4`) | 0.765 | 1.31 | baseline |
| skip `RootStatsWindow.add` only | 0.764 | 1.31 | negligible |
| stop lifecycle retained gradients and accumulation | 0.683 | 1.46 | 10.7% faster |
| timing-only `child_count=1` conversion | 0.448 | 2.23 | 41.4% less time |

The accepted V11 ends with about 8.50M generated Gaussians and about 4.16M
visible Gaussians per sampled view.  The newer clean route ends with about
2.15M generated Gaussians.  Its measured median speed is about 6.96 iter/s
early and 4.52 iter/s after 20k.  Therefore the old route's four child strands
and larger adaptive strand sampling explain the largest part of the gap, but
not all of it.

Function-level profiling of five post-30k iterations found these remaining
cost centers:

- strand parameter decoding and strand-to-Gaussian construction dominate the
  forward preparation;
- gsplat projection/rasterization plus its backward dominate total GPU work;
- retaining gradients on every intermediate Gaussian remains active after
  render-root densification ends at 10k and guide densification ends at 16k;
- the trainer computes `effective_groom_graph_smoothness`, early-capacity
  staging, clean-flow length anchoring, and effective-geometry budget even
  when their configured weights are zero;
- full model gradient and parameter finite scans run every iteration;
- RGB-derived flow loss is active but measured at only about 8 ms/iteration,
  so it is not the principal slowdown;
- the actual densification event intervals do not produce a material timing
  spike.  Densification increases later per-iteration cost indirectly by
  increasing roots and Gaussians, rather than through the event operation
  itself.

The exact reproduction source must remain unchanged.  A future optimized line
can preserve behavior while gating lifecycle retention to active collection
windows, skipping zero-weight regularizers, reducing finite-check frequency,
and testing whether one strand per render root plus render-root densification
can recover V11 quality without the four-child Gaussian multiplier.
