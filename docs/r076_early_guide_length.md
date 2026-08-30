# R076 Early Guide-Length Gate

Status date: 2026-08-31.

Status: completed and rejected as a training route. The bounded experiment
proves that length can learn immediately from the `0.30` initialization, but
the current image-gradient ownership lets it trade uncontrolled long/short
length variation for silhouette coverage. No continuation beyond 3k is
authorized.

## Question

R074 keeps the accepted V8 confidence-guided direction target and uses the
inherited `CLEAN_FLOW_LENGTH_INIT_SCALE=0.30`. That produces a deliberately
short initialization while the guide controls remain frozen through iteration
9000. R075 instead inherits R074 and changes only the initialization scale to
`1.0`, establishing the data-identity full-shell comparison.

R076 asks the remaining timing question: can the `0.30` short initialization
learn guide length during the first 3k iterations while V8 direction and all
other guide attributes stay frozen? This isolates early length ownership from
R075's scale change.

## Exact contract

R076 sources
`configs/r074_v8_confidence_flow_0_3k_gate.env` and has exactly one executable
override:

```text
GUIDE_LENGTH_FREEZE_UNTIL=0
```

Everything else is inherited. In particular, the resolved contract must keep:

- `CLEAN_FLOW_LENGTH_INIT_SCALE=0.30`;
- `GUIDE_FREEZE_UNTIL=9000`;
- `ROOT_COUNT=400000`;
- `ITERATIONS=3000`;
- `VIEW_GATE_NORMALIZATION=equal_owner_budget` and zero floor;
- the formal V8 clean-flow target supplied through `CLEAN_FLOW_TARGET`.

At iteration 1, the preflight must report
`guide_length_frozen=false` and `guide_frozen=true`. The first flag proves
that only guide length has been released; the second proves that direction and
the other guide attributes remain frozen under the inherited guide freeze.

## Launcher and acceptance boundary

`scripts/server/run_panda_r076_early_guide_length.sh` requires an explicit
source commit, new runtime directory, V8 target, target SHA-256, and granted
CUDA device. It validates the V8 NPZ/schema and summary, records input hashes,
runs the full test suite, performs the full-resolution view-09 preflight,
trains from zero through 3k, reloads the checkpoint for view-09 rendering,
and records checkpoint, render, and output hashes.

The gate is valid only if the target/source hash, equal-owner budget, length
scale, both freeze settings, root population, and iteration-1 metric flags
pass together. R074 and R075 remain the comparison controls; this experiment
does not change the V8 target, root population, view ownership, or later-stage
schedule.

## Attempt ledger

The first H100 launch passed all 426 tests and the native one-iteration
full-resolution preflight. It then stopped before training because the outer
launcher incorrectly asserted that the preflight's saved `iterations` field
must be 3000. `STAGE1_PREFLIGHT_ONLY=1` intentionally resolves that field to
1. The useful preflight evidence is valid: `guide_length_frozen=false`,
`guide_frozen=true`, effective length mean `0.0109705`, 400000 roots, finite
gradients, and 7111.63 MB peak allocation. No checkpoint or training output was
created. The retry requires `preflight iterations=1`; the inherited config and
required `checkpoint_003000.pt` retain the formal 3k training contract.

The corrected retry ran from clean source
`c3806b59a109a0b7f1fa2bc92f67f0e96aa5ce65` and completed without a traceback,
OOM, or scheduler failure. The remote checkpoint is
`/home/wangyy/panda-r076-early-length-runtime-20260830-retry1/outputs/panda_r076_early_guide_length_0_3k_h100_20260830/checkpoint_003000.pt`,
SHA-256
`4b83c9e0058498f084fc5b28448b8d9acccadb5f5ffbe8b768a33c1a4893afd0`.
The held qlogin allocation was preserved.

## Training result

Only guide length was released. Direction and every other guide attribute
remained frozen, as proved by `guide_length_frozen=false` and
`guide_frozen=true` at every saved metric.

| Iteration | Effective mean | q05 / q50 / q95 | Train / test composite |
|---:|---:|---:|---:|
| 1 | `0.010956` | `0.008823 / 0.011016 / 0.012874` | `14.7660 / 15.0506` |
| 1000 | `0.048385` | `0.020374 / 0.049101 / 0.073112` | `20.4223 / 20.3472` |
| 2000 | `0.054303` | `0.021028 / 0.053131 / 0.091209` | `20.3394 / 20.2860` |
| 3000 | `0.055631` | `0.021349 / 0.053157 / 0.097325` | `20.4415 / 20.3812` |

The final minimum/maximum is `0.006785/0.155769`. Lifecycle processing grows
the initial 400000 roots to 445850, with 6382954 Gaussians after reload and
10510.44 MB peak allocated CUDA memory. Root population is therefore in the
same range as R075; it is not the cause of the visual difference.

R076 improves test mask L1 from R075 `0.029063 -> 0.024758`, proving that long
strands can cover the silhouette, but loses `1.532745 dB` test composite
(`21.913940 -> 20.381195`). The configured `guide_prior_loss` is identical in
R074, R075, and R076 (`1.72817e-5`) even though R076 length escapes. Code audit
confirms why: that prior owns the render residual, not primary-guide length;
the active primary-guide term is only spatial smoothness. The R072/R073 view
gate likewise owns root position, opacity, and lifecycle evidence, but does not
gate the guide-derived length used by image reconstruction. R076 therefore
receives competing length gradients from every training view.

## Asset and numeric acceptance

All exports are complete and hash-verified:

- full 445850-strand NPZ SHA-256
  `9e0fd6f31b3eea52baa92a9aceb2e4e0eec4d1cc0a77ff7ea52dcaa6be529f1a`;
- full 6382954-Gaussian PLY
  `D:/RTS/exports/panda_r076_003000_3dgs/r076_003000_full_3dgs.ply`, SHA-256
  `dcb56a1757474f778e7905b081ba9db4b259bbef2db49410e2f313e05e018955`;
- physical-width 240k asset
  `D:/RTS/exports/panda_r076_003000_blender_asset/panda_r076_003000_240k_preview_side_y_pos.png`,
  SHA-256
  `cfd502a01baf1aa732b19fcf4c9aa9b646c010b975e92c21e406135a50c1571f`;
- validated 100k Blender scene SHA-256
  `231f2221003e8500c542096c49792a28b3b8fc52253d9a0b31336c98f59092b2`.

The full-strand audit separates density from length. R075 and R076 have nearly
identical sampled nearest-root spacing means (`3.0241` and `3.0092 mm`), while
R076 arc-length mean/std becomes `5.6762/2.3928 cm` versus R075
`3.7061/0.4136 cm`. Relative to the formal V8 reliable range:

- `60.1130%` of R076 strands exceed V8 q95, versus `0.01669%` for R075;
- `26.0727%` exceed twice the V8 median and `1.43591%` exceed three times it;
- `8.36582%` are simultaneously below V8 q05;
- K8 local log-length discontinuity q95 rises `0.067174 -> 0.115880`.

This two-sided tail and local discontinuity match the long ridges, short gaps,
and jagged upper silhouette in the 100k/240k assets. R076 reduces exposed-mesh
baldness by overgrowing the field rather than learning one coherent coat.

The original user upper-back flow region remains directionally valid. Formal
guide and trained asset all/front/back negative counts are all zero; every
greater-than-120-degree count is zero; render-chord versus V8-guide reversals
are `0/4228`; and the front/back projected mean dot is `0.995696`. The observed
asset defect is therefore length variation, not a returned flow reversal.

## Decision and next gate

R075 remains the accepted data-identity length/coverage gate. R076 is retained
as the causal control showing that unconstrained early guide-length learning is
not identifiable under the current multiview backward path.

The next gate must remain species-, region-, and view-independent. It will:

1. extend the existing trusted-view straight-through ownership to the
   guide-derived geometry fields, with unchanged forward rendering;
2. softly anchor reliable primary-guide lengths to the target's own unscaled
   shell height in log-ratio space, weighted by length confidence and intrinsic
   source area;
3. leave zero-confidence lengths unanchored so the existing intrinsic surface
   smoothness propagates neighboring reliable evidence.

No physical length endpoint, Panda region mask, or hand-selected view is added.
