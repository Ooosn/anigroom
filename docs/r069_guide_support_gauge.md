# R069: Guide Support Gauge

Status: causal evidence, schema-10 implementation, local regression, and
checkpoint counterfactual rendering complete; matched short training gates are
pending. R068 remains the accepted single-sample execution baseline. R069 is
not yet an accepted method.

## Failure

Panda R068+V7 completes cleanly and improves directed initialization, but its
30k checkpoint contains a real upper-back bald patch with coarse surviving
hairs and speckled appearance compensation. The defect begins after the 9k
primary-guide freeze boundary:

- at 9k every primary-guide length and root-width coordinate is exactly at its
  stored reference;
- at 30k, `74 / 4500` primary guides are at or below one quarter of their own
  clean-flow-derived length reference;
- those `74` guides have mean/median root-width ratios `9.368x / 9.797x`;
- primary-guide log length and log width ratios correlate at `-0.7374`;
- the worst guide slenderness expansion reaches `96.11x`;
- the affected guides have mean length-evidence confidence `0.746`, above the
  global `0.569` mean;
- secondary residuals change length only over about `0.938x..1.076x` and width
  over `0.997x..1.003x`.

The failure is therefore a primary-guide optimization escape: reliable length
support collapses while width expands. Existing graph losses penalize local
differences but are nearly blind when a connected surface region drifts
coherently. Shape detail and Gaussian RGB residual then compensate over the
damaged support and expose high-frequency noise.

## Single Method Change

R069 adds one soft primary-guide support-gauge loss. For the existing positive
reference-relative coordinates,

```text
ell_i   = asinh(guide_length_raw_i)
omega_i = asinh(guide_root_width_raw_i)
q_i     = A_i * (0.25 + 0.75 * clamp(length_confidence_i, 0, 1))

collapse_i    = relu(-ell_i)
slenderness_i = relu(omega_i - ell_i)

M4(x; q) = (sum_i q_i |x_i|^4 / sum_i q_i)^(1/4)
L_support = M4(collapse; q) + M4(slenderness; q)
```

`A_i` is the existing intrinsic guide-graph source-area weight. The candidate
weight is `GUIDE_SUPPORT_GAUGE_WEIGHT=0.001`.

The first term activates only when a guide becomes shorter than its own stored
clean-flow reference. The second activates only when width grows faster than
length relative to their respective stored references. Equal positive global
length/width rescaling remains exactly zero. The fourth moment prevents a fixed
small bad population from being diluted, while continuous confidence and area
weights avoid anchor thresholds and sampling-density dependence.

R069 adds no decoded clamp, absolute length or width endpoint, body label,
image coordinate, camera index, species rule, selected-root count, or second
schedule. It changes no V7 flow target, interpolation, renderer, lifecycle,
SDF, curl, opacity, appearance, or secondary residual behavior.

## Why This Is Not R029

Rejected R029 anchored the complete guide-length deviation in both directions
and materially reduced reconstruction before render residuals unlocked. R069
is one-sided and coupled:

- guide lengthening remains free;
- ordinary stored clean-flow length variation remains free;
- width can grow only in proportion to supported length growth;
- only the observed collapse and short-wide compensation receive the new
  gradient.

## Diagnostic Calibration

The unweighted loss is exactly zero in Panda at 9k. At 30k its reference-based
value is `2.3481` for Panda and `1.1793` for white tiger, so weight `0.001`
contributes approximately `0.00235 / 0.00118` at the failed checkpoints. This
is comparable to, rather than orders of magnitude above, the existing weighted
guide-smooth contribution.

## Validation Gates

Before any formal 30k run:

1. focused tests must prove zero initialization/equal-scale behavior,
   corrective finite gradients, continuous confidence weighting, area and
   permutation invariance, population-stable sparse response, strict config
   recording, and exact weight-zero inactivity;
2. a checkpoint-only hard projection onto the same support inequalities must
   improve the Panda view-09/view-27 bald region without relying on a new
   renderer or appearance ablation; this is causal evidence only, not a formal
   method result;
3. Panda must pass a from-zero H100 gate through the guide/geometry transition,
   with checkpoint-native length, width, opacity, RGB-residual, and connected
   low-support maps;
4. white tiger must run the same source, weight, resolution, schedule, and
   diagnostics; no sample-specific parameter is allowed;
5. PSNR cannot authorize promotion when pure-fur support, slenderness, opacity,
   or canonical visuals fail.

The strict checkpoint config records the new loss weight. If this changes the
exact checkpoint config schema, the candidate uses a new strict schema and
trains from zero; schema-9 R067/R068 checkpoints remain frozen evidence and are
not silently migrated.

## Implemented Contract

R069 introduces no model tensor. It adds the explicit
`GUIDE_SUPPORT_GAUGE_WEIGHT` config value, so the exact checkpoint config schema
is bumped from 9 to 10 rather than silently defaulting a missing field. Schema
8 and historical schema 9 are rejected before model loading by current source.
The R069 config inherits R068 and assigns only:

```text
GUIDE_SUPPORT_GAUGE_WEIGHT=0.001
```

The weight-zero path skips graph-area acquisition and gauge evaluation and is
the numerical R068 behavior. Compilation, launcher/config syntax, focused
gradient and population tests, historical contract tests, and the complete
repository suite pass: `346 passed, 14 warnings`.

## Checkpoint Counterfactual

A diagnostic-only hard projection modifies only the final Panda 30k primary
guide length/width coordinates in memory and renders original-resolution views
09/27. It is not the soft R069 training loss and is not a formal result.

| arm | changed length / width guides | view09 / view27 composite PSNR | visual finding |
| --- | ---: | ---: | --- |
| baseline | `0 / 0` | `30.052 / 29.544` | coarse noisy upper-back patch hides missing support |
| length support only | `345 / 0` | `29.726 / 28.937` | extending the same over-wide strands produces darker coarse streaks |
| slenderness only | `0 / 3730` | `25.587 / 24.184` | finer strands expose a large low-alpha upper-back hole |
| combined projection | `345 / 3730` | `27.351 / 26.580` | coarse noise is reduced, but the hidden coverage hole remains |

The rerendered baseline differs from the preserved view-09 PNG in only
`22 / 6,220,800` uint8 channels, all by one level; this is recorded as bounded
rasterization quantization, not byte identity. The counterfactual report is:

`D:/RTS/_tmp/panda_r068_v7_acceptance_20260828/counterfactual_support_gauge/counterfactual_support_gauge_report.json`

SHA-256:
`2c008b2400cf388517a5cf27e4201123540605f85fc4dbb44563fc254ce10fa6`.

This confirms that length collapse and width inflation must be corrected
together. It also exposes a coupled width-opacity shortcut: from 9k to 20k,
effective mean width grows `0.000160 -> 0.000545` while render-root opacity P05
falls `0.745 -> 0.345` and tip opacity P05 falls `0.334 -> 0.050`. The short
training gate must therefore verify that the soft gauge lets RGB/mask evidence
retain opacity and fine-strand coverage. If opacity or uncovered projected
support becomes the new escape route, R069 is rejected rather than hidden by
PSNR or a post-hoc width clamp.

The first gate ends at 12k because the baseline failure is already measurable
there: effective minimum length is `0.00270`, mean/P95 root width is
`0.000367 / 0.000718`, and root/tip opacity P05 is `0.571 / 0.173`. Panda and
white tiger must use the same from-zero source, weight, and audit protocol.
The duration-only gate config is
`configs/r069_guide_support_gauge_0_12k_gate.env`; its shell snapshot differs
from the 30k candidate only by `ITERATIONS=30000 -> 12000`.
