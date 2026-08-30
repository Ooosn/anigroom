# R077 Confidence-Owned Guide-Length Gate

Status date: 2026-08-31.

Status: contract prepared. No HGC execution or acceptance result is claimed by
this note.

## R076 causal evidence

R076 is the causal evidence for the remaining short-coat failure. It starts
from the inherited `0.30` clean-flow length scale, releases only primary-guide
length at iteration zero, and keeps direction plus every other guide attribute
under the inherited guide freeze. Length learns, but the multiview image
gradient uses uncontrolled overgrowth to close mask gaps. R076 reaches
effective length mean/q95/max `0.055631/0.097325/0.155769`, while `60.1130%`
of full strands exceed the V8 reliable q95 and `8.36582%` remain below V8
q05. Test composite falls `1.532745 dB` below R075 even though mask L1
improves. The upper-back guide and trained asset retain zero direction
reversals, so this is a length-ownership failure rather than a returned flow
failure.

R077 tests whether the existing trusted-view ownership can reach the
guide-derived geometry while a weak data-relative target anchors reliable
primary-guide length. The forward render remains unchanged; only backward
ownership and the optional anchor are added.

## Exact contract

R077 sources
`configs/r076_early_guide_length_0_3k_gate.env` and has exactly two executable
assignments:

```text
VIEW_GATE_GEOMETRY_SUPPORT=1
CLEAN_FLOW_GUIDE_LENGTH_ANCHOR_WEIGHT=0.080
```

It preserves the V8 target, `CLEAN_FLOW_LENGTH_INIT_SCALE=0.30`,
`GUIDE_LENGTH_FREEZE_UNTIL=0`, `GUIDE_FREEZE_UNTIL=9000`,
`ROOT_COUNT=400000`, `ITERATIONS=3000`, and the inherited
`VIEW_GATE_NORMALIZATION=equal_owner_budget` with `VIEW_GATE_FLOOR=0.0`.
All other fields remain inherited from R076 and its source chain.

`VIEW_GATE_GEOMETRY_SUPPORT=1` applies the existing trusted-view
straight-through gate to decoded guide-derived geometry fields. It does not
change their forward values, the view set, the renderer, or the ownership
normalization.

## Data-relative log anchor

The clean-flow target stores the initialized guide length after the inherited
short-scale factor. R077 restores the target's data identity inside the loss
without changing initialization:

```text
L_data,i = L_target,i / 0.30
r_i = log(L_current,i / L_data,i)
L_anchor = sum_i (confidence_i * intrinsic_area_i * abs(r_i))
            / sum_i (confidence_i * intrinsic_area_i)
```

The weighting is explicitly confidence * intrinsic-area weighting. Confidence
and intrinsic source-area quadrature weights are detached from the loss.
Finite positive target lengths with positive confidence contribute;
zero-confidence targets contribute no anchor force. Those unanchored lengths remain
under the existing intrinsic surface smoothness, which propagates neighboring
reliable evidence without inventing a value at the missing location.

This adds no physical length cap and no species, region, or view rule
(including selected-view rules). The target's existing reliable initialization
filter and the surface-safe graph remain the only data and topology mechanisms.

## Launcher boundary

`scripts/server/run_panda_r077_confidence_owned_length.sh` is a strict
from-zero H100 0-3k launcher. It requires an explicit reviewed source
checkout, expected source commit, new runtime directory, formal V8 target and
SHA-256, Panda data/mesh/SDF inputs, and the granted CUDA device. Before
training it verifies the V8 NPZ schema and summary, records input hashes, runs
the full test suite, and performs a one-iteration full-resolution view-09
preflight.

The preflight checks the resolved R077 fields, the equal-owner view budget,
zero floor, root count, `guide_length_frozen=false`, `guide_frozen=true`,
geometry support, and exact anchor weight. It also requires finite positive
anchor loss, a positive reliable anchor fraction, and an initial effective
length mean inside the target-derived short-scale q05-q95 interval. It does
not assert a final accepted length or PSNR; those remain measured evidence.

After the preflight, the launcher trains from zero through 3k, reloads
`checkpoint_003000.pt` for the canonical view-09 render, and records render,
checkpoint, and output hashes.

## Bounded 3k comparison

| Control | Init scale | Length timing | Geometry ownership | Role |
|---|---:|---|---|---|
| R075 | `1.0` | inherited guide freeze | off | accepted data-identity coverage/length control |
| R076 | `0.30` | released at iteration zero | off | causal uncontrolled-length control |
| R077 | `0.30` | released at iteration zero | on; anchor `0.080` | confidence-owned treatment |

Acceptance is a matched 3k comparison against R075 and R076:

1. The full suite, clean-source/runtime checks, V8 target checks, full-
   resolution preflight, from-zero training, strict checkpoint reload, and
   hash records all complete.
2. R077 retains the inherited short-scale initialization at iteration 1,
   with positive finite anchor loss and positive reliable anchor fraction.
3. Measured R077 final length distributions and local continuity are compared
   with R076's two-sided overgrowth/tail failure and R075's data-identity
   coverage. The comparison records train/test PSNR and final length
   statistics as evidence, but neither a final length endpoint nor a PSNR
   threshold is encoded as this launcher gate.
4. Fixed-protocol flow and asset checks confirm that geometry ownership does
   not disturb the accepted zero-reversal direction field or transfer a new
   species-, region-, or view-specific artifact.

No continuation beyond this bounded 3k comparison is authorized by this
contract alone.
