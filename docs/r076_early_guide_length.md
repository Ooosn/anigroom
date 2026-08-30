# R076 Early Guide-Length Gate

Status date: 2026-08-30.

Status: bounded 0-3k candidate. The four owned files are the config,
launcher, focused contract test, and this note. No later training stage is
authorized by this gate.

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
