# R079 3k->4k Length-Unlock Continuation

Status: completed and rejected as a length-only route. Delaying the isolated
length release from iteration zero to 3001 improves reconstruction but still
reproduces the long-coat distribution after 1000 length updates.

## Question

R076 released primary-guide length from iteration zero and exposed the
uncontrolled early length-ownership behavior. R079 asks whether the same
length-update budget behaves differently when it starts from the completed,
historical R074 V8 state at iteration 3000.

The matched comparison is:

- R076 iteration 1000: 1000 length-update steps from a from-zero state;
- R079 iteration 4000: 1000 length-update steps, iterations 3001–4000, after
  the R074 state was frozen through iteration 3000.

Both arms use the same V8 direction target. R079 therefore compares equal
length-update exposure while retaining a frozen 3k state, rather than
confounding the comparison with another target or a from-zero restart.
In compact form: R076 iteration1000 versus R079 iteration4000, with the same
V8 target and the historical step3000 state preserved by R079.

## Contract

`configs/r079_length_unlock_at3k_resume.env` sources the R074 V8 config and
changes exactly:

```text
ITERATIONS=4000
GUIDE_LENGTH_FREEZE_UNTIL=3000
STAGE_SAVE_ITERS=4000
```

The inherited `GUIDE_FREEZE_UNTIL=9000` keeps every non-length primary-guide
attribute frozen during this arm. Coverage, root population, lifecycle,
optimizer settings, RNG behavior, and the formal V8 target remain inherited.
Schema14 defaults must resolve the later R077/R078 additions to their
inactive values: `view_gate_geometry=false`,
`view_gate_length_confidence=false`, clean-flow guide-length anchor weight
`0`, and reduction `mean_l1`.

`configs/r079_length_unlock_at3k_preflight.env` sources the main continuation
config and changes exactly:

```text
ITERATIONS=3001
STAGE_SAVE_ITERS=3001
SAVE_EVERY=0
TRAIN_VIEWS=9
TEST_VIEWS=9
```

The preflight runs one resumed step from 3000->3001. The inclusive freeze
boundary must consequently report `guide_length_frozen=false` and
`guide_frozen=true` at 3001.

## Checkpoint continuity

The launcher requires the untouched original R074 checkpoint and its expected
SHA-256:

```text
/home/wangyy/panda-r074-v8-runtime-20260830/outputs/panda_r074_v8_flow_0_3k_h100_20260830/checkpoint_003000.pt
fcd62694663a7ab9383ff0250fa6a44544b7bafff1ebc96ffd7a2e05ad8d013e
```

After the full test suite, the externally owned
`tools/migrate_stage1_schema12_checkpoint.py` utility is invoked once. Its
schema12->schema14 output checkpoint and migration report live under
`runtime/contracts`. The launcher verifies schema14, iteration 3000, the
480292-root source state, report hashes, report tensor identity, and exact
identity of model, optimizer, RNG, and lifecycle sections. It also verifies
that the source `guide_length_raw` Adam state has step 3000 with zero
`exp_avg` and `exp_avg_sq` moments.

The same migrated checkpoint is passed read-only to both the 3000→3001
preflight and the separate 3000→4000 resume. Its hash is checked after each
run. Optimizer and RNG resume events, the lifecycle-history prefix, metrics at
4000, and `checkpoint_004000.pt` are required.

## Separate freeze-state hypothesis

The freeze-state bug is a separate hypothesis from uncontrolled length
ownership: a checkpoint/config reconstruction could mishandle the inclusive
3000/3001 boundary or accidentally reset the guide-length optimizer clock.
R079 isolates that possibility with explicit start-iteration and freeze flags,
the source Adam step/moments check, and a preserved historical step 3000.
The arm does not rewrite the historical tensors, reset optimizer/RNG state, or
change the non-length guide freeze.

## Evidence and boundary

The launcher runs the full tests, performs the resumed one-step assertions,
renders view 9 at migrated iteration 3000 and output iteration 4000, and
writes input, checkpoint, output, preflight, and render hashes. No final length threshold and no final PSNR threshold are encoded here.
Any visual or metric decision remains a separate review of the matched
evidence.

## Attempt ledger

The first H100 invocation from source `8864915` stopped at the mandatory test
gate before migration. It passed 462 tests and failed only the R078 document
contract because the later deferral edit no longer contained the literal phrase
`bounded 3k`. No migrated checkpoint, preflight checkpoint, training process,
or render was created; the original R074 checkpoint and V8 target hashes stayed
unchanged, and the qlogin allocation was preserved. The retry requires the full
463-test suite to pass from a new source/runtime/log/PID path.

The second invocation from `2e35db8` passed all 463 tests and stopped inside
the migration utility before writing an output. The utility incorrectly
required every Adam parameter ID to have a state entry. R074 legitimately has
24 optimizer parameters but only 23 states because
`gaussian_rgb_residual.raw` had never received a non-`None` gradient and Adam
initializes state lazily. Guide length itself has a complete state with step
3000 and zero moments. The corrected migration accepts stateless declared
parameters while still rejecting undeclared state IDs and requiring the exact
guide-length state. No checkpoint, render, or training output was created.

The third invocation from `4a126b4` passed all 464 tests, completed exact
migration, rendered migrated iteration 3000, resumed optimizer and RNG, and
successfully wrote the iteration-3001 checkpoint. Its effective
q05/mean/q50/q95/max is
`0.008857/0.011029/0.011079/0.012956/0.014491`, so the unlock boundary does not
immediately produce long hair. The outer validator then stopped because setup
events are emitted under `setup_progress`, while it searched only `progress`.
No 3000-to-4000 continuation started. The corrected validator accepts either
event channel without changing training or checkpoint state.

The fourth invocation from clean source `207717f` completes through
`R079_DONE`. Migration preserves every model and optimizer tensor digest; the
original R074 checkpoint remains untouched. The migrated checkpoint SHA-256 is
`e015ea65c1dc473ff1b23e385f8c810ba70cde9deaf3b26244a127e2bf2fc63b`,
and the final iteration-4000 checkpoint SHA-256 is
`c9069dd98129f4e27eac34de0dcf3055b210c333ac4bbd8505ee559957e5d101`.

## Completed comparison

Iteration 3001 remains at the frozen short state:
q05/mean/q50/q95/max
`0.008857/0.011029/0.011079/0.012956/0.014491`. After the matched 1000
length updates, iteration 4000 becomes:

| Measurement | R076 1k | R077 1k | R079 4k | Normal R068 10k |
|---|---:|---:|---:|---:|
| Mean length | `0.04838` | `0.04541` | `0.04199` | `0.02006` |
| q95 length | `0.07311` | `0.06853` | `0.07170` | `0.03392` |
| Maximum | `0.09811` | `0.10620` | `0.11381` | `0.09071` |
| Test composite | `20.34724` | `20.01886` | `21.60187` | `26.96223` |
| Test mask L1 | `0.02788` | `0.03211` | `0.02110` | `0.00927` |
| Roots | `408472` | `408952` | `496183` | `669143` |

R079 therefore disproves the narrow hypothesis that iteration-zero release is
the sole cause. A frozen 3k state still reaches a q95 near 7.2 cm when length is
the only unlocked low-frequency guide attribute. The normal 10k route differs
in a more fundamental way: it unlocks the coupled low-frequency guide group
together after coverage and population have matured. R080 isolates that group
coupling from the same R074 3k checkpoint.
