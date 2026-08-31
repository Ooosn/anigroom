# R080 3k->4k All-Low-Frequency-Guide Unlock

Status: bounded matched continuation contract; no final acceptance claim.

## Question

R079 showed that delaying an isolated primary-guide length release to iteration
3001 does not remove the long-coat tail after the same 1000 length updates.
R080 asks the next causal question: does releasing the normal low-frequency
guide group together at iteration 3001 let the guide fields share the physical
explanation of the image evidence?

This is a one-variable matched comparison:

- R079 iteration 4000 (length-only): the R074 iteration 3000 state receives
  1000 updates while only primary-guide length is released.
- R080 iteration 4000 (coupled low-frequency guide): it receives the same R074 3k checkpoint
  and the same 1000 updates, while the normal guide group is released together.

Both arms use the same R074 V8 target, migrated model/optimizer/RNG/lifecycle
state, coverage, camera contract, and 3000->4000 continuation window. The only
method variable is the group freeze boundary: R079 leaves GUIDE_FREEZE_UNTIL at
9000 and overrides only GUIDE_LENGTH_FREEZE_UNTIL=3000; R080 changes only
GUIDE_FREEZE_UNTIL=3000 while preserving the inherited
GUIDE_LENGTH_FREEZE_UNTIL=3000.

In compact form: R079 4k (length-only) versus R080 4k (coupled low-frequency
guide), both from the same R074 3k checkpoint and each with 1000 updates.

## Big-picture physical coupling

Guide length, direction, width, and clump are coupled physical explanations of
one strand field. If length alone is movable while those neighboring
low-frequency fields are frozen, image reconstruction can use length as a
compensating degree of freedom for mask or appearance mismatch. Releasing the
coupled low-frequency guide group lets the related physical fields share that
evidence instead of forcing length to absorb it. R080 isolates this coupling
question after the same mature R074 3k state; it does not add a cap, endpoint,
species rule, region rule, selected-view rule, or a new loss.

The shape-detail freeze remains active through the continuation, so shape detail
continues to hold curl and turns. The Gaussian RGB render residual is still on
its pre-unlock side and its gaussian_rgb_residual_multiplier remains 0.0 at
both 3001 and 4000.

## Exact config contract

configs/r080_all_guide_unlock_at3k_resume.env sources the R079 main contract and
changes exactly:

~~~text
GUIDE_FREEZE_UNTIL=3000
~~~

It therefore inherits R079's GUIDE_LENGTH_FREEZE_UNTIL=3000, ITERATIONS=4000,
STAGE_SAVE_ITERS=4000, the R074 V8 target/checkpoint contract, equal-owner
coverage, root population, lifecycle, optimizer, and RNG behavior.

configs/r080_all_guide_unlock_at3k_preflight.env sources the R080 main contract
and changes exactly:

~~~text
ITERATIONS=3001
STAGE_SAVE_ITERS=3001
SAVE_EVERY=0
TRAIN_VIEWS=9
TEST_VIEWS=9
~~~

The inclusive 3000/3001 boundary must report guide_length_frozen=false and
guide_frozen=false. It must also report shape_detail_frozen=true and
gaussian_rgb_residual_multiplier=0.0. The same assertions are required at
iteration 4000.

R080 carries no R077/R078 geometry gate, length-confidence gate, or
clean-flow guide-length anchor. The inherited resolved values remain the
inactive/default contract: geometry support false, length-confidence support
false, anchor weight 0.0, and anchor reduction mean_l1.

## Checkpoint continuity and workflow

The launcher requires the untouched original R074 checkpoint:

~~~text
/home/wangyy/panda-r074-v8-runtime-20260830/outputs/panda_r074_v8_flow_0_3k_h100_20260830/checkpoint_003000.pt
fcd62694663a7ab9383ff0250fa6a44544b7bafff1ebc96ffd7a2e05ad8d013e
~~~

After the full static test suite, the schema12-to-schema14 migration utility runs
once. It must preserve exact model, optimizer, optimizer-parameter-name, RNG,
and lifecycle sections. The migrated source guide_length_raw Adam state must
remain at step 3000 with zero exp_avg and exp_avg_sq moments. The migrated
checkpoint is passed read-only to both the one-step preflight and the full
continuation; its hash is checked after each run.

The launcher validates the V8 target schema/hash, runs the full pytest suite,
renders migrated iteration 3000 view 9, runs the 3000->3001 preflight, resumes
the same migrated checkpoint through iteration 4000, renders view 9 again, and
writes checkpoint, input, output, preflight, and render hash manifests. Bash
syntax for the shared trainer is checked before execution.

## Context and boundary

R028 provides historical context: early guide optimization produced useful
metric evidence but changed width, opacity, and long-hair structure together, so
it is not a clean group-coupling comparison. R068 provides the normal delayed
guide-unlock context after coverage and population mature; its low-frequency
fields are the relevant physical baseline, while its zero-curl fast path is an
implementation detail rather than a new R080 variable.

R080 is judged with the fixed child-strand/clump, RGB-to-flow or edge-style,
cleaned-flow initialization, render, and metric review protocols. There is no final metric threshold,
no final PSNR threshold, and no final length threshold in this contract. Metrics
and final length distributions remain measured evidence for review.

## Attempt ledger

The first H100 invocation passes all 468 tests, exact migration, migrated-3k
render, optimizer/RNG resume, and the iteration-3001 all-guide preflight. The
first-step guide fields and length remain finite and near initialization. The
outer validator then stops before the 4k continuation because it asserts the
generic baseline shape-detail freeze `14000`, while the inherited R074 contract
is `20000` with unlock end `25000`. The corrected validator records the actual
parent values; no model, config behavior, checkpoint, or optimizer state is
changed.
