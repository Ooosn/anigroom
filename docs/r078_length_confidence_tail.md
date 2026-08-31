# R078 Length-Confidence Tail Gate

Status date: 2026-08-31.

Status: deferred before H100 execution. The mechanisms remain implemented and
default-off, but the experiment inherits R077's iteration-zero length release
and therefore cannot answer whether the extreme coat is caused by a length
code defect or by unlocking into an immature support state.

## Question

R077 added geometry view ownership and a confidence/intrinsic-area-weighted
mean absolute data-relative length anchor. It repaired much of the short side,
but sparse long guides still escaped. R078 tests the two distinct omissions
identified by R077:

1. suppress render-path primary-guide length updates from untrusted image
   gradients when the source guide has no stored length confidence;
2. replace the mean-only anchor reduction with tail concentration so sparse
   trusted tails are not diluted by the guide population.

The forward renderer, target, root population, schedules, and all unrelated
losses remain unchanged.

## R077 causal evidence

R077's exact guide attribution shows that view ownership and length trust are
different signals. Every guide in the top 1% ratio tail is view-owned:
`45/45` top-tail guides have at least one view owner. Among those same guides,
`24/45` have zero length confidence and `21/45` have positive length confidence.
R077's K8 local log-length discontinuity reaches q95
`0.170800` (`0.1708`).

These measurements identify two separate mechanisms. The zero-confidence,
view-owned guides expose untrusted image gradients: geometry ownership alone
still lets their length receive image reconstruction updates. The positive-
confidence tail exposes sparse trusted tails: a mean anchor is population-
diluted even when a runaway guide has legitimate positive length evidence.
The R077 local discontinuity is therefore evidence for both a
length-confidence-specific render gate and a tail-sensitive anchor reduction,
not evidence for a physical endpoint or a returned flow failure.

## Exact contract

R078 sources
`configs/r077_confidence_owned_length_0_3k_gate.env` and has exactly two
executable assignments:

```text
VIEW_GATE_LENGTH_CONFIDENCE_SUPPORT=1
CLEAN_FLOW_GUIDE_LENGTH_ANCHOR_REDUCTION=tail_concentration
```

It preserves R077's anchor weight `0.080`, `VIEW_GATE_GEOMETRY_SUPPORT=1`,
the formal V8 clean-flow target, `CLEAN_FLOW_LENGTH_INIT_SCALE=0.30`,
`GUIDE_LENGTH_FREEZE_UNTIL=0`, and the inherited non-length guide freeze
`GUIDE_FREEZE_UNTIL=9000`. It also preserves `ROOT_COUNT=400000`,
`ITERATIONS=3000`, `VIEW_GATE_NORMALIZATION=equal_owner_budget`, and
`VIEW_GATE_FLOOR=0.0`. Direction, width, child, clump, curl, and every other
non-length guide attribute keep R077's inherited schedules and freezes. All
other settings remain inherited from R077 and its source chain.

The source-guide render-path-only confidence gate is enabled by
`VIEW_GATE_LENGTH_CONFIDENCE_SUPPORT=1`. Each source guide's stored length
confidence multiplies only that guide's decoded length gradient before exact
surface interpolation on the render path. It does not alter forward values,
camera views, geometry view ownership, or the equal-owner budget. The
view-independent propagation path is retained: the data anchor and intrinsic
surface smoothness continue to propagate evidence without a view-specific
length update.

`CLEAN_FLOW_GUIDE_LENGTH_ANCHOR_REDUCTION=tail_concentration` retains the
`0.080` data-relative anchor and changes only its reduction. For
r_i=log(L_current,i/L_data,i), with detached
w_i=length_confidence_i * intrinsic_area_i, the reduction is weighted mean-L1
plus weighted L4-L2 concentration:

```text
weighted mean-L1+(L4-L2)
```

The L4-L2 term measures concentration of residual energy in a sparse tail; it
does not prescribe a target endpoint. Reliable target lengths still recover
the formal target's data identity by dividing the stored short-scale target by
`0.30`. Zero-confidence guides receive no direct anchor force, while the
existing view-independent intrinsic surface propagation remains active.

There is no physical length cap and no species, region, or view rule
(including selected views), and no percentile behavior is added to model or
training. The q05/q95 values in
the launcher are only a target-derived initialization preflight diagnostic;
they are not model thresholds or final acceptance endpoints.

## Launcher and preflight boundary

`scripts/server/run_panda_r078_length_confidence_tail.sh` is the strict H100
from-zero launcher for the bounded 0-3k gate. It requires an explicit clean
source checkout and commit, a new R078 runtime directory, the formal V8 target
and SHA-256, the Panda data/mesh/SDF inputs, and the granted CUDA device.

Before training it:

- validates the V8 NPZ schema, finite arrays, summary, and zero-new-severe
  verification;
- records input hashes and runs the full test suite;
- performs one full-resolution view-09 step through the native preflight path;
- asserts the parsed R078 fields, guide-root source, geometry/view/length-
  confidence gates, equal-owner normalization, and zero floor;
- requires finite positive anchor loss and positive reliable anchor fraction;
- derives short-scale q05/q95 from observed positive formal-target `shell_h`
  values whose normalized target weight reaches the configured confidence
  floor, then multiplies that data-relative interval by `0.30`;
- requires positive reliable-guide count and complete guide-length fill.

The preflight keeps `iterations=1` because it is a one-step gate, while the
resolved training contract remains `ITERATIONS=3000`. It asserts only
iteration-1 initialization/ownership evidence and memory safety. There is no
no final PSNR threshold and no final length threshold. Train/test metrics and
final length distributions remain measured evidence, not launcher gates.

After the gate, the launcher trains from zero through 3k, requires
`checkpoint_003000.pt`, reloads it for the canonical view-09 render, and
records render, checkpoint, and output hashes. No continuation beyond this
bounded comparison is authorized by the contract alone.

## Comparison boundary

R077 remains the matched mean-anchor control. R078 changes only the two
R077 omissions encoded by the assignments above. The experiment must be
judged with the fixed flow and asset protocols, including child-strand/clump,
RGB-to-flow or edge-style loss, and cleaned-flow guide initialization checks.
No result, PSNR endpoint, or final length endpoint is assumed by this contract.

## Deferral boundary

R078 was not deployed: no remote source, runtime, log, PID record, checkpoint,
or asset exists. Its bounded 3k contract remains recorded but is not authorized
for execution. Before launch, comparison with the normal Panda R068 schedule
and the historical R028 early-guide ablation showed that iteration-zero guide
learning is itself a confound. At equal 1000 guide-length update steps, normal
R068 iteration 10000 has effective mean/q95 `0.019797/0.034871`, while R076
iteration 1000 has `0.048385/0.073112` and R077 has
`0.045406/0.068528`. Normal unlock begins after roughly 660k roots, completed
coverage ramp, and the final lifecycle event; R076/R077 begin from 400k roots
during lifecycle and early mask correction.

Code audit also found a separate freeze semantic defect: frozen parameters keep
a zero gradient tensor and still execute Adam step, advancing state and allowing
momentum movement if moments are nonzero. This does not cause R076, where length
was never frozen, but it must be isolated rather than hidden inside R078.

R079 therefore supersedes R078 as the next diagnostic. It migrates the exact
frozen R074 3k model/optimizer state without changing any tensor, unlocks only
length at iteration 3001, and compares iteration 4000 against R076 iteration
1000 at the same 1000 length-update count.
