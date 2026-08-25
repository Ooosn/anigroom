# R067 Runtime And Module Audit

Status date: 2026-08-26.

This document separates measured runtime evidence from method-value decisions.
It does not change the accepted R067 checkpoint, configuration, schedule,
losses, graph K, image resolution, or training code.

## Formal Runtime Lineage

| Route | Wall time | Delta from preceding listed route | Added or removed method content |
| --- | ---: | ---: | --- |
| R055 | `9582.171 s` (`2.66 h`) | - | staged primary/secondary shape and Gaussian RGB residual reference |
| R060 | `13593.378 s` (`3.78 h`) | `+4011.207 s` | R057 gradient ownership plus R058-R060 local-frame curl/frizz redesign and relative amplitudes |
| R061 | `13186.907 s` (`3.66 h`) | `-406.471 s` | removes obsolete local render-root color outlet |
| R062 | `12512.557 s` (`3.48 h`) | `-674.350 s` | adds differentiable mesh no-penetration SDF loss |
| R065 | `14770.677 s` (`4.10 h`) | `+2258.120 s` | adds exact strand-crossing discovery, active loss, and local residual routing |
| R066 | `15267.730 s` (`4.24 h`) | `+497.053 s` | learns signed primary-guide curl turns |
| R067 | `15775.028 s` (`4.38 h`) | `+507.298 s` | removes differentiable frizz |

The R055-to-R067 increase is `6192.857 s` (`1.72 h`). These runs were not
executed under a calibrated exclusive-server benchmark, so a positive wall
delta is not automatically the cost of the named method change. R067 is an
important counterexample: removing frizz cannot add frizz compute, yet its run
is `507 s` slower than R066. At least that scale of run-to-run variation must
not be assigned to a module.

R057 is excluded as a speed measurement because its held node experienced a
documented system-wide swap/page-fault storm.

## Exact Event Costs In R067

The complete formal log contains all 85 lifecycle records and 11 crossing
refresh records.

### Lifecycle

- total: `489.027 s` (`3.10%` of R067 wall time);
- selection: `284.918 s`;
- candidate KNN: `272.807 s`;
- render-row update: `168.017 s`;
- graph rebuild: `24.736 s` total, `0.291 s/event` mean;
- optimizer migration: `0.363 s` total.

Graph rebuild is already cached and is not the current problem. Candidate KNN
is the remaining lifecycle cost, but eliminating the entire lifecycle overhead
could save only `8.15 min` and would not recover the R055 runtime.

### Crossing

- 11 exact refreshes from iteration 9001 through 29001;
- refresh total: `810.248 s` (`5.14%` of wall time);
- mean/max refresh: `73.659 / 87.595 s`;
- late refreshes test about `35.5M` exact segment pairs each;
- R062-to-R065 total wall increase: `2258.120 s` (`37.64 min`).

The `810 s` refresh cost is measured directly. The remaining R062-to-R065
increase is an upper bound on active crossing forward/backward plus unrelated
run variation; it is not an isolated timing measurement.

## Module Value And Cost Decisions

### Keep: Gaussian-only appearance decomposition (R061)

R061 removes a semantically redundant render-root RGB outlet. It changes final
test composite by only `-0.02455 dB`, preserves structure, and increases the
intended Gaussian-residual contribution without saturation. It is not a speed
regression.

### Keep: mesh no-penetration (R062)

R062 reduces penetrating points by `82.43%`, penetrating roots by `38.35%`, and
mean normalized penetration depth by `84.59%`. Its formal run is actually
`674 s` faster than R061, so the current evidence does not identify SDF as the
wall-time regression source. A per-step profile may still optimize duplicate
strand construction later, but removing the loss is not justified.

### Keep representation, optimize frozen phase: local-frame curl (R058-R060)

The local-frame and relative-amplitude geometry removes R059's 34 strict
foldbacks and the short-hair scale failure. It is central to editable curl and
must not be replaced by the old root-frame formula merely for speed.

However, the full local-frame/cross/trigonometric deformation was still
executed while the shape-detail multiplier was exactly zero through the first
14k iterations. The isolated `codex/r068-zero-curl-fastpath` candidate skips
only that mathematically inactive call:

- H100 existing path median/P95: `52.556 / 52.714 ms`;
- H100 disabled path median/P95: `7.601 / 7.635 ms`;
- outputs and active length/direction/stiffness gradients: exact zero error;
- estimated upper-bound saving over 14k frozen iterations: `629.4 s`
  (`10.5 min`);
- no post-unlock speed change and no method change.

This is the only current speed candidate recommended for a formal gate before
the user decides on module removal.

### Keep: learned signed turns (R066)

Learned curl-only cumulative turn P50/P95 is `2.10/20.49 deg` versus the fixed
1.2-turn control's `24.60/129.16 deg`; backward/foldback remain zero. The extra
guide scalar is not a plausible explanation for a `497 s` wall delta by
itself. Its structural value is clear.

### Keep: no-frizz state (R067)

Removing frizz lowers final cumulative turn P95 from `68.65` to `21.20 deg`
and local-turn max from `45.39` to `3.60 deg`, with only `-0.048 dB` fixed-view
change. The slower wall time cannot be causally assigned to code that removes
deformation work.

### Reconsider as a default module: crossing (R063-R065)

Crossing is the expensive module with the weakest retained benefit:

- R062 pairs at least 45 degrees: `230`;
- R065: `198` (`13.9%` reduction);
- R067: `217` (`5.7%` below R062);
- R062-to-R065 wall delta: `+37.6 min`;
- refresh alone: `+13.5 min`.

It is valid research evidence and its local-residual ownership prevents the
failed R063/R064 length escape. The current evidence does not establish that it
belongs in the default reconstruction route after learned turns and frizz
removal. Before optimizing its exact broadphase, run one controlled R067
from-zero no-crossing ablation and inspect both exact contacts and assets. If
the visual/structural delta is negligible, crossing should move to an optional
validity refinement rather than remain a default cost.

## Rejected Mainline Speed Candidate

`codex/r068-exact-speed` packs the K32 appearance-only smoothness pass. On H100
it reduces that isolated forward/backward from `35.39` to `16.66 ms`, an upper
bound of about `9.4 min` over 30k, but raises isolated peak allocation from
`1.74` to `3.29 GB` and adds substantially more cache/fusion code. K32
appearance smoothness already existed in the 2.6h R055 route and therefore
does not explain the later regression. It is not recommended for mainline use.

## PLY Count Audit

The documented `5,382,959` is the iteration-30000 pre-step render metric. The
saved post-step checkpoint reconstructs `5,382,896` Gaussians. Three complete
H100 checkpoint reconstructions are exactly identical:

- count: `5,382,896` in all repeats;
- per-root count SHA256:
  `808fe6d404d286ca3786fb6cbf060d48dc92de7cfae360b0ea9c73ea861ccce4`;
- root/segment-order SHA256:
  `69894668f704100039e2385cb06801854b9915bd74baf77ead938941bac0f968`;
- segment range: `10-19`;
- status: pass.

The PLY is correct for the persisted checkpoint. Padding or deleting 63
Gaussians would make it incorrect.

The completed audit is stored at
`D:/RTS/_tmp/r067_acceptance_20260825/postprocess/r067_no_frizz/analysis/r067_gaussian_count_audit.json`
with SHA256
`17276d3ad57c1beac0cce453b0be1d03b04807b6d0068f1bd57b94fed4e6d989`.

## Recommended Next Decision

1. Keep R061, R062, R066, and R067 method changes.
2. Gate the simple exact-zero curl fast path; do not merge packed K32 fusion.
3. Decide whether crossing remains default by running one R067 no-crossing
   ablation before investing in a complex exact broadphase optimizer.
4. Do not spend effort further optimizing lifecycle until the larger
   geometry/crossing decisions are made.
