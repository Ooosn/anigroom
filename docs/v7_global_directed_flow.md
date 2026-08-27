# V7 Global Directed Flow

Status date: 2026-08-28.

Status: accepted initialization target. Implementation, local regression,
formal Panda/white generation, fixed-view inspection, and graph-streamline
acceptance all pass. V6 remains the immutable fixed-axis parent and V5 remains
the completed trained rollback.

## Reopened defect

The V6 trusted-view-cluster target repaired the two reported Panda view-27
tangent-axis discontinuities, but its acceptance metrics treated directions as
axes through `abs(dot)`. The user then identified nearby arrows whose shafts
were coherent while their heads pointed in opposite directions. Exact
parallel-transported graph auditing confirmed that this is a directed-field
defect rather than a visualization artifact:

- Panda V6 final observed graph edges: `1754 / 54244` directed-negative
  (`3.2335%`);
- severe reverse edges, defined as dot at most `-cos(45 deg)`:
  `1025 / 54244` (`1.8896%`);
- directed angle P99 / maximum: `155.48 / 179.93 deg`;
- the corresponding unsigned-axis P99 / maximum was only
  `74.16 / 89.84 deg`, so the old metric hid the reversals.

The stage trace also showed why V6 regressed after its discrete cleaner:

| Stage | Directed-negative observed edges |
| --- | ---: |
| discrete cleaned | 551 (`1.016%`) |
| continuous-ratio pre-consensus | 1380 (`2.544%`) |
| final fixed-ratio | 1754 (`3.234%`) |

Of the `377` edges newly negative in the last fixed-ratio transition, `304`
had at least one accepted ratio-update endpoint. The ratio guard used an axial
angle and therefore could not see this failure.

The saved anchor field was also not a valid sign-confidence field. It had been
merged with ratio/evidence coverage: `89.15%` of observed roots were marked as
direction anchors, including `75.35%` of final negative-edge endpoints. V7
therefore computes sign observability directly and does not reuse that merged
confidence as a sign anchor.

## Method contract

V7 separates three owners:

1. the V6 robust multiview cluster owns the unsigned tangent axis;
2. global surface orientation lifting owns one binary tangent sign per root;
3. fixed-sign multiview least squares owns the final nonnegative
   normal/tangent ratio.

For tangent axis `a_i`, surface normal `n_i`, nonnegative ratio `rho_i`, and
binary sign `s_i`, the final direction is

`d_i = normalize(rho_i * n_i + s_i * a_i)`.

### Exact multiview sign unary

At the provisional final ratio, V7 projects both `rho*n+a` and `rho*n-a` into
every contributing camera. For saved direct axis `o_vi` and weight `w_vi`, the
per-root unary is

`h_i = sum_v w_vi * (dot(p_plus, o_vi)^2 - dot(p_minus, o_vi)^2)`.

This measures actual head-tail observability at the final shell point. It does
not use species, body region, view index, image coordinate, or the merged
legacy anchor confidence.

### Surface pairwise term

For an intrinsic graph edge `(i,j)`, `a_j` is parallel transported from
`n_j` to `n_i`. The signed coupling is

`J_ij = w_ij * dot(a_i, PT(a_j))`.

The binary objective is

`sum_(i,j) J_ij*s_i*s_j + alpha*sum_i h_i*s_i`.

`alpha` is sample-adaptive: the q90 absolute current pairwise field is divided
by the q90 absolute unary field, then multiplied by the fixed public factor
`0.5`. The same rule is used for Panda and white tiger.

### Trust-preserving supernodes

A single-root flip can temporarily create a reverse boundary even when a
coherent patch flip is correct. For every edge that is not severely reversed
in the provisional field, V7 evaluates both exactly-one-endpoint flips. If
either would make the edge severe, the endpoints are unioned and share one
flip variable. Optimization therefore acts on trusted connected blocks rather
than individual arrows.

The equality construction gives an explicit invariant: no edge that was
non-severe before global orientation may become severe afterward. The final
generator raises instead of writing a target if this invariant fails.

### Canonical deterministic solve

Root storage order is not method input. V7 constructs an exact float32
canonical identity from surface point, face id, and barycentric coordinate,
canonicalizes graph edges and block members, and uses canonical float64 sums.
At each step it flips only the block with the largest strictly positive exact
objective gain. Geometry-derived hashes break exact ties; root array indices
are never used.

Three deterministic root permutations and a reversed view order produce zero
sign mismatches on both samples. Objective differences are at most
`3.64e-12`; mapped direction differences are below `0.035 deg`.

### Fixed-sign directed ratio refit

After global signs are frozen, V7 recomputes the same analytic, uncapped,
nonnegative multiview ratio LS solution. Eligible roots are ordered by
normalized direct-residual improvement and canonical rank. An update is
accepted only if:

- its direct residual improves strictly;
- every incident edge that was non-severe after sign lifting remains
  non-severe;
- its maximum incident **directed** angle does not increase;
- the ratio and direction remain finite and the ratio remains nonnegative.

No `abs(dot)` appears in this guard. The formal generator also raises if the
final pass introduces any severe edge.

## Diagnostic experiment ledger

Every arm used identical Panda/white settings. No sample-specific fallback was
accepted.

| Arm | Outcome |
| --- | --- |
| whole-surface maximum spanning tree | rejected: changed 1996 observed white roots and worsened white direct mean by `+3.11 deg` |
| unconstrained per-root Ising, multiplier 0.5 | rejected: repaired Panda crop `35 -> 4` but introduced 56 severe white edges in 13 new patches |
| per-root zero-new-severe guard | rejected: blocked the coherent Panda patch; crop stayed `35 -> 35` |
| connected-block ascent in input order | numerical pass, rejected as formal method because root permutation changed the local optimum |
| canonical best-block ascent | accepted diagnostic: exact root/view-order invariance and zero new severe edges |
| canonical sign plus directed ratio refit | accepted diagnostic: sign field preserved, direct evidence improved or remained within gate |

Canonical sign-only diagnostic:

| Metric | Panda V6 | Panda sign | White V6 | White sign |
| --- | ---: | ---: | ---: | ---: |
| observed negative edges | `3.305%` | `2.057%` | `4.447%` | `3.633%` |
| observed severe edges | `1.940%` | `0.836%` | `1.396%` | `0.854%` |
| newly severe edges | - | `0` | - | `0` |
| resolved severe edges | - | `322` | - | `174` |
| all-view direct mean delta | - | `+0.005 deg` | - | `-0.003 deg` |

The Panda view-27 direct-evidence crop changed from `35` severe screen-neighbor
reversals to `0`; the four-root upper-back region changed `3 -> 0`.

Canonical post-sign ratio diagnostic:

- accepted ratio updates: Panda `554`, white tiger `536`;
- new severe edges relative to the sign field: `0 / 0`;
- all-view direct mean delta versus V6: Panda `+0.003 deg`, white
  `-0.113 deg`;
- Panda crop and upper-back severe reversals remain zero;
- upper-back four-root weighted view-27 mismatch improves from the frozen-ratio
  sign candidate `45.87 -> 42.90 deg`. It remains `+4.93 deg` above V6, a
  recorded local image-evidence tradeoff rather than a hidden regression.

## Formal implementation and local validation

- global orientation module:
  `anigroom/flow/global_sign_orientation.py`;
- directed ratio refit:
  `anigroom/flow/view_cluster_refinement.py`;
- formal target integration:
  `tools/fuse_gpt_flow_shell_multiview.py`;
- implementation commit: `1664857`;
- HGC launcher commit: `ba07109`;
- HGC launcher working-directory repair: `0712587`;
- local focused core/integration tests: `37 passed`;
- local implementation full regression: `332 passed`, `14` existing dependency
  warnings;
- final baseline-inclusive full regression: `333 passed`, the same `14`
  existing warnings;
- local reliable-runner logs:
  `D:/RTS/_tmp/anigroom_v7_local_tests_20260828`.

Final local test-log SHA-256:

`3fa8c4e15562649d13c2e984e1680a0fa6d2cd89f7160b84800e6be7cf206b81`

## Formal HGC execution

Runtime root:

`/home/wangyy/anigroom-global-directed-v7-20260828`

Source commit:

`0712587e2c32c621f5566b7a8706c9dc061fc85b`

The first launch used the correct source but invoked pytest with the checkout
path while keeping the compute shell's parent working directory. Collection
therefore failed with 44 `ModuleNotFoundError` errors before target generation.
The exact failed log is preserved under
`failed_runs/collection_import_20260828`; no output or GPU work was produced.
Commit `0712587` changes only the test invocation to enter the trusted checkout.
The downloaded failed-log SHA-256 is
`9f4d074aef9acd3ceed9cc06ba75d663dae2627cb4b45bd5f3fd2c0d522e2264`.

The retry completed at `2026-08-28T00:50:26+09:00` with `332` tests passing,
Panda and white target generation complete, no traceback/OOM/non-finite or
invariant failure, and both held qlogin allocations still running.

Formal outputs:

| Artifact | SHA-256 |
| --- | --- |
| Panda target | `6a220f52b15ca996c88e71802d3309f9499ade442f79dc72300f1af12b5fa56f` |
| Panda summary | `ac5474a6d4157c30cdf9de04dd7121493b84a84dd8874797a49ab70d503614b3` |
| Panda view-27 overlay | `def4c9c2c0d691a234b252dbd62b983d75b4925960287341c9a7c7b822bb9d19` |
| White target | `f009af820560adf19b6eedbb8bf2c5d29df00cca576be13161b4ee2ebaed6510` |
| White summary | `850e062dca5a92318a0807450e30c07b7980770f2cadd019d6c0dfe6dbf4d9a6` |
| White view-27 overlay | `fb4492ca348ef83cd6fdba60671c6ef11f4ab2e176e6a687cd212a5c42eec62b` |

Formal counts:

| Sample | Observed | Global sign changes | Resolved severe edges | Postratio accepted | New severe edges |
| --- | ---: | ---: | ---: | ---: | ---: |
| Panda | 4194 | 112 | 322 | 527 | 0 |
| White tiger | 4407 | 62 | 174 | 490 | 0 |

Local downloaded acceptance root:

`D:/RTS/_tmp/anigroom_v7_formal_results_20260828`

Formal generation-log SHA-256:

`69bdd5d9a9845f3e87210fcb411aa30cdbaee91d6bff5d1d617c6062f68b1598`

The held qlogin jobs `127114669` and `127181739` remain allocated and must not
be released unless the user explicitly asks.

### Fixed-view gate

The exact prior visual protocol generated views `00`, `09`, `18`, and `27` for
both samples from the formal targets. Each view directory contains the anchor
and cleaned arrow overlays, two strand images, and one report. All eight views
passed manual inspection without a new directed patch. The visual-gate log
SHA-256 is
`36e3abfda3c4147991130b0585e9ac5a3ba7ff76a1a2c3ddfeb1e0ecd05e1142`.

Remote visuals:

`/home/wangyy/anigroom-global-directed-v7-20260828/visuals`

Local visuals:

`D:/RTS/_tmp/anigroom_v7_formal_results_20260828/visuals`

### Streamline gate

A deterministic discrete surface audit traces every observed root for at most
64 successor steps. The successor is chosen only by positive tangent alignment
with an intrinsic graph-neighbor displacement, so the neighbor direction does
not bias continuity measurement.

| Metric | Panda V6 | Panda V7 | White V6 | White V7 |
| --- | ---: | ---: | ---: | ---: |
| selected severe transitions | 113 | 75 | 144 | 113 |
| immediate two-cycles | 27 | 20 | 32 | 26 |
| path-step P50 | 13 | 14 | 15 | 15 |
| in-degree P99 | 3 | 3 | 4 | 4 |

In the Panda view-27 crop, selected severe transitions become `13 -> 0` and
two-cycles become `2 -> 0`. The full audit, reliable-runner records, hashes,
and original-resolution diagnostic are under
`D:/RTS/_tmp/anigroom_v7_streamline_audit_20260828`.
The streamline report SHA-256 is
`b001ea3ff526e76658cc757f2960d86c3038c63bdd5430bd9a4171a08dd64b31`.

## Acceptance gates

V7 replaces the initialization baseline because all of the following hold:

1. complete Panda and white-tiger target generation exits zero from the
   reviewed Git commit;
2. both summaries report the global and postratio zero-new-severe invariants;
3. all arrays are finite and observed-root populations are preserved;
4. fixed original-resolution views `00`, `09`, `18`, and `27` pass without a
   new directed patch;
5. Panda view-27 crop and upper-back directed reversals remain zero;
6. surface streamline integration shows no unsupported backtracking or new
   convergence seam;
7. exact target/summary hashes and all commands are recorded here;
8. V5 remains the trained rollback and V6 remains the immutable fixed-axis
   parent.
