# R077 Confidence-Owned Guide-Length Gate

Status date: 2026-08-31.

Status: completed and rejected as a training route. R077 corrects most of the
short-length tail, but the mean-L1 anchor and orientation-derived view owner
gate do not suppress sparse long-length escape. No continuation beyond 3k is
authorized.

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

## Attempt ledger

The first H100 invocation from source `12df16a` passed all 436 tests and the
native one-iteration full-resolution run. The model-side preflight evidence is
valid: anchor loss `1.191977`, reliable anchor fraction `0.762444`, effective
q05/mean/q50/q95/max
`0.008840/0.011025/0.011065/0.013004/0.014132`, finite gradients, and
7108.67 MB peak allocation. The outer launcher then stopped before 3k training
because it read the render-root `clean_flow_length_init_q05/q95` fields. Those
fields are intentionally zero in guide-driven initialization; the populated
guide evidence is reported by the guide-length counts. No checkpoint was
created and the qlogin allocation was preserved.

The corrected preflight derives its short-scale q05/q95 directly from the
formal target's observed positive `shell_h` values at the configured confidence
floor, then multiplies that data-relative range by the resolved initialization
scale. It also requires positive reliable-guide count and complete guide-length
fill. This changes only launcher verification, not model code, loss, schedule,
or experiment parameters.

The corrected retry uses clean source
`a4d4ac4288cf9da1339e56e7e9dffec9e6566897`, passes all 436 tests and the
corrected full-resolution gate, then completes from zero through 3k and strict
view-09 reload. The remote checkpoint is
`/home/wangyy/panda-r077-confidence-owned-runtime-20260831-retry1/outputs/panda_r077_confidence_owned_length_0_3k_h100_20260831/checkpoint_003000.pt`,
SHA-256
`2c79bca20fec862b1d23a19ab396b72e05730baf60b28ffcf6677149bd24f6f4`.

## Training result

| Iteration | Effective q05 / mean / q50 / q95 / max | Anchor loss | Train / test composite | Train / test mask L1 |
|---:|---|---:|---:|---:|
| 1 | `0.008845 / 0.011029 / 0.011073 / 0.013006 / 0.014131` | `1.191994` | `14.783858 / 15.067747` | `0.118188 / 0.115549` |
| 1000 | `0.028238 / 0.045406 / 0.043687 / 0.068528 / 0.106195` | `0.237656` | `20.066484 / 20.018856` | `0.031749 / 0.032105` |
| 2000 | `0.029305 / 0.049836 / 0.045263 / 0.084932 / 0.149588` | `0.277838` | `19.686378 / 19.661909` | `0.030921 / 0.031266` |
| 3000 | `0.029576 / 0.050888 / 0.044832 / 0.091755 / 0.186631` | `0.262583` | `19.557491 / 19.540701` | `0.031216 / 0.031476` |

The reliable anchor fraction remains `0.762444`. Metric-time roots/Gaussians
are 448917/6249881; reload has 450932/6278003. Peak allocated CUDA memory is
10789.47 MB. Compared with R076, the mean and q95 shrink, but the maximum grows
and the test composite loses another `0.840494 dB`. R075 remains substantially
better at `21.913940` test composite, `0.029063` mask L1, and
`0.036326/0.042835` effective mean/q95.

## Full asset evidence

All full-population and matched assets are complete:

- full 450932-strand NPZ SHA-256
  `c01bb125dc617e5b034b209e1fa328ada7946f9997649b592f23129a5d265370`;
- full 6278003-Gaussian PLY
  `D:/RTS/exports/panda_r077_003000_3dgs/r077_003000_full_3dgs.ply`, SHA-256
  `43a5c2172c53cd84063d3756626bf76dbcd65ff0d50ee71f7ebf220de70cebab`;
- physical-width 240k asset
  `D:/RTS/exports/panda_r077_003000_blender_asset/panda_r077_003000_240k_preview_side_y_pos.png`,
  SHA-256
  `0fff4ef817f71e2db0df89f5e94199d53ccf90ae222ecdabf036ade5a6b5d609`;
- validated 100k Blender scene SHA-256
  `d7729f89a5f45b64c125e26acc1a2260006b2583570e964202190560419caf80`.

The 240k asset and checkpoint render show long spikes and length bands on the
head, shoulder, upper-back silhouette, rump, and limbs. Root spacing remains
matched across R075/R076/R077 (`3.0241/3.0092/3.0060 mm` mean), so population
density is not the cause.

The full-strand comparison shows what R077 changes:

| Measurement | R075 | R076 | R077 |
|---|---:|---:|---:|
| Arc mean | `3.7061 cm` | `5.6762 cm` | `5.1917 cm` |
| Arc q05 / q50 / q95 | `2.9721 / 3.7241 / 4.3775` | `2.1757 / 5.4199 / 9.9513` | `3.0143 / 4.5770 / 9.3570` |
| Over V8 reliable q95 | `0.0167%` | `60.1130%` | `46.4489%` |
| Over twice V8 median | `0%` | `26.0727%` | `14.3665%` |
| Under V8 reliable q05 | `0.0120%` | `8.3658%` | `1.4938%` |
| K8 local log-length discontinuity q95 | `0.067174` | `0.115880` | `0.170800` |

Thus the data-relative anchor materially repairs the short/bald side of the
distribution, but concentrates a worse local long-length discontinuity. The
R077 global appearance change is large enough that free template matching
moves `2.431894` template diagonals away from the accepted user region. Using
the fixed same-camera R076 physical box, formal guide and R077 asset
all/front/back negative and greater-than-120-degree counts remain zero;
render-chord versus V8-guide reversals are `0/4223`. Direction remains valid;
the visible failure is length.

## Guide confidence attribution

Exact checkpoint attribution uses each run's own checkpoint schema and the
model's exact surface support, not a Euclidean proxy. R077 guide-length ratio
mean/q95 relative to its full data-identity field is `1.4168/2.7687`, versus
R076 `1.5916/2.9012`. At render roots, R077 mean/q95 is `1.4033/2.5454`.

R077 has 3431 positive length-confidence guides and 1069 zero-confidence
guides; 3995 guides have at least one view owner and 505 have none. Of the top
1% guide ratios, `24/45` are zero-confidence, `8/45` have confidence in
`(0.5,0.75]`, and `13/45` are above `0.75`; every one of the 45 has a view
owner. Guide ratio versus length confidence has near-zero Pearson/Spearman
correlation (`-0.0136/0.0736`). The R077 guide-graph local log-ratio
discontinuity mean/q95 is `0.21363/0.74182`, worse than R076
`0.16050/0.46334`.

This proves two independent omissions. Orientation-derived view ownership does
not mean the same guide has trustworthy length evidence, so zero-confidence but
view-owned guides still receive image length gradients. Confidence gating
alone is also insufficient because `21/45` top-tail guides have positive length
confidence; the mean-L1 anchor dilutes their sparse escape.

## Decision and next gate

R077 is retained as the failed confidence-owned mean-anchor control. R075
remains the accepted 3k length/coverage gate.

The next bounded gate must make two isolated, data-relative changes while
retaining R077's `0.30` initialization, weight `0.080`, view ownership, and all
other schedules:

1. multiply only the render-path primary-guide length gradient by that guide's
   stored length confidence, so zero-confidence guide length receives no image
   update while view-independent anchor and surface-smoothness gradients remain
   active;
2. replace mean-L1 anchor reduction with mean-L1 plus weighted `L4-L2` tail
   concentration in natural log-ratio space, penalizing sparse escape without
   imposing a physical endpoint or suppressing a coherent regional change.

No species, body-region, selected-view, centimeter, or percentile threshold is
introduced into model behavior.
