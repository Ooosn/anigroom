# White Tiger Stage 1 Long-Strand Diagnosis

This note is diagnostic only. It records evidence for the current white tiger Stage 1 failure mode without changing the baseline training behavior.

## Observed Failure

The best current single-view run reaches good image metrics but still has visible long streaks and color dragging:

- `server_20260626015952_view09_rgb_iter3000`
- raw PSNR: `31.558`
- composite PSNR: `32.689`
- visible issue: some strands become long, wide, and opaque enough to paint across neighboring texture regions.

## Evidence

At `iter=3000` in the best run:

- root count: `18352`
- Gaussian count: `2162267`
- segment mean: `14.73`
- length mean / p95 / max: `0.0691 / 0.1060 / 0.2140`
- root width mean: `0.0006963`, close to the configured upper range `0.00075`
- opacity mean: `0.733`

This means PSNR is being helped by high-capacity appearance/shape parameters, especially length, width, and opacity. The visual artifact is consistent with long opaque strands carrying one root color across a larger image region.

Lifecycle records show another issue. In the same family of runs, densification starts by growing roots, then pruning removes many more roots than each split inserts:

```text
500:  20000 -> 20256, inserted 512, prune 256
600:  20256 -> 20512, inserted 512, prune 256
700:  20512 -> 20768, inserted 512, prune 256
800:  20768 -> 21024, inserted 512, prune 256
900:  21024 -> 21280, inserted 512, prune 256
1000: 21280 -> 21536, inserted 512, prune 256
1100: 21536 -> 21792, inserted 512, prune 256
1200: 21792 -> 20958, inserted 512, prune 1346
1300: 20958 -> 20166, inserted 512, prune 1304
1400: 20166 -> 19414, inserted 512, prune 1264
1500: 19414 -> 18699, inserted 512, prune 1227
1600: 18699 -> 18020, inserted 512, prune 1191
```

The root count falls sharply after pruning starts, so fewer roots must explain the same image. This encourages the remaining strands to become longer/wider/more opaque.

The new lifecycle diagnostic probe also shows that selected densification parents are already high-contribution/high-visibility roots:

```text
global contribution mean: 207.24
selected contribution mean: 368.21
global visible mean: 391.94
selected visible mean: 695.33
selected need mean: 0.001524
global need p95: 0.000864
```

So the current densification rule primarily subdivides already-visible, high-contribution roots. It is not explicitly driven by local image residual or uncovered holes.

An experimental residual-evidence switch was added on branch `codex/diagnose-white-tiger-long-strands` and kept default-off (`--densify-residual-weight 0.0`) so the baseline is unchanged.
It projects each root into the current view, samples the RGB/mask residual only if the root is mesh-depth visible, and records this value in the lifecycle diagnostics.

Short probe, same 12-iteration view09 setup:

```text
residual OFF:
  global need/residual/contribution/visible: 0.000257 / 0.000000 / 207.23 / 391.94
  selected need/residual/contribution/visible: 0.001506 / 0.000000 / 368.62 / 695.38

root-projected residual ON:
  global need/residual/contribution/visible: 0.184554 / 0.184298 / 207.24 / 391.94
  selected need/residual/contribution/visible: 0.890048 / 0.889423 / 380.10 / 713.93

coverage-pooled residual ON:
  global need/residual/contribution/visible: 0.520453 / 0.520196 / 207.24 / 391.94
  selected need/residual/contribution/visible: 1.131650 / 1.131004 / 376.60 / 706.94

sample-normalized lifecycle, residual OFF:
  global need/contribution/visible/samples: 0.000142 / 207.24 / 391.93 / 391.93
  selected need/contribution/visible/samples: 0.000872 / 365.92 / 690.35 / 690.35
  selected contribution-per-sample: 0.5354 vs global 0.3190

pixel-to-root residual ON:
  global need/residual/contribution/visible: 0.061842 / 0.061585 / 207.24 / 391.94
  selected need/residual/contribution/visible: 1.001053 / 1.000000 / 381.48 / 717.57
  selected contribution-per-sample: 0.5319 vs global 0.3190
```

This confirms the baseline is not changed when the switch is off. It also shows that naive root-projected residual is not enough: it raises residual evidence, but the selected parents are still the highest-visible/highest-contribution roots. In other words, point-sampling residual at the root projection still follows the already-rendered streaks and strong texture edges instead of reliably identifying under-covered holes.

The first coverage-pooled version (`alpha deficit + pooled RGB residual`) is also not enough. It reduces the residual ratio gap compared with root-pixel residual, but selected parents are still `1.82x` the global contribution and `1.80x` the global visible sample count. The evidence is still too tied to the current rendered hair layer.

The sample-normalized lifecycle probe shows that per-sample normalization alone also does not fix early parent selection. At iteration 10, selected roots are not meaningfully longer than average yet (`~1.00x` length), and retained sample count is almost the same as visible count. The selected roots are still `1.77x` the global raw contribution and `1.76x` the raw visible count. So sample-count feedback is a later amplifier, not the initial trigger.

The first pixel-to-root version also fails. It collects high residual pixels and assigns them to nearby visible root projections, but the selected parents remain even more biased toward already high-visible/high-contribution roots (`1.84x` global contribution and `1.83x` global visible count). The selected residual saturates to `1.0`, which means the evidence is concentrated on a small set of nearby visible roots rather than distributing structural demand to missing surface regions.

This exposes a second issue in the split placement path. `select_densify_parents` chooses parents by score, but `propose_split_children` then places children by local surface emptiness around each parent. It does not receive the image-space residual pixel positions or a residual-directed target. Therefore even when evidence is computed from under-covered pixels, the new roots are not explicitly placed toward those pixels. The current densification is parent-centric, not hole-centric.

A default-off target-placement probe was then added. It keeps the parent selection unchanged, but when `--densify-target-placement-weight > 0`, high residual pixels are unprojected through mesh depth into local surface targets. Split candidates are scored by both local surface spacing and distance to the accumulated target for that parent.

Short probe, 32 iterations, densify at `10/20/30`, same full-resolution view09 setup:

```text
pixel-to-root residual, target placement OFF:
  final composite PSNR: 17.552
  final mask L1: 0.1061
  selected/global contribution ratios: 1.841, 1.837, 1.826
  selected/global visible ratios:      1.831, 1.820, 1.803

pixel-to-root residual, target placement ON, weight=2.0:
  final composite PSNR: 17.683
  final mask L1: 0.1045
  selected/global contribution ratios: 1.841, 1.843, 1.828
  selected/global visible ratios:      1.831, 1.820, 1.799
  child target-distance improvements:
    iter 10: parent 0.005254 -> child 0.002169
    iter 20: parent 0.005604 -> child 0.001877
    iter 30: parent 0.005792 -> child 0.001946
```

This proves the placement half of the repair is real: children are being moved much closer to the residual surface targets, and the short run improves slightly. It also proves the repair is incomplete: parent selection remains high-visible/high-contribution because the same selected roots are still chosen before placement. The next step is not more scalar weighting; it is to change the candidate generation/selection so holes can nominate nearby surface roots/faces directly, instead of always starting from roots that already render strongly.

A target-parent selection probe was then tested. It keeps the target-placement path, but ranks parents by accumulated target weight instead of the original score. This still did not break the bias:

```text
target parent selection, 32 iterations:
  final composite PSNR: 17.660
  final mask L1: 0.1046
  selected/global contribution ratios: 1.846, 1.837, 1.834
  selected/global visible ratios:      1.835, 1.816, 1.803
```

So assigning residual pixels back onto existing roots is still the wrong abstraction: the residual targets are absorbed by nearby high-visibility roots before the structure update can add genuinely new coverage.

A default-off direct target insertion diagnostic was added next. It does not replace any selected parent root. Instead, high residual pixels are unprojected through mesh depth directly onto visible mesh faces, and those surface points become new roots. The nearest old root is used only for attribute inheritance. This path is enabled only with `--densify-parent-selection target_direct`; the default remains the original score mode.

Short probe, same 32-iteration setup:

```text
target placement OFF:
  final composite PSNR: 17.552
  final mask L1: 0.1061
  roots: 20768

target placement ON:
  final composite PSNR: 17.683
  final mask L1: 0.1045
  roots: 20768

target parent selection:
  final composite PSNR: 17.660
  final mask L1: 0.1046
  roots: 20768

direct residual-pixel-to-face insertion:
  final composite PSNR: 17.734
  final mask L1: 0.1025
  roots: 21536
  inserted direct roots per event: 512
  mean nearest-parent distance for inheritance: 0.00545-0.00561
```

This is the clearest diagnostic so far. Direct surface insertion is better than only changing parent score or child placement, and it reduces the mask error most in the short probe. It also keeps memory normal (`~3.4 GB` max allocated in this run), so the earlier memory issue is not part of this direct insertion path.

This is still not the final fix. Direct insertion currently uses top residual pixels directly, with only mesh-depth visibility and nearest-root attribute inheritance. Before it becomes a formal training path, it needs proper deduplication/diversity on the surface, a stable spacing rule, and a prune schedule that does not immediately remove useful low-contribution newborn roots. But it identifies the real direction: hole densification should be residual-pixel/mesh-face driven, not parent-root driven.

A surface-spacing version of direct insertion was then tested. It keeps the residual-pixel-to-mesh-face path, but rejects new roots that are too close to existing roots or to already accepted new roots in the same event. This is controlled by `--split-min-child-distance`; the default value remains `0`, so the baseline path is unchanged unless the diagnostic flag is explicitly set.

Short probe, 32 iterations:

```text
direct insertion, no surface spacing:
  final composite PSNR: 17.734
  final mask L1: 0.1025
  roots: 21536

direct insertion, surface spacing = 0.004:
  final composite PSNR: 18.040
  final mask L1: 0.0964
  roots: 21482
  accepted/rejected direct targets:
    iter 10: insert 512, valid 3631, reject existing 307, reject new 1606
    iter 20: insert 512, valid 2161, reject existing 192, reject new 1075
    iter 30: insert 458, valid 1845, reject existing 238, reject new 1149
```

This is an important result: the spaced version inserts fewer roots by the third event, but improves both image and mask metrics. The improvement is therefore not just "more roots"; it is better-placed roots. Surface diversity is part of the fix.

Longer 120-iteration probe with direct insertion and `0.004` spacing:

```text
iter 40:  composite PSNR 18.583, mask L1 0.0928, length p95 0.0681, width p95 0.000207, opacity p95 0.804
iter 80:  composite PSNR 22.975, mask L1 0.0532, length p95 0.0777, width p95 0.000263, opacity p95 0.852
iter 120: composite PSNR 25.587, mask L1 0.0324, length p95 0.0845, width p95 0.000324, opacity p95 0.888

root lifecycle:
  20: 20000 -> 20512, insert 512
  40: 20512 -> 21024, insert 512
  60: 21024 -> 21384, insert 360
  80: 21384 -> 21672, insert 288
 100: 21672 -> 21902, insert 230
```

This confirms two separate effects:

- the coverage side improves: mask error drops steadily and direct insertion naturally slows down as the valid spaced residual targets become fewer;
- the long-streak side is still present: length, width, and opacity keep increasing, so better densification alone does not remove long opaque strokes.

Two negative controls were also run to avoid guessing:

```text
direct spacing 0.004, normal priors:
  iter 120 composite PSNR: 25.587
  mask L1: 0.0324
  p95 length/width/opacity: 0.0845 / 0.000324 / 0.888

stronger smooth + shape priors:
  iter 120 composite PSNR: 25.471
  mask L1: 0.0325
  p95 length/width/opacity: 0.0827 / 0.000324 / 0.888

hard early stroke-capacity freeze to iter 80:
  iter 120 composite PSNR: 23.157
  mask L1: 0.0822
  p95 length/width/opacity: 0.0680 / 0.000206 / 0.802

soft early stroke-capacity scale 0.25 to iter 80:
  iter 120 composite PSNR: 25.564
  mask L1: 0.0336
  p95 length/width/opacity: 0.0844 / 0.000323 / 0.886
```

These controls rule out two easy explanations:

- simply increasing smooth/shape prior is not enough; it barely changes width/opacity and slightly hurts PSNR;
- simply freezing or uniformly scaling length/width/opacity gradients is not the right fix. A hard freeze suppresses the artifact capacity but loses coverage/fitting; a soft scale almost exactly returns to the normal run.

The remaining problem is more specific: roots are still allowed to use long/wide/opaque one-color strokes to explain texture gaps. The repair should not be a blanket capacity freeze. It needs either better local color/attribute support for each strand, stronger local grooming consistency tied to surface neighborhoods, or a lifecycle rule that treats high-residual high-contribution long strokes as evidence for additional local roots rather than as a reason to keep expanding the same strand.

A logging-only stroke-drag diagnostic was then added behind `--stroke-drag-diagnostics`. It does not change training. It only measures whether high-capacity roots are receiving disproportionate contribution/residual evidence. The same 120-iteration direct+spacing run produced essentially the same image metrics, so the diagnostic is non-invasive:

```text
direct spacing 0.004 + stroke-drag diagnostics:
  iter 120 composite PSNR: 25.615
  mask L1: 0.0325
  roots: 21929
  p95 length/width/opacity: 0.0843 / 0.000324 / 0.888
```

Stroke-drag diagnostic summary:

```text
iter 20:
  high-capacity roots: 3.86% of roots, 7.18% contribution, 10.70% residual
  width/opacity vs contribution-per-sample correlation: 0.660 / 0.589

iter 40:
  high-capacity roots: 4.39% of roots, 8.22% contribution, 13.16% residual
  width/opacity vs contribution-per-sample correlation: 0.749 / 0.662

iter 80:
  high-capacity roots: 3.57% of roots, 6.89% contribution, 4.89% residual
  width/opacity vs contribution-per-sample correlation: 0.785 / 0.728

iter 100:
  high-capacity roots: 3.12% of roots, 6.32% contribution, 4.58% residual
  width/opacity vs contribution-per-sample correlation: 0.787 / 0.741
```

This refines the diagnosis:

- Early in training, high-capacity roots are also over-represented in residual, so they are visibly involved in hard regions.
- Later in training, residual becomes less concentrated on those roots, but contribution remains strongly correlated with width and opacity. This means the long-strand artifact is not simply "the highest residual roots get long." It is more structural: wide/opaque strands are rewarded as high-contribution explanations, even after coverage improves.
- Length is involved, but width and opacity are the stronger measured correlates. So a length-only fix would likely miss the main stroke-painting path.
- A blanket capacity freeze is too blunt. The better repair target is local support: prevent a single root color and one wide opaque strand from explaining a large texture region, while still allowing normal hair length where it is geometrically correct.

The next repair should therefore be tested along one of these cleaner directions:

1. **Local color/attribute support.** Keep the current single guide/root geometry, but let color/opacity vary more locally along the strand or through child/local support, so a long strand is not forced to paint one color across stripe boundaries.
2. **Stroke-footprint regularization.** Penalize the combination of high width, high opacity, high contribution-per-sample, and high local RGB residual. This targets the measured failure mode more directly than a global width/opacity prior.
3. **Densify-from-overpaint.** Treat high-contribution/high-capacity roots in high residual neighborhoods as evidence that the region needs more local roots, not as evidence that the current root should keep growing stronger.
4. **Surface-neighborhood grooming consistency.** Strengthen local agreement for geometry and appearance parameters on the mesh graph, but only where it does not erase real stripe/color variation. This should be separate from the failed blanket shape-prior increase.

## Current Root Cause Hypothesis

The current failure is a competition between long-strand fitting and densification:

1. RGB/mask loss can be reduced by increasing strand length, width, opacity, and color strength.
2. This lets existing roots cover missing or hard regions before densification gets a clean signal.
3. Densification then selects high-gradient/high-contribution roots instead of missing local detail.
4. Pruning removes many low-contribution roots once it starts, further reducing local coverage.
5. The remaining roots compensate by becoming longer/wider/opaque, which creates streaks and color dragging.

This is not just an orientation-map issue. Orientation noise can worsen it, but the logs show capacity allocation and lifecycle behavior are also involved.

The first attempted residual attribution also shows the repair has to be coverage-aware. A simple root-pixel residual is still biased toward high-visibility strands, so it does not break the long-strand feedback loop by itself.

There is also a segment-budget feedback loop:

- `strand_segment_budgets` increases Gaussian count with decoded length and curvature.
- `RootStatsWindow` accumulates visibility/contribution per visible Gaussian sample.
- A longer root therefore receives more visible samples and more contribution budget.
- `select_prune_mask` ranks/prunes by absolute contribution/visibility, so shorter or less visible roots are easier to delete.
- Once pruning starts, root count falls, and surviving roots have an incentive to become longer/wider/opaque to cover the same pixels.

This does not mean adaptive segments are wrong; the renderer needs enough segments for long/curved hair. The issue is that lifecycle evidence currently treats extra samples from a long strand as stronger structural evidence. That makes long-strand fitting compete with densification.

Current diagnosis split:

- Initial trigger: densification evidence is still concentrated on already highly visible/contributing roots, even with root-pixel or pooled residual.
- Placement trigger: the original split placement is not residual-directed. It only samples topology neighborhoods around selected parents and chooses locally empty candidates, so it cannot guarantee that children fill image-space holes. The target-placement probe partially fixes this by moving child candidates toward unprojected residual targets, but it does not fix biased parent selection.
- Direct target trigger: bypassing parent-root selection and inserting new roots from residual-pixel-to-mesh-face targets improves the short probe more than the parent-based variants. This suggests the proper structural repair is direct surface candidate generation with root-neighborhood inheritance, not additional scalar weights on existing parents.
- Later amplifier: pruning and absolute sample-count lifecycle evidence favor roots with more visible samples; once some roots grow longer/wider, they become structurally harder to remove and can keep painting larger regions.
- Appearance amplifier: length, root width, and opacity are still available as faster ways to reduce image loss than adding new local roots.

## Fix Directions To Validate

The next changes should be tested one at a time on a diagnostic branch.

1. Replace naive root-pixel residual with coverage-aware local residual.
   - Current densification is based on root/gaussian gradients and visibility/contribution.
   - Root-pixel residual is measurable, but it still selects high-visibility roots.
   - The first pooled version is still too correlated with visible/contribution.
   - The next version should separate under-coverage evidence from already-painted residual: high `relu(mask - alpha)` should identify holes, while high contribution plus high residual should be treated as a suspicious long-painting signal, not automatically as a densify target.
   - The evidence should favor roots near under-covered regions, not roots already painting strongly.
   - Pixel-to-root evidence was tested and is not enough by itself. It still selects high-visible/high-contribution parents when the placement step remains parent-centric.
   - Target-directed child placement was tested and works as a partial repair: it moves inserted children closer to residual surface targets and slightly improves the short probe.
   - The next valid repair is fully hole-directed densification: keep pixel evidence as a set of image-space targets, assign targets to nearby visible surface roots/faces, and let targets nominate parent/candidate regions directly. This changes both parent selection and child placement; changing only the scalar parent score is insufficient.

2. Delay or soften true pruning.
   - Parent replacement during split is expected.
   - Additional pruning should not aggressively reduce root count before the new roots have stabilized.
   - Prune should operate after a full evidence window and should be separated from early densification tests.
   - Prune should not rank purely by absolute Gaussian-sample contribution; it should account for per-root segment/sample budget or use a stricter "no evidence for a full pass" criterion early.
   - Sample-normalized lifecycle mode is now available as a diagnostic switch, but it should be validated on a later iteration where segment counts actually differ.

3. Stage appearance capacity.
   - Early training should prevent length/width/opacity from becoming the primary way to close holes.
   - Densification should get the first chance to explain missing local detail.
   - After root coverage improves, loosen local appearance parameters.
   - This is likely required because residual-only routing did not break the feedback loop.

4. Penalize abnormal long-strand painting.
   - Not all long hair is wrong.
   - The target is long, opaque, high-color-contrast strands that cross local texture/detail boundaries.
   - A simple first version can strengthen length/width/opacity priors during early densification.
   - A better version should look at image-space footprint: long/wide roots with high residual and high contribution are suspicious because they are likely dragging color rather than adding local coverage.

5. Reconsider multi-level roots only after the single-level diagnosis is verified.
   - Multi-level roots are not rejected.
   - They require separate guide/render root lifecycle logic and multi-level densification.
   - They should be introduced only if evidence shows the current coupling between guide attributes and render coverage cannot be fixed cleanly.

## 2026-06-26 Child-Local Appearance Probe

The next diagnostic tested whether one part of the long-streak artifact comes
from appearance being tied too coarsely to each guide root.

Code evidence:

- `build_strands(...)` produces one `root_color -> tip_color` ramp per guide
  root.
- `expand_child_strands(...)` then expands child strands with
  `child_colors = colors[:, None].expand(...)` and
  `child_opacities = opacities[:, None].expand(...)`.
- Therefore all child strands spawned by the same guide root share exactly the
  same color and opacity profile. If that root becomes long, wide, or opaque, it
  can paint one color across several stripe boundaries.

A default-off diagnostic was added to `tools/train_white_tiger_stage1.py`:

- `--local-child-color-support`
- `--local-child-opacity-support`

These add bounded per-child color / opacity offsets after child expansion. The
default path is unchanged; the new parameters are absent unless the flags are
explicitly enabled. The parameters are also carried through split/prune via the
same root-attribute interpolation path as the groom parameters.

Same 120-iteration single-view probe, same direct insertion and spacing
configuration:

```text
baseline:
  composite PSNR: 25.615
  raw PSNR:       25.447
  mask L1:        0.0325

local child color:
  composite PSNR: 26.674
  raw PSNR:       26.443
  mask L1:        0.0343

local child color + opacity:
  composite PSNR: 26.758
  raw PSNR:       26.514
  mask L1:        0.0310
```

Comparison image:

```text
D:\petsgaussianhair\_downloads\white_tiger_childlocal_comparison_120iter.png
```

Conclusion:

- Local child appearance support is a real improvement, not just metric noise.
  It gives the renderer a way to separate local stripe color without forcing one
  guide-root color to cover an entire child bundle.
- This confirms appearance coupling is one real cause of the long-streak /
  dragged-color artifact.
- It is not sufficient by itself. Width and opacity still grow with contribution
  and remain strongly correlated with contribution-per-sample. The geometry /
  opacity capacity loop is still present.

Updated repair direction:

1. Keep child-local appearance as a candidate formal module, but only with
   bounded support and a later smoothness/regularization check. It should not
   become an unconstrained texture-like escape hatch.
2. The next root cause to address is still geometry/opacity overpaint: roots can
   become wider/opaque and explain missing coverage before densification inserts
   enough local roots.
3. A clean next diagnostic should target footprint/capacity directly, preferably
   by limiting early width/opacity growth or routing high-capacity/high-residual
   roots into densification evidence rather than letting them keep increasing
   opacity/width.

## 2026-06-26 Targeted Overpaint Capacity Probe

The next diagnostic tested whether the remaining artifact can be fixed by
penalizing high-capacity roots only where residual evidence remains high.

A default-off diagnostic loss was added:

```text
--overpaint-capacity-weight
--overpaint-residual-threshold
--overpaint-length-target
--overpaint-width-target
--overpaint-opacity-target
```

The default training path is unchanged. If the loss is enabled without residual
evidence, training raises instead of silently falling back.

Same 120-iteration single-view probe, with local child color and opacity support
enabled:

```text
local child color + opacity:
  composite PSNR: 26.758
  raw PSNR:       26.514
  mask L1:        0.0310
  p95 length:     0.083129
  p95 width:      0.00032149
  p95 opacity:    0.886510

overpaint 0.10, residual threshold 0.08:
  composite PSNR: 26.725
  raw PSNR:       26.484
  mask L1:        0.0333
  p95 length:     0.082038
  p95 width:      0.00031843
  p95 opacity:    0.885089

overpaint 0.20, residual threshold 0.02:
  composite PSNR: 26.712
  raw PSNR:       26.466
  mask L1:        0.0338
  p95 length:     0.082001
  p95 width:      0.00031742
  p95 opacity:    0.884985
```

Stroke-drag evidence at the final lifecycle event:

```text
baseline:
  residual drag candidates: 81
  residual share in drag candidates: 0.035807
  width/opacity correlation with contribution per sample: 0.787 / 0.741

local child color + opacity:
  residual drag candidates: 75
  residual share in drag candidates: 0.025356
  width/opacity correlation with contribution per sample: 0.802 / 0.773

overpaint 0.10:
  residual drag candidates: 3
  residual share in drag candidates: 0.000184
  width/opacity correlation with contribution per sample: 0.785 / 0.758

overpaint 0.20:
  residual drag candidates: 0
  residual share in drag candidates: 0.0
  width/opacity correlation with contribution per sample: 0.780 / 0.756
```

Comparison crop:

```text
D:\petsgaussianhair\_downloads\white_tiger_overpaint_diagnosis_crop_120iter.png
```

Conclusion:

- The residual-conditioned overpaint loss is a useful diagnostic, but not a
  complete repair.
- It successfully removes the specific combination "high residual + high
  capacity + high contribution". However, the visible artifact remains because
  high-capacity roots become high-contribution, low-residual painters after they
  already fill the image. At that point a residual-conditioned penalty is too
  late and too local.
- Therefore the true remaining issue is a training-order / capacity competition:
  width and opacity can explain holes before densification adds enough local
  roots. Once they do, the image residual drops and a residual-only overpaint
  penalty no longer sees the artifact.
- The next repair should not be another residual-weight tweak. It should give
  densification the first chance to explain missing local coverage, either by
  staging width/opacity growth during early densification or by routing alpha
  under-coverage directly into root insertion before width/opacity become the
  easiest solution.

Next valid repair candidates:

1. Early capacity staging: temporarily cap or strongly regularize width/opacity
   while densification is active, then loosen after root coverage stabilizes.
2. Under-coverage-first densification: use `relu(mask - alpha)` as primary
   insertion evidence, so missing coverage creates new roots instead of asking
   existing roots to become wider or more opaque.
3. Image-footprint regularization: penalize high-opacity, high-footprint roots
   even when residual is already low, but only during the early coverage-building
   stage to avoid suppressing valid final fur.
