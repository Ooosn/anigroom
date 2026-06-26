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
```

This confirms the baseline is not changed when the switch is off. It also shows that naive root-projected residual is not enough: it raises residual evidence, but the selected parents are still the highest-visible/highest-contribution roots. In other words, point-sampling residual at the root projection still follows the already-rendered streaks and strong texture edges instead of reliably identifying under-covered holes.

The first coverage-pooled version (`alpha deficit + pooled RGB residual`) is also not enough. It reduces the residual ratio gap compared with root-pixel residual, but selected parents are still `1.82x` the global contribution and `1.80x` the global visible sample count. The evidence is still too tied to the current rendered hair layer.

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

## Fix Directions To Validate

The next changes should be tested one at a time on a diagnostic branch.

1. Replace naive root-pixel residual with coverage-aware local residual.
   - Current densification is based on root/gaussian gradients and visibility/contribution.
   - Root-pixel residual is measurable, but it still selects high-visibility roots.
   - The first pooled version is still too correlated with visible/contribution.
   - The next version should separate under-coverage evidence from already-painted residual: high `relu(mask - alpha)` should identify holes, while high contribution plus high residual should be treated as a suspicious long-painting signal, not automatically as a densify target.
   - The evidence should favor roots near under-covered regions, not roots already painting strongly.

2. Delay or soften true pruning.
   - Parent replacement during split is expected.
   - Additional pruning should not aggressively reduce root count before the new roots have stabilized.
   - Prune should operate after a full evidence window and should be separated from early densification tests.
   - Prune should not rank purely by absolute Gaussian-sample contribution; it should account for per-root segment/sample budget or use a stricter "no evidence for a full pass" criterion early.

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
