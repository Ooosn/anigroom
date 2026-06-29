# White Tiger Stage 1 Long-Streak Diagnosis

Date: 2026-06-26

This note records the current diagnosis and accepted formal settings for the
view09 white-tiger Stage 1 baseline.

## Formal Baseline Update: 2026-06-28

The formal baseline is now the stricter 0.075 length-split run with a small RGB
term in densification evidence, higher target-direct root insertion capacity,
higher overlong-split throughput, 3-child overlong replacement,
screen-footprint split evidence, slightly shorter split children, stronger
child-local color support, larger independent screen-footprint split budgets
for dark and non-dark high-footprint roots, and decoupled primary-overlong and
screen-footprint parent queues. It is not the old 3000-iteration server pull
and not the guide-root, dark-stroke, overlap, compact-child, screen-stroke-loss,
projected-child-color, child-width0.90, all-luma screen-footprint, lineage,
child-color0.75, or overcapacity variants.

Run:

`D:\petsgaussianhair\_downloads\white_tiger_view09_stage1_overlong3child_screenbudget256_192_view09_duntil1000_overlong1100_capacity1300_1300_20260628`

Final checkpoint:

`D:\petsgaussianhair\_downloads\white_tiger_view09_stage1_overlong3child_screenbudget256_192_view09_duntil1000_overlong1100_capacity1300_1300_20260628\checkpoint_001300.pt`

Final render:

`D:\petsgaussianhair\_downloads\white_tiger_view09_stage1_overlong3child_screenbudget256_192_view09_duntil1000_overlong1100_capacity1300_1300_20260628\iter_001300\view_09_eval_pred.png`

Metrics:

- raw PSNR: 35.0037
- raw SSIM: 0.966042
- composite PSNR: 37.8497
- composite SSIM: 0.972438
- root count: 41,942
- Gaussian count: 3,102,581
- p95 length: 0.073819
- max GPU memory: 6885.0 MB

The view09 calibration baseline uses this schedule:

- ordinary target-direct densification stops at iteration 1000.
- overlong/screen-footprint replacement continues to iteration 1100.
- early capacity regularization stays active through iteration 1300, so the
  200-step recovery phase after the final replacement does not immediately
  regrow wide/high-opacity projected brush strokes.

The core algorithmic settings are:

- 72 samples per strand, 12-44 segments, 4 child strands.
- smooth weight 0.008 and strand-shape smooth weight 0.0.
- pixel-to-root densification evidence.
- densification evidence uses alpha residual plus a small RGB residual term
  (`DENSIFY_RESIDUAL_RGB_WEIGHT=0.35`) so stripe-color errors can request local
  roots. This is a marginal metric improvement, not a full streak solution.
- target-direct parent selection.
- higher target-direct insertion capacity
  (`DENSIFY_PIXEL_EVIDENCE_TOPK=16384`, `MAX_SPLITS_PER_EVENT=384`), which
  lets local RGB/alpha residuals request enough new roots instead of forcing
  existing strands to grow or widen.
- local child color and opacity support enabled, with child color scale 0.50.
  This replaces the previous 0.20 scale because it improves PSNR and reduces
  dark-streak diagnostics without adding another global capacity loss.
- strict overlong length split enabled at length threshold 0.075, with 384
  overlong parents per densification event, 3 children per overlong parent, and
  child length scale 0.50. The extra child reduces the raw overlength and
  dark-overlength tails while preserving enough capacity for stripe detail.
- screen-footprint split evidence enabled with a 40px projected-root diagonal
  threshold and luma threshold 0.38. This is split evidence only, not a loss:
  it lets remaining visible screen-sized strokes request local child roots
  instead of globally suppressing dark-stripe capacity.
- screen-footprint candidates are not mixed into the primary overlong top-k
  queue. They are handled by their dark and neutral screen-footprint budgets
  below. This prevents a lower screen threshold from stealing parent slots from
  true length/width/opacity overlong roots.
- screen-footprint split now has an independent 256-parent budget per
  densification event. This fixes the previous failure mode where dark,
  screen-sized stroke roots were valid candidates but were crowded out by the
  ordinary overlong top-k queue. The budget is still local split evidence, not a
  capacity loss or a global suppression rule.
- non-dark screen-footprint candidates now have a separate neutral 192-parent
  budget per densification event. This accepts the useful part of the all-luma
  diagnostic without removing the dark luma gate: white/gray screen-sized roots
  can request local children, while dark stripe roots keep their own budget.
- standalone low-contribution prune disabled for this baseline. Parent
  replacement inside split/densification is still active; the 1300-step run
  reached 41,942 roots.

This baseline improves over the previous accepted baseline in PSNR and most
length-tail diagnostics: composite PSNR rises from 37.4128 to 37.8497 over the
previous `overlong3child` baseline, and from 36.8041 over the older
capacity-through-recovery baseline. `overlength_gt_0080` drops from 353 to 146,
`dark_overlength` drops from 20 to 11, `screen_diag_p95` mean projected diagonal
drops from 51.58px to 47.35px, `visible_screen_stroke_p95` mean projected
diagonal drops from 47.34px to 43.02px, and `axis_brush_p95` mean projected
diagonal drops from 42.72px to 38.36px. It still does not completely solve the
visual problem. `dark_screen_stroke` count rises from 50 to 96 even though its
mean projected diagonal is smaller at 41.49px, so the remaining failure is not
just raw length. The next repair should focus on separating local density,
strand footprint, and color/opacity capacity rather than simply adding more
split children or another global clamp. For full 30000-step formal training, do
not blindly copy the 1300-step iteration numbers; map this schedule only after a
separate long-run calibration.

Enhanced diagnostics on the accepted baseline show that these remaining
screen-sized strokes are not caused by single-Gaussian major-axis explosion:
in the previous `overlong3child` baseline, only 4.7% of
`visible_screen_stroke_p95` roots had `length > 0.08`. Those roots were instead
distinguished by screen footprint, width, opacity, child radius, target-luma
span, and moderate local sparsity. `visible_screen_stroke_p95` had mean
projected bbox diagonal 47.34px, but its mean single-Gaussian major axis was
only 3.98px. The failure is therefore a whole root/child-strand projected
footprint and color/opacity dragging problem, not a CUDA splat-size bug or a
simple per-Gaussian scale outlier.

Diagnostics on the current screenbudget256/192 baseline refine this further.
The remaining `dark_screen_stroke` roots have mean length 0.0637, so they are
not primarily true overlength roots. They are characterized by high opacity
(0.888), elevated width (0.000399), elevated child radius (0.00391), and mean
screen bbox diagonal 41.49px. The `visible_screen_stroke_p95` category shows the
same pattern at larger scale: mean length 0.0671, width 0.000418, opacity 0.896,
and child radius 0.00388. In contrast, the raw `overlength_gt_0080` population
is now only 146 roots. The remaining visual artifact is therefore a projected
footprint and opacity/color-density competition, not a length-threshold problem.
The next structural fix should target how local render roots distribute
coverage and color, not simply lower length limits.

A follow-up bright/gray screen-shape capacity loss was rejected. It reduced
some global shape attributes but barely changed the actual visible footprint:
`visible_screen_stroke_p95` mean diagonal moved only 48.59px -> 48.28px, while
composite PSNR dropped from 36.8041 to 36.7105 and `dark_screen_stroke` rose
from 36 to 42. A separate screen-footprint lineage capacity test was also
rejected: it shortened the tracked split descendants, but did not reduce the
global visible-stroke categories and lowered PSNR. The remaining repair target
is therefore not another global capacity penalty and not a simple lineage clamp;
it is the competition between local density, strand footprint, and stripe color
assignment. The next structural direction is guide/render-root separation or a
true split policy for overlong projected strokes.

## 2026-06-27 Diagnostic Experiments After Baseline

These runs are explicitly diagnostic and must not replace the baseline.

| Run | Composite PSNR | Result |
| --- | ---: | --- |
| `white_tiger_view09_stage1_darkstroke_1200_20260627` | 34.3829 | Suppressed dark high-capacity roots, but the model compensated with brighter/gray overlength strokes. |
| `white_tiger_view09_stage1_overlap115_1200_20260627` | 34.8010 | Lower overlap reduced segment coverage but increased width/opacity compensation. |
| `white_tiger_view09_stage1_overlong_shape_reset_1200_20260627` | 34.8598 | Slightly cleaner, still below baseline; did not reduce the real streak categories enough. |
| `white_tiger_view09_stage1_guide1024_1200_20260627` | 34.7206 | Guide length/bend alone did not solve the visual streaks. |
| `white_tiger_view09_stage1_guide1024_flow_1200_20260627` | 34.0140 | Guide length/bend/flow made length low-frequency but increased overlength roots to 3,954. |
| `white_tiger_view09_stage1_guide4096_lenbend_rgb035_1200_20260627` | 34.9067 | Denser guide length/bend recovered more PSNR than 1024-guide, but still lost to baseline and increased `overlength_gt_0080` to 4,270. |
| `white_tiger_view09_stage1_color_contrast_capacity_005_1200_20260627` | 34.9103 | Reduced some overlength statistics, but softened reconstruction and stayed below baseline. |
| `white_tiger_view09_stage1_overlong_capacity_split_075_1200_20260627` | 34.9202 | Split children with weaker capacity reduced the length tail, but width/opacity compensation remained. |
| `white_tiger_view09_stage1_early_capacity012_width032_opacity088_1200_20260627` | 34.7718 | Stronger early capacity prior cut overlength roots to 899, but suppressed useful reconstruction capacity. |
| `white_tiger_view09_stage1_width_triggered_split_1200_20260627` | 34.9761 | Width-triggered split lowered p95 width but increased `overlength_gt_0080` to 2,422 and still lost PSNR. |
| `white_tiger_view09_stage1_rgb_evidence035_1200_20260627` | 35.0859 | Previous baseline. It slightly improved metric and dark high-capacity count, but visual long streaks remained. |
| `white_tiger_view09_stage1_compact_overlong_children_rgb035_1200_20260627` | 35.0197 | Overlong split children were made shorter/tighter/lower-capacity, but `overlength_gt_0080` still rose to 2,309 and `dark_high_capacity` rose to 122. |
| `white_tiger_view09_stage1_more_target_roots_rgb035_1200_20260627` | 35.7980 | Previous baseline. More target-direct root insertion improved metric and reduced `overlength_gt_0080` to 2,141, but visible long streaks remained. |
| `white_tiger_view09_stage1_screenstroke015_rgb035_1200_20260627` | 34.0758 | Screen-stroke capacity loss reduced the diagnostic count but suppressed useful dark-stripe capacity; rejected. |
| `white_tiger_view09_stage1_overlong384_view09_rgb035_1200_20260627` | 36.1445 | Previous accepted baseline. Increasing strict-overlong parent throughput from 160 to 384 reduced `overlength_gt_0080` to 1,072 and improved PSNR. |
| `white_tiger_view09_stage1_screenfoot50_overlong384_rgb035_1200_20260627` | 36.1958 | Previous baseline. Screen-footprint split evidence gives a small metric improvement and reduces `dark_screen_stroke` from 351 to 335, but visible streaks remain. |
| `white_tiger_view09_stage1_projected_child_color_screenfoot50_view09_1200_20260627` | 33.8737 | Rejected. Projected child color lowered overlength count but increased `dark_screen_stroke` and hurt PSNR. |
| `white_tiger_view09_stage1_screen_stroke_w002_screenfoot50_1200_20260627` | 33.8177 | Rejected. Screen-stroke capacity loss reduced some per-root lengths but increased `dark_screen_stroke` to 546 and suppressed useful stripe capacity. |
| `white_tiger_view09_stage1_screenfoot45_duntil1000_1200_20260627` | 36.1450 | Candidate. Lower screen-footprint threshold reduced `dark_screen_stroke` to 281, but slightly reduced composite PSNR and increased the overlength tail. |
| `white_tiger_view09_stage1_screenfoot48_duntil1000_1200_20260627` | 36.1233 | Rejected. Intermediate threshold did not improve over screenfoot45 or screenfoot50. |
| `white_tiger_view09_stage1_screenfoot45_len050_duntil1000_1200_20260627` | 36.1950 | Previous baseline. Keeps screenfoot50 PSNR while reducing `dark_screen_stroke` to 287 and `overlength_gt_0080` to 1,286. |
| `white_tiger_view09_stage1_child_width090_view09_duntil1000_1200_20260627` | 36.0621 | Rejected. Reducing split-child width to 0.90 barely changed `dark_screen_stroke` (285) and increased `dark_high_capacity` to 137, so the remaining streaks are not caused by child width inheritance alone. |
| `white_tiger_view09_stage1_child_color_scale040_view09_duntil1000_1200_20260627` | 36.3051 | Previous baseline. Stronger child-local color freedom reduced black-stripe drag: `dark_screen_stroke` 244, `dark_high_capacity` 92, and `overlength_gt_0080` 1,185. |
| `white_tiger_view09_stage1_child_color_scale060_view09_duntil1000_1200_20260627` | 36.0437 | Rejected. It reduced `dark_screen_stroke` to 187, but the stronger child-local color scale hurt PSNR and slightly increased `overlength_gt_0080` to 1,203. |
| `white_tiger_view09_stage1_child_color_scale050_view09_duntil1000_1200_20260627` | 36.3260 | Previous baseline. It kept the PSNR gain while further reducing black-stripe drag: `dark_screen_stroke` 212, `dark_high_capacity` 89, and `overlength_gt_0080` 1,221. |
| `white_tiger_view09_stage1_guide4096_child_color050_view09_duntil1000_1200_20260627` | 35.9097 | Rejected. Adding 4096 guide roots on top of the current child-color baseline lowered dark-stroke counts (`dark_screen_stroke` 125, `dark_high_capacity` 12), but it converted the artifact into broad coherent overlength regions: `overlength_gt_0080` rose to 5,528 and PSNR dropped. |
| `white_tiger_view09_stage1_overlong_child_spread075_clump018_color050_view09_duntil1000_1200_20260627` | 36.2459 | Rejected. Making overlong split children more compact did not address the real artifact: `dark_screen_stroke` rose to 223, `dark_high_capacity` rose to 94, and `overlength_gt_0080` rose to 1,328. |
| `white_tiger_view09_stage1_child_opacity006_color050_view09_duntil1000_1200_20260628` | 36.1925 | Rejected. Lowering local child opacity freedom barely changed `dark_screen_stroke` (209 vs 212 baseline), increased `dark_high_capacity` to 123, and reduced PSNR. |
| `white_tiger_view09_stage1_screen_score8_child_color050_view09_duntil1000_1200_20260628` | 36.2582 | Rejected. Increasing the shared screen-footprint score weight did not solve candidate starvation: `dark_screen_stroke` stayed at 211 and PSNR dropped. |
| `white_tiger_view09_stage1_screen_extra128_child_color050_view09_duntil1000_1200_20260628` | 36.4850 | Previous baseline. Giving screen-footprint candidates an independent 128-parent budget reduced `dark_screen_stroke` to 84, `dark_high_capacity` to 84, and `overlength_gt_0080` to 972 while improving PSNR. |
| `white_tiger_view09_stage1_strand_shape006_screen_extra128_view09_duntil1000_1200_20260628` | 36.5034 | Rejected as baseline despite the tiny PSNR gain. `overlength_gt_0080` increased to 1,037, `dark_screen_stroke` stayed essentially unchanged at 85, `dark_high_capacity` increased to 86, `overlength_high_frizz` increased to 287, and `overlength_high_child_radius` increased to 258. The existing strand-shape consistency term at 0.006 does not solve the remaining streaks. |
| `white_tiger_view09_stage1_screen_extra128_all_luma_view09_duntil1000_1200_20260628` | 36.3680 | Rejected. Removing the dark luma gate helped non-dark overlength (`overlength_gt_0080` 907), but it exploded `dark_screen_stroke` to 274 and reduced PSNR. This proved white/gray screen-footprint roots need extra split budget, but not by stealing the dark budget. |
| `white_tiger_view09_stage1_screen_extra128_neutral64_view09_duntil1000_1200_20260628` | 36.5514 | Previous baseline. A separate neutral 64-parent screen-footprint budget reduced `overlength_gt_0080` to 819, `dark_high_capacity` to 78, `dark_overlength` to 95, `overlength_high_curl` to 146, `overlength_high_frizz` to 199, and `overlength_high_child_radius` to 184. It regressed `dark_screen_stroke` to 112, which exposed the remaining screen-footprint issue. |
| `white_tiger_view09_stage1_screen_extra128_neutral64_screen40_view09_duntil1000_1200_20260628` | 36.6128 | Rejected as baseline despite higher PSNR. Lowering the screen-footprint threshold from 45px to 40px reduced `dark_screen_stroke` from 112 to 60, but shifted the failure into ordinary overlength: `overlength_gt_0080` rose to 914, `overlength_high_curl` to 179, `overlength_high_frizz` to 224, and `overlength_high_child_radius` to 250. This is useful evidence, but not a clean repair. |
| `white_tiger_view09_stage1_screen40_childcompact_view09_duntil1000_1200_20260628` | 36.4760 | Rejected. Keeping screen40 while shrinking split-child width, opacity, and spread further reduced `dark_screen_stroke` to 47, but PSNR dropped and ordinary overlength worsened further: `overlength_gt_0080` 1,015, `dark_overlength` 112, `overlength_high_curl` 224, `overlength_high_frizz` 242, and `overlength_high_child_radius` 321. This shows child compactness alone turns into under-coverage plus residual overlength, not a solution. |
| `white_tiger_view09_stage1_screen40_decoupled_view09_duntil1000_1200_20260628` | 36.6066 | Previous baseline. Screen-footprint candidates are handled only by their dark/neutral budgets and no longer crowd the primary overlong top-k queue. This keeps the screen40 dark-stroke gain while fixing the ordinary-overlength regression: `overlength_gt_0080` 607, `dark_screen_stroke` 60, `dark_high_capacity` 49, `dark_overlength` 49, `overlength_high_curl` 105, `overlength_high_frizz` 117, and `overlength_high_child_radius` 117. |
| `white_tiger_view09_stage1_screen40_decoupled_spread075_view09_duntil1000_1200fix_20260628` | 33.5918 | Rejected. Compressing only overlong split-child shape spread (`child_radius`, `curl_radius`, and `frizz`) kills the numeric overlength tail (`overlength_gt_0080` 139, `dark_screen_stroke` 0), but it removes too much reconstruction capacity and drops composite PSNR by 3.01 dB. This is under-fitting, not a valid streak repair. |
| `white_tiger_view09_stage1_screen40_decoupled_residualtarget1_fix_view09_duntil1000_1200_20260628` | 33.5526 | Rejected. Adding strong residual-target scoring to overlong directional split kills the visible streak statistics (`overlength_gt_0080` 110, `dark_screen_stroke` 0), but it removes useful stripe/fur detail and drops composite PSNR by 3.05 dB. This is also under-fitting, not a clean repair. |
| `white_tiger_view09_stage1_screen40_decoupled_residualtarget025_view09_duntil1000_1200_20260628` | 33.6244 | Rejected. A weaker residual-target score still behaves like the strong version: `overlength_gt_0080` 110 and `dark_screen_stroke` 1, but composite PSNR remains 2.98 dB below the accepted baseline. Residual-target placement is useful as a diagnostic, not as a global split score. |
| `white_tiger_view09_stage1_screen40_decoupled_screenadd_view09_duntil1000_1200_20260628` | 28.5623 | Rejected. Keeping primary overlong parent replacement but making screen-footprint extra roots add-only is much worse than the accepted baseline: `overlength_gt_0080` rises to 3,584, `dark_screen_stroke` rises to 294, and composite PSNR drops by 8.04 dB. Screen-footprint extras must not be simple add-only coverage; without replacement they create excessive low-quality local roots and do not repair the streak mechanism. |
| `white_tiger_view09_stage1_screen40_decoupled_view09_duntil1100_view09_1200_20260628` | 36.5028 | Candidate diagnostic, not a baseline replacement. This is the same view09 calibration setup as the accepted baseline except densification/overlong split continues to iteration 1100. It reduces remaining long-streak diagnostics (`overlength_gt_0080` 607 -> 335, `dark_screen_stroke` 60 -> 24, `dark_overlength` 49 -> 27, `overlength_high_curl` 105 -> 59, `overlength_high_frizz` 117 -> 55, `overlength_high_child_radius` 117 -> 61), but drops composite PSNR by 0.10 dB and increases roots from 32,096 to 33,440. This supports the hypothesis that part of the residual artifact is late regrowth after the 1000-step split window, but the repair is not yet free. |
| `white_tiger_view09_stage1_screen40_decoupled_view09_duntil1000_overlong1100_1200_20260628` | 36.3648 | Candidate diagnostic, not a baseline replacement. Ordinary target-direct densification still stops at iteration 1000, while only overlong/screen-footprint replacement continues to 1100. This reduces long-streak diagnostics without adding ordinary target roots (`overlength_gt_0080` 607 -> 394, `dark_screen_stroke` 60 -> 27, `dark_overlength` 49 -> 37, `overlength_high_curl` 105 -> 68, `overlength_high_frizz` 117 -> 64, `overlength_high_child_radius` 117 -> 76), but drops composite PSNR by 0.24 dB. The likely confound is recovery time: this run performs its last split at 1100 and only trains 100 iterations afterward, while the accepted 1200-step baseline has 200 iterations after the final 1000-step split. |
| `white_tiger_view09_stage1_screen40_decoupled_view09_duntil1000_overlong1100_recover200_1300_20260628` | 36.8187 | Diagnostic. Keeping the overlong-only 1100 replacement schedule and giving it 200 recovery iterations recovers PSNR, but ordinary overlength regrows: `overlength_gt_0080` rises to 867 while `dark_screen_stroke` stays low at 25. This proves the 1200-step overlong-only drop was partly recovery-time related, but it also exposes post-split regrowth once capacity is released. |
| `white_tiger_view09_stage1_screen40_decoupled_view09_duntil1000_overlong1100_capacity1300_1300_20260628` | 36.8041 | Previous accepted baseline. Same recovery as above, but early capacity stays active through iteration 1300. It keeps the PSNR recovery while preventing most post-split regrowth: `overlength_gt_0080` 441, `dark_screen_stroke` 36, `dark_high_capacity` 43, and `dark_overlength` 36. Remaining artifacts are dominated by visible screen-footprint categories, not by the raw length tail. |
| `white_tiger_view09_stage1_neutral_screen005_view09_duntil1000_overlong1100_capacity1300_1300_20260628` | 36.7380 | Rejected. A small neutral/bright screen-footprint capacity loss reduces the pure overlength tail (`overlength_gt_0080` 441 -> 91) and some dark counts (`dark_screen_stroke` 36 -> 27, `dark_overlength` 36 -> 17), but it does not solve the actual visible brush footprint: `visible_screen_stroke_p95` mean diagonal changes only 48.59px -> 48.39px, while `axis_brush_p95` slightly worsens and composite PSNR drops by 0.066 dB. This confirms the remaining artifact is not fixed by another global capacity penalty. |
| `white_tiger_view09_stage1_screen_shape002_view09_duntil1000_overlong1100_capacity1300_1300_20260628` | 36.7105 | Rejected and removed from the formal entry point. A small bright/gray screen-shape capacity loss lowers some child/curl/frizz statistics, but does not repair the real visible footprint: `visible_screen_stroke_p95` mean diagonal changes only 48.59px -> 48.28px, `screen_diag_p95` remains 51.89px, and `dark_screen_stroke` worsens from 36 to 42. This shows the residual long streaks are not solved by globally penalizing child radius, curl, frizz, or clump on screen-sized roots. |
| `white_tiger_view09_stage1_screen_childcompact075_view09_duntil1000_overlong1100_capacity1300_1300_20260628` | 36.7556 | Rejected. Screen-footprint extra parents received a separate child policy at split time (`width_scale=0.92`, `opacity_scale=0.95`, `spread_scale=0.75`, `clump_min=0.20`), while primary overlong split stayed on the accepted baseline. The override was active (`screen_child_override_count=384` at each split event), but the final footprint did not improve: `visible_screen_stroke_p95` mean diagonal is 48.61px and `screen_diag_p95` is 52.48px, essentially unchanged from the accepted baseline. This shows one-shot compact initialization is not enough; the remaining artifact either regrows during recovery or comes from roots not covered by the screen-extra child policy. |
| `white_tiger_view09_stage1_screen_lineage010_view09_duntil1000_overlong1100_capacity1300_1300_20260628` | 36.6283 | Rejected and removed from the formal entry point. A persistent lineage penalty was applied only to descendants of screen-footprint split roots. It successfully made those lineage roots shorter/tighter on average (`screen_footprint_lineage` mean screen diagonal 30.99px), but it did not improve the real global footprint: `visible_screen_stroke_p95` mean diagonal slightly worsened 48.59px -> 48.69px, `screen_diag_p95` stayed 52.43px, and `dark_screen_stroke` worsened from 36 to 48. This proves the remaining long strokes are not simply screen-split descendants regrowing. |
| `white_tiger_view09_stage1_overlong3child_view09_duntil1000_overlong1100_capacity1300_1300_20260628` | 37.4128 | Previous accepted baseline. Increasing overlong replacement from 2 to 3 children improves composite PSNR and reduces the length-tail categories: `overlength_gt_0080` 441 -> 353, `dark_overlength` 36 -> 20, `overlength_high_curl` 73 -> 54, `overlength_high_frizz` 71 -> 60, and `overlength_high_child_radius` 78 -> 65. It does not eliminate the visible footprint category (`visible_screen_stroke_p95` mean diagonal 48.59px -> 47.34px, root count 1,634 -> 1,922), so the next repair must separate local density, strand footprint, and color/opacity capacity rather than simply adding more children. |
| `white_tiger_view09_stage1_overlong3child_screenmildcompact_view09_duntil1000_overlong1100_capacity1300_1300_20260628` | 37.4375 | Rejected as a structural fix despite the tiny metric gain. It applies mild one-shot compact initialization only to screen-footprint split children (`width_scale=0.96`, `opacity_scale=0.96`, `spread_scale=0.90`, `clump_min=0.18`). The true footprint barely changes (`visible_screen_stroke_p95` mean diagonal 47.34px -> 47.15px, `screen_diag_p95` 51.58px -> 51.50px), while several length-tail categories regress (`dark_overlength` 20 -> 29, `overlength_high_child_radius` 65 -> 74, `overlength_high_frizz` 60 -> 73). This confirms one-shot child compactness is not the core repair. |
| `white_tiger_view09_stage1_guide4096_residual035_skipfix_overlong3child_view09_duntil1000_overlong1100_capacity1300_1300_20260628` | 37.2910 | Rejected. Guide-root interpolation with 0.35 local residual on length/bend almost removes the raw length tail (`overlength_gt_0080` 353 -> 9, `dark_overlength` 20 -> 0), but it does not solve the actual visible artifact and worsens dark projected strokes: `dark_screen_stroke` 50 -> 372, `visible_screen_stroke_p95` mean diagonal 47.34px -> 47.84px, and `axis_brush_p95` mean diagonal 42.72px -> 44.63px. This is useful evidence: remaining long stripes are dominated by screen-footprint/color-opacity behavior, not by simple true-length outliers. The diagnostic loader was fixed to restore guide residual scales from config before measuring this run. |
| `white_tiger_view09_stage1_overlong3child_opacity065_view09_duntil1000_overlong1100_capacity1300_1300_20260628` | 37.3263 | Rejected. Split children were initialized with lower opacity (`overlong_split_child_opacity_scale=0.65`, `screen_footprint_split_child_opacity_scale=0.65`) to test coverage conservation. The final visible footprint did not improve meaningfully: `visible_screen_stroke_p95` mean diagonal 47.34px -> 47.14px and final opacity stays 0.897, while composite PSNR drops and length-tail categories regress (`overlength_gt_0080` 353 -> 393, `dark_overlength` 20 -> 42, `dark_screen_stroke` 50 -> 59). This shows one-shot opacity conservation at split time is overwritten during recovery; the artifact is a training competition between coverage, width/opacity, and local color, not a split-initialization-only issue. |
| `white_tiger_view09_stage1_overlong3child_childcolor075_view09_duntil1000_overlong1100_capacity1300_1300_20260628` | 36.9631 | Rejected. Higher child-local color scale (0.75) lowers metric and worsens the footprint/length tail (`overlength_gt_0080` 503, `screen_diag_p95` mean diagonal 52.16px, `dark_screen_stroke` 59). More child-local color freedom alone is not the repair. |
| `white_tiger_view09_stage1_overlong3child_screenbudget256_192_view09_duntil1000_overlong1100_capacity1300_1300_20260628` | 37.8497 | Accepted as current baseline. Increasing the independent screen-footprint budgets to 256 dark and 192 neutral parents per event resolves candidate starvation: `overlength_gt_0080` 353 -> 146, `dark_overlength` 20 -> 11, `screen_diag_p95` mean diagonal 51.58px -> 47.35px, `visible_screen_stroke_p95` mean diagonal 47.34px -> 43.02px, and `axis_brush_p95` mean diagonal 42.72px -> 38.36px. `dark_screen_stroke` count rises 50 -> 96, so residual tail/leg dark strokes remain and the next fix must address footprint/color-density structure, not another global length clamp. |

Important diagnostic numbers:

- Previous alpha-only baseline `overlength_gt_0080`: 2,188 roots.
- Previous RGB-evidence baseline `overlength_gt_0080`: 2,183 roots.
- Previous more-target-roots baseline `overlength_gt_0080`: 2,141 roots.
- Previous overlong384 baseline `overlength_gt_0080`: 1,072 roots.
- Previous screenfoot50 baseline `overlength_gt_0080`: 1,290 roots.
- Previous screenfoot45+len0.50 baseline `overlength_gt_0080`: 1,286 roots.
- Previous child-color-scale0.40 baseline `overlength_gt_0080`: 1,185 roots.
- Previous child-color-scale0.50 baseline `overlength_gt_0080`: 1,221 roots.
- Previous screen-extra128 baseline `overlength_gt_0080`: 972 roots.
- Rejected strand-shape0.006+screen-extra128 `overlength_gt_0080`: 1,037 roots.
- Rejected all-luma screen-footprint `overlength_gt_0080`: 907 roots.
- Previous screen-extra128+neutral64 baseline `overlength_gt_0080`: 819 roots.
- Rejected screen40 threshold diagnostic `overlength_gt_0080`: 914 roots.
- Rejected screen40+childcompact diagnostic `overlength_gt_0080`: 1,015 roots.
- Current screen40-decoupled baseline `overlength_gt_0080`: 607 roots.
- Rejected screen40-decoupled spread0.75 diagnostic `overlength_gt_0080`: 139 roots, but composite PSNR fell to 33.5918.
- Rejected residual-target1 diagnostic `overlength_gt_0080`: 110 roots, but composite PSNR fell to 33.5526.
- Rejected residual-target0.25 diagnostic `overlength_gt_0080`: 110 roots, but composite PSNR fell to 33.6244.
- Rejected screen-footprint add-only diagnostic `overlength_gt_0080`: 3,584 roots, and composite PSNR fell to 28.5623.
- Candidate densify-until1100 view09 diagnostic `overlength_gt_0080`: 335 roots; composite PSNR 36.5028 versus the accepted baseline 36.6066.
- Candidate overlong-only-until1100 diagnostic `overlength_gt_0080`: 394 roots; composite PSNR 36.3648 versus the accepted baseline 36.6066.
- Overlong-only-until1100 with 200 recovery iterations `overlength_gt_0080`: 867 roots; composite PSNR 36.8187. This recovers PSNR but regrows the length tail.
- Current capacity-through-recovery baseline `overlength_gt_0080`: 441 roots; composite PSNR 36.8041.
- Previous overlong3child baseline `overlength_gt_0080`: 353 roots; composite PSNR 37.4128.
- Current screenbudget256/192 baseline `overlength_gt_0080`: 146 roots; composite PSNR 37.8497.
- Rejected guide4096+child-color0.50 `overlength_gt_0080`: 5,528 roots.
- Rejected overlong-child compactness variant `overlength_gt_0080`: 1,328 roots.
- Rejected child-opacity0.06 variant `overlength_gt_0080`: 1,292 roots.
- Guide-flow `overlength_gt_0080`: 3,954 roots.
- Guide4096 length/bend `overlength_gt_0080`: 4,270 roots. It reduced
  `dark_high_capacity` to 53, but the total long-strand population grew and
  composite PSNR fell to 34.9067.
- Early-capacity-0.12 `overlength_gt_0080`: 899 roots, but composite PSNR fell to 34.7718.
- Width-triggered split `overlength_gt_0080`: 2,422 roots; p95-width root mean dropped from 0.000558 to 0.000536, but the long-streak category did not improve.
- Compact-overlong-child split `overlength_gt_0080`: 2,309 roots. It did
  not solve the artifact; simply making split children tighter removes useful
  fitting capacity while the model recreates long/colored strokes elsewhere.
- Previous alpha-only baseline `dark_high_capacity`: 117 roots.
- Previous RGB-evidence baseline `dark_high_capacity`: 108 roots.
- Previous more-target-roots baseline `dark_high_capacity`: 125 roots.
- Previous overlong384 baseline `dark_high_capacity`: 102 roots.
- Previous screenfoot50 baseline `dark_high_capacity`: 126 roots.
- Previous screenfoot45+len0.50 baseline `dark_high_capacity`: 116 roots.
- Previous child-color-scale0.40 baseline `dark_high_capacity`: 92 roots.
- Previous child-color-scale0.50 baseline `dark_high_capacity`: 89 roots.
- Previous screen-extra128 baseline `dark_high_capacity`: 84 roots.
- Rejected strand-shape0.006+screen-extra128 `dark_high_capacity`: 86 roots.
- Rejected all-luma screen-footprint `dark_high_capacity`: 93 roots.
- Previous screen-extra128+neutral64 baseline `dark_high_capacity`: 78 roots.
- Rejected screen40 threshold diagnostic `dark_high_capacity`: 76 roots.
- Rejected screen40+childcompact diagnostic `dark_high_capacity`: 76 roots.
- Current screen40-decoupled baseline `dark_high_capacity`: 49 roots.
- Rejected screen40-decoupled spread0.75 diagnostic `dark_high_capacity`: 38 roots, but this was caused by under-fitting.
- Rejected residual-target1 diagnostic `dark_high_capacity`: 25 roots, but this was caused by under-fitting.
- Rejected residual-target0.25 diagnostic `dark_high_capacity`: 37 roots, but this was caused by under-fitting.
- Rejected screen-footprint add-only diagnostic `dark_high_capacity`: 69 roots.
- Candidate densify-until1100 view09 diagnostic `dark_high_capacity`: 49 roots, unchanged from the accepted baseline, but its max length drops from 0.10310 to 0.09157.
- Candidate overlong-only-until1100 diagnostic `dark_high_capacity`: 36 roots, with max length 0.08265.
- Overlong-only-until1100 with 200 recovery iterations `dark_high_capacity`: 19 roots, but this run regrew ordinary overlength to 867 roots.
- Current capacity-through-recovery baseline `dark_high_capacity`: 43 roots.
- Rejected guide4096+child-color0.50 `dark_high_capacity`: 12 roots.
- Rejected overlong-child compactness variant `dark_high_capacity`: 94 roots.
- Rejected child-opacity0.06 variant `dark_high_capacity`: 123 roots.
- Rejected child-width0.90 run `dark_high_capacity`: 137 roots.
- Previous more-target-roots baseline `dark_screen_stroke`: 373 roots.
- Previous overlong384 baseline `dark_screen_stroke`: 351 roots.
- Previous screenfoot50 baseline `dark_screen_stroke`: 335 roots.
- Previous screenfoot45+len0.50 baseline `dark_screen_stroke`: 287 roots.
- Previous child-color-scale0.40 baseline `dark_screen_stroke`: 244 roots.
- Previous child-color-scale0.50 baseline `dark_screen_stroke`: 212 roots.
- Previous screen-extra128 baseline `dark_screen_stroke`: 84 roots.
- Rejected strand-shape0.006+screen-extra128 `dark_screen_stroke`: 85 roots.
- Rejected all-luma screen-footprint `dark_screen_stroke`: 274 roots.
- Previous screen-extra128+neutral64 baseline `dark_screen_stroke`: 112 roots.
- Rejected screen40 threshold diagnostic `dark_screen_stroke`: 60 roots.
- Rejected screen40+childcompact diagnostic `dark_screen_stroke`: 47 roots.
- Current screen40-decoupled baseline `dark_screen_stroke`: 60 roots.
- Rejected screen40-decoupled spread0.75 diagnostic `dark_screen_stroke`: 0 roots, but composite PSNR collapsed to 33.5918.
- Rejected residual-target1 diagnostic `dark_screen_stroke`: 0 roots, but composite PSNR collapsed to 33.5526.
- Rejected residual-target0.25 diagnostic `dark_screen_stroke`: 1 root, but composite PSNR collapsed to 33.6244.
- Overlong-only-until1100 with 200 recovery iterations `dark_screen_stroke`: 25 roots, but ordinary overlength regrew to 867 roots.
- Current capacity-through-recovery baseline `dark_screen_stroke`: 36 roots.
- Rejected screen-footprint add-only diagnostic `dark_screen_stroke`: 294 roots, so add-only local screen-root coverage worsens the visible dark-stroke failure instead of fixing it.
- Candidate densify-until1100 view09 diagnostic `dark_screen_stroke`: 24 roots; this is a real improvement without under-fitting, but the PSNR cost means it remains a candidate rather than the accepted baseline.
- Candidate overlong-only-until1100 diagnostic `dark_screen_stroke`: 27 roots. This is close to the full densify-until1100 improvement and shows that late ordinary target-direct densification is not required for the dark screen-stroke fix.
- Rejected guide4096+child-color0.50 `dark_screen_stroke`: 125 roots.
- Previous overlong3child baseline `dark_screen_stroke`: 50 roots.
- Current screenbudget256/192 baseline `dark_screen_stroke`: 96 roots, but with smaller mean projected diagonal than the previous overlong3child baseline.
- Rejected overlong-child compactness variant `dark_screen_stroke`: 223 roots.
- Rejected child-opacity0.06 variant `dark_screen_stroke`: 209 roots.
- Rejected child-width0.90 run `dark_screen_stroke`: 285 roots.
- Compact-overlong-child split `dark_high_capacity`: 122 roots.
- Dark-stroke experiment `dark_high_capacity`: 1 root, but composite PSNR fell to 34.3829 and total overlength rose.

Conclusion: the remaining streaks are not a simple "single root too long" bug.
The visible artifact is a capacity-entanglement problem: length, width, opacity,
flow, color, child spread, and adaptive segment count can combine into a
continuous brush-like region. Guide root smoothing is not automatically safe; if
it controls only length/bend/flow without handling color/opacity/capacity, it
can make the long-streak structure more coherent and worse.

The failed capacity and width-triggered split tests sharpen the diagnosis:
globally suppressing length/width/opacity can reduce numeric tails, but it also
removes legitimate reconstruction capacity. Conversely, replacing already-wide
roots does not prevent the model from forming new long brush strokes elsewhere.
The guide4096+child-color0.50 run sharpened the diagnosis further: low-frequency
guide roots can reduce dark-stroke counts, but they do so by making overlength
regions more coherent and spatially broader. The next repair should focus on
local representation, not another global threshold and not guide-only
length/bend smoothing: stripe detail should be explained by denser local
roots/shorter local strands, while long single-color strands should be prevented
from dragging one stripe color across neighboring white fur.

## Symptom

The current single-view baseline reaches a reasonable composite PSNR, but the
rendered fur contains long dark streaks. The artifact looks like black tiger
stripes are being explained by long, curved, high-opacity fur strands rather
than by local color variation on short fur.

Reference run:

`D:\petsgaussianhair\server_pull\white_tiger_stage1_server_20260626070424\metrics.jsonl`

At iteration 3000:

- raw PSNR: 31.505
- composite PSNR: 32.618
- RGB L1: 0.01956
- orientation loss: 0.4541
- orientation detail loss: 0.8234
- root count: 18,341
- Gaussian count: 2,162,315

The metric is not the main issue. The image is visually wrong because some
strands become too long, soft, and high-opacity.

## Evidence From Current Baseline

Current dense groom range in
`D:\petsgaussianhair\tools\train_white_tiger_stage1.py`:

```text
length=(0.040, 0.220)
curl_radius=(0.0, 0.026)
curl_frequency=(0.0, 5.5)
frizz=(0.0, 0.010)
child_radius=(0.0, 0.012)
clump_strength=(0.0, 1.0)
opacity=(0.05, 0.98)
```

At iteration 3000, the learned parameter tails are:

```text
length mean 0.0691, p95 0.1057, max 0.2177
stiffness mean 0.4543, min 0.0675
curl_radius p95 0.00939, max 0.02587
frizz p95 0.00203, max 0.00813
child_radius p95 0.00583, max 0.01089
clump_strength mean 0.0779
opacity p95 0.9781
tip_opacity p95 0.8751
```

This combination is exactly the failure mode seen in the render:

- a few roots can grow very long strands;
- low stiffness and nonzero curl/frizz make them curve or wobble;
- low clump strength lets child strands spread instead of staying tight;
- high opacity makes dark strands occlude nearby white fur;
- RGB reconstruction then rewards this because black stripes can be painted by
  long strands.

The formal config also makes RGB the dominant term:

```text
RGB_WEIGHT=1.0
ORIENTATION_WEIGHT=0.02
ORIENTATION_DETAIL_WEIGHT=0.02
SMOOTH_WEIGHT=0.025
STRAND_SHAPE_SMOOTH_WEIGHT=0.012
SHAPE_PRIOR_WEIGHT=0.015
```

So the current objective lets RGB pull geometry more strongly than orientation
and shape priors can keep it hair-like.

## NeuralFur / GaussianHaircut Contrast

NeuralFur's released GaussianHaircut path avoids this failure with several
strong constraints:

1. It does not mainly optimize RGB for fur shape.
   `simple_run_panda.sh` sets orientation, mask, chamfer, SDF, shape, and
   gravity losses, while `lambda_dl1` defaults to 0 in the argument parser.

2. Length is region-controlled.
   `metrical_panda_furless_15k_small.yaml` has a `mapping_length` table for
   body, face, legs, tail, paws, etc. The strand generator normalizes decoded
   arc length and multiplies by region length. Length does not freely grow to
   explain stripes.

3. Shape comes from a pretrained strand prior.
   `optimizable_textured_fur.py` loads `strand_ckpt.pth`; the decoder is frozen
   and only texture/latent parameters are optimized. This restricts shape to a
   hair-like manifold.

4. Each Gaussian is a thin strand segment.
   `gaussian_model_latent_strands.py` places Gaussians at adjacent strand
   segment midpoints, sets the long axis from segment length, uses a small
   global transverse `strand_scale`, and returns opacity 1. This produces
   sharp line-like fur instead of soft free blobs.

5. It uses SDF/chamfer/shape/gravity constraints:
   - SDF: prevents penetration into the body.
   - Chamfer: attracts strands toward outer surface.
   - Shape consistency: penalizes inconsistent curvature profiles.
   - Gravity/region directions: keeps body-region flow plausible.

## Root Cause

The long dark streaks are not primarily a mesh alignment or depth-clipping bug.
They are an objective/parameterization failure:

```text
RGB-dominant optimization
  + too-large length range
  + high-opacity strand color shared along the whole strand
  + weak orientation/shape constraints
  + low clump strength and wide child spread
  -> dark stripes become long geometry strokes
```

The current baseline can raise PSNR while hurting visual hair quality because
RGB loss accepts "black paint strokes" as a cheap explanation for tiger stripes.

## Repair Direction Without Polluting Baseline

The next experiment should be a separate config/branch/run, not a baseline code
mutation. The purpose is to prove the root cause by making a
NeuralFur-style sharp-fur variant.

Recommended controlled changes:

1. Restrict long-tail geometry.
   - Reduce `length` upper bound or add a stronger prior above 0.075-0.10.
   - Increase stiffness prior or lower the allowed low-stiffness tail.

2. Reduce high-frequency curve freedom early.
   - Lower LR for bend/curl/frizz.
   - Freeze or heavily damp curl/frizz for the first training phase.
   - Raise `strand_shape_smooth_weight` to actually affect the objective.

3. Keep child strands tighter.
   - Lower `child_radius`.
   - Raise initial/minimum `clump_strength`.

4. Make shape supervision stronger than stripe painting.
   - Lower RGB weight temporarily, or delay strong RGB until alpha/mask/orient
     are stable.
   - Increase orientation and orientation-detail weights for the geometry phase.

5. Keep color local.
   - Avoid letting one dark root color dominate a long visible strand.
   - If needed, separate geometry fitting from color fitting: first shape,
     then color.

Verification gate:

- The same view09 crop should show fewer long black streaks.
- `length max` and `length p95` should stay bounded.
- `stiffness` should not collapse.
- `clump_strength` should not collapse toward zero.
- composite PSNR should not be the only pass criterion; strand-line visualization
  must look shorter, sharper, and less chaotic.

## 2026-06-28 Multiroot Diagnostics

These runs are diagnostic only. They must not replace the accepted single-root
baseline unless both metric quality and long-streak diagnostics improve.

Accepted baseline remains:

```text
D:\petsgaussianhair\_downloads\white_tiger_view09_stage1_overlong3child_screenbudget256_192_view09_duntil1000_overlong1100_capacity1300_1300_20260628
```

Comparison images:

```text
D:\petsgaussianhair\_downloads\white_tiger_stage1_multiroot_final_torso_compare_20260628.png
D:\petsgaussianhair\_downloads\white_tiger_stage1_multiroot_final_streak_compare_20260628.png
```

Same diagnostic/eval path, view09, full resolution:

| Run | Composite PSNR | Raw PSNR | Composite SSIM | `overlength_gt_0080` | `dark_overlength` | `dark_high_capacity` | `dark_screen_stroke` | `visible_screen_stroke_p95` | `length_p95` mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| accepted single-root baseline | 37.8497 | 35.0037 | 0.9724 | 146 | 11 | 50 | 96 | 2098 | 0.07670 |
| flowprior multiroot | 37.5480 | 34.8164 | 0.9708 | 153 | 14 | 88 | 125 | 2090 | 0.07667 |
| guideprior multiroot | 37.6908 | 34.9049 | 0.9718 | 182 | 22 | 77 | 97 | 2095 | 0.07684 |
| residual unlock schedule | 36.6582 | 34.3366 | 0.9651 | 490 | 64 | 96 | 103 | 2176 | 0.07882 |
| hard guide length+bend | 36.6557 | 34.3834 | 0.9646 | 15035 | 2639 | 0 | 83 | 2131 | 0.09542 |

Conclusions:

1. Flowprior and guideprior are close to the accepted baseline numerically, but
   neither reduces the long-streak categories enough to justify replacing the
   baseline.
2. The residual unlock schedule is not a clean repair. It drops roughly 1.19 dB
   composite PSNR and increases overlength categories.
3. Hard guide control over only `length` and `bend` is worse. Render-root
   decoded length stays fixed, but the effective guide-interpolated length grows
   to a high-p95 field. This converts the problem into a broad overlength
   guide field: `overlength_gt_0080` rises from 146 to 15035.
4. Therefore the current multiroot variants do not solve the artifact. The
   failure is not just "render root residual too free"; RGB/detail pressure can
   also push the guide geometry field itself into long paint-stroke behavior.

Next direction:

- Keep the accepted single-root baseline as the formal reference.
- Do not continue tuning simple guide residual schedules.
- If multiroot is revisited, guide geometry must be protected by stronger
  non-RGB evidence: cleaned/multi-view orientation anchors, explicit strand
  shape priors, or a split policy that redistributes long projected strokes
  instead of allowing guide length to absorb RGB residual.

## 2026-06-28 Dense Root Count Sweep

Purpose: test whether the previous 20k-ish root field is too sparse. The
hypothesis is that a sparse root field lets one root explain too much image area,
so RGB loss can turn individual strands into long dark paint strokes. These runs
keep the accepted lifecycle/loss recipe and only change initial root/child
capacity.

Comparison images:

```text
D:\petsgaussianhair\_downloads\white_tiger_dense_root_sweep_compare_20260628\render_torso_gt_baseline_50k_100k.png
D:\petsgaussianhair\_downloads\white_tiger_dense_root_sweep_compare_20260628\render_torso_zoom2x.png
D:\petsgaussianhair\_downloads\white_tiger_dense_root_sweep_compare_20260628\dark_stroke_baseline_50k_100k.png
```

Full-resolution view09 results:

| Run | Composite PSNR | Raw PSNR | Composite SSIM | Roots | Gaussians | `length_p95` | `length_max` | `overlength_gt_0080` | `dark_overlength` | `dark_screen_stroke` | `visible_screen_stroke_p95` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| accepted baseline, 20k-ish roots / 4 child | 37.8497 | 35.0037 | 0.9724 | 41942 | 3102581 | 0.07382 | 0.11108 | 146 | 11 | 96 | 2098 |
| dense 50k roots / 2 child | 38.5042 | 35.3164 | 0.9761 | 73346 | 2775922 | 0.07680 | 0.11289 | 1026 | 115 | 206 | 3668 |
| dense 100k roots / 1 child | 39.2636 | 35.6725 | 0.9795 | 123550 | 2324198 | 0.07894 | 0.10622 | 4551 | 523 | 509 | 6178 |

Conclusions:

1. Root count is a major variable. 100k roots / 1 child improves composite PSNR
   by +1.41 dB over the accepted baseline while using fewer final Gaussians.
2. The improvement is not just a Gaussian-count effect. 100k/1child has fewer
   Gaussians than the accepted baseline but better reconstruction, so denser root
   placement reduces the area each strand must explain.
3. Dense roots do not automatically solve the long-streak failure. Absolute and
   relative long-streak categories increase, especially in dark stripe regions.
   Visually the 100k render is closer to GT at normal scale, but diagnostics show
   it still uses many fine projected strokes to explain stripes.
4. Therefore 100k/1child is a strong candidate for the next metric baseline, but
   it must be paired with a real stroke-control mechanism before being treated as
   the final asset-quality baseline.

Next direction:

- Use 100k/1child as the dense-root candidate for subsequent controlled
  experiments.
- Do not merely raise root count further on the same local GPU; 100k already
  reaches high memory pressure in training and still leaves stroke artifacts.
- The next repair should target dense-root stroke behavior directly: better
  multi-view orientation anchors, local strand-shape consistency, or a split
  rule that shortens/replaces projected long strokes without adding ad-hoc
  animal-specific thresholds.

## 2026-06-29 Structured Stage1 Foundation

Purpose: replace the earlier overlong/screen-footprint heuristic baseline with a
cleaner Stage1-A/B foundation. The new line separates smooth guide-root geometry
from render-root color/detail fitting, then uses image-evidence densification
for capacity growth. Screen/luma overlong split heuristics are disabled.

Validated config:

```text
D:\petsgaussianhair\configs\white_tiger_stage1_multiroot_stageab_structured70k_strict_view09_local.env
```

Formal config now points to that validated foundation:

```text
D:\petsgaussianhair\configs\white_tiger_stage1_formal.env
```

Runs:

```text
D:\petsgaussianhair\_downloads\white_tiger_view09_stage1_stageab_structured70k_budget_1300_20260629
D:\petsgaussianhair\_downloads\white_tiger_view09_stage1_stageab_structured70k_strict_1300_20260629
```

Full-resolution view09 results:

| Run | Raw PSNR | Composite PSNR | Composite SSIM | Roots | Visible Gaussians | `overlength_gt_0080` | `dark_overlength` | Effective length p95 | Effective length max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| structured70k budget | 34.3643 | 36.6771 | 0.9662 | 100720 | 1377136 | 492 | 105 | 0.06655 | 0.13538 |
| structured70k strict | 34.3876 | 36.7314 | 0.9669 | 100720 | 1350616 | 121 | 32 | 0.06026 | 0.12451 |

Conclusion:

1. The strict structured version keeps the same reconstruction quality as the
   budget version while sharply reducing overlength categories.
2. The remaining long-strand artifacts are now a small tail of the effective
   guide-controlled geometry, not the raw render-root length. Raw render-root
   length stays fixed at 0.042 throughout the run.
3. This is the current Stage1 foundation for the next step. It is not the final
   animal asset result yet, but it is a clean baseline that can be extended to
   multi-view training without relying on the removed screen/luma split
   heuristics.

Next gate:

- Use `white_tiger_stage1_formal.env` for the next verified run.
- Keep composite PSNR as the primary reconstruction metric, but always inspect
  the torso/hip crops and overlength diagnostics.
- Next improvements should be structural: better guide-root geometry evidence,
  cleaned/multi-view orientation anchors, or true multi-level guide/render-root
  controls. Do not reintroduce screen/luma-specific split budgets as the main
  solution.
