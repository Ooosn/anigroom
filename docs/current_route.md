# Current Route

Status date: 2026-08-08.

This is the only source of truth for active Stage 1 behavior. The recovery
ledger records measured experiments, but it does not define executable schema.

## Baseline Status

R043 is the active structural/lifecycle Stage 1 baseline:

- final train/test composite PSNR: `33.46581 / 32.51159`
- best test composite PSNR: `32.71421` at 29k
- final render roots/training-metric Gaussians: `469620 / 5295653`
- elapsed H100 training time: `17388.655 s`
- peak allocated CUDA memory: `19733.46 MB`
- formal checkpoint SHA-256:
  `748018581c0eac02eefe1f2361c10dbfe0fa5e5a742ff3beee2e3163c143a632`
- canonical asset render:
  `D:/RTS/_tmp/r043_30k_final/r043_030000_asset_side_y_v11_protocol.png`

The `stage1-r043` tag and
`configs/r043_density_matched_render_support.lock.json` freeze the exact
source, behavior configuration, formal metrics, checkpoint, and postprocess
evidence. R043 uses 400k initial independent render roots with
`child_count=1`; no deterministic child expansion is part of the active route.
Its render-domain surface support is K32, matching the fourfold independent-root
density increase from R039, while guide-domain support remains K8. R042 is the
frozen K8 parent.

R036 remains the immutable higher-PSNR metric control:

- final train/test composite PSNR: `33.42397 / 32.66322`
- best test composite PSNR: `32.83977` at 29k
- final render roots/Gaussians: `317245 / 14215421`
- formal checkpoint SHA-256:
  `b4bedaa3d111c529e38d43ed7ffd922391292a5977934d10c4f10b7aeaf85f6b`
- canonical asset render:
  `D:/RTS/_tmp/r036_30k_final/r036_030000_asset_side_y_v11_protocol.png`

The `stage1-r036` Git tag is byte-identical to its formal runtime snapshot, and
`configs/stage1_baseline.lock.json` continues to verify that control. R043 is
`0.15163 dB` lower at the final test metric and `0.12556 dB` lower at the best
test metric, while using about `62.7%` fewer generated Gaussians. It is
accepted for independent-root structure, exact finite lifecycle, lower
capacity cost, density-matched surface smoothing, and coherent fixed-protocol
assets, not reported as a PSNR gain over R036. Relative to R042, final/best
test composite changes by only `-0.00384/-0.00496 dB`, while the 100k-strand
audit removes every length above `0.15` and reduces maximum local turn from
`14.89` to `2.73` degrees. The exact v4 target remains tracked under
`baseline_inputs/` with unchanged content.

R050 is the frozen appearance diagnostic parent, not an accepted color
decomposition. It keeps R049's 20k secondary geometry field and adds a
normalized arc-length Gaussian RGB residual profile. Final/best test composite
reaches `32.12111/32.20936`, improving R049 by `+0.51791/+0.46836` dB, and the
fixed 100k-strand audit retains R049's structural advantage. However, the
same-checkpoint four-layer ablation shows that the old render-root color field
and the Gaussian residual both learn image noise and cancel one another.
Disabling only the Gaussian residual loses `0.88-2.73` dB and visibly increases
noise; disabling the old local color loses still more. R050 therefore proves
that Gaussian-level RGB residuals are useful, but not that the intended
low-frequency/high-frequency separation was achieved.

R051 is the active appearance experiment. It makes sparse primary-guide
root/tip colors the only low-frequency fur-color field, interpolates that field
to render roots through the existing surface support, removes the competing
local render-root color field, and retains the Gaussian RGB residual as the
only high-frequency outlet. The guide color field learns through 10k and is
then frozen with gradients set to `None`; the existing Gaussian residual ramp
starts at the same boundary. Formal H100 validation is pending. See
`docs/r050_gaussian_rgb_residual.md` and
`docs/r051_guide_color_decomposition.md`.

## Active Entry Points

- training: `tools/train_white_tiger_stage1.py`
- frozen R036 metric-control configuration: `configs/stage1_baseline.env`
- active R043 behavior configuration:
  `configs/r043_density_matched_render_support_0_30k.env`
- active R043 lock: `configs/r043_density_matched_render_support.lock.json`
- active R049 geometry-parent configuration:
  `configs/r049_secondary_guide_resume16k_30k.env`
- frozen R050 diagnostic-parent configuration:
  `configs/r050_gaussian_rgb_residual_0_30k.env`
- active R051 appearance experiment configuration:
  `configs/r051_guide_color_gaussian_residual_0_30k.env`
- historical R038/R039 configurations remain evidence, not fallbacks
- server launcher: `scripts/server/run_white_tiger_stage1.sh`
- strand export: `tools/export_white_tiger_checkpoint_strands.py`
- Gaussian export: `tools/export_white_tiger_checkpoint_gaussians_ply.py`
- checkpoint rendering: `tools/render_white_tiger_stage1_checkpoint_views.py`
- groom diagnostics: `tools/visualize_white_tiger_groom_attributes.py`

The launcher requires an explicit `CONFIG_PATH`. There is no fallback route,
retired configuration migration, or alternate executable baseline.

## Representation Contract

Each guide and render root stores one normalized local 3D direction. Its three
components are expressed in the root surface frame and are converted to world
space only through that frame. Direction interpolation, transport, smoothing,
initialization, and residual composition all operate on the same 3D vector
field.

The active explicit groom fields are:

- length
- root and tip width plus taper
- normalized local 3D direction
- guide-owned brush stiffness
- curl radius, frequency, and phase
- frizz
- child radius and clump strength
- root and tip color
- root and tip opacity

Brush stiffness controls one quadratic normal-to-groom transition while
preserving the root, tip, straight length, and 3D endpoint direction. The
effective value is brush stiffness multiplied by the continuous tangential
difference between the normal and endpoint direction. There is no second
interior deformation field. Curl and frizz remain separate optional shape
controls but stay disabled in R043. The active route has one strand per render
root (`child_count=1`); density comes from independent render roots and their
finite lifecycle, not deterministic child expansion.

Render-root geometry is a zero-centered residual around the interpolated guide
field. Direction residuals are local 3D vectors. Length uses the positive,
guide-relative `exp(asinh(raw))` coordinate. Zero residual means exact guide
interpolation.

Guide length, root width, and child spread use the same positive coordinate around a local
reference: `reference * exp(asinh(raw))`. Length references come from clean-flow
evidence; width and child-spread references preserve initialization through
lifecycle changes. None of these fields has an animal-scale minimum or maximum. Guide roots own tip/root
ratio and width taper. Render
roots carry only zero-centered, guide-relative width-profile residuals, which
unlock with the shared geometry schedule. Root opacity and tip-opacity ratio use their semantic
sigmoid domain `[0, 1]`; there is no padded decoder interval.

Tip width is a semantic `[0,1]` ratio of root width. Width taper is positive
and unbounded. The retired absolute root-width interval is no longer present.
Render child spread is a positive guide-relative residual and has no absolute
physical endpoint. It follows the measured R036 1k-7k coverage ramp rather
than the later geometry-residual unlock.

## Initialization And Interpolation

The clean-flow module supplies surface points, normalized 3D directions,
confidence, and length evidence. Guide roots are initialized directly from
that field. Unsupported values are reconstructed through surface-aware
neighbors; render roots then sample the guide field through the same surface
interpolator used during training and lifecycle updates.

Clean-flow length initialization keeps the observed 5%-95% interval only as a
robust anchor filter. Surface inpainting reconstructs the complete positive
reference field, and the trainable guide coordinate starts at zero. No fixed
physical length interval is applied before or after inpainting.

No render-to-guide reverse initialization exists. No second directional
parameterization is stored beside the normalized 3D vector.

## Training Contract

- full resolution: `1920x1080`
- initial guide/render roots: `4500 / 400000`
- render-root lifecycle: every 100 iterations from 600 through 9k
- guide-root lifecycle: disabled in R043
- guide fields unlock after 9k; render-root geometry residuals ramp 10k to 20k
- render child-spread coverage unlock: 1k to 7k
- shape detail freeze: through 14k, then shared gradual unlock
- pruning: disabled in the current baseline

Root lifecycle uses accumulated per-Gaussian evidence mapped to owning roots,
surface local maxima, and topology-local child placement. New rows inherit
attributes through the same surface interpolation contract. Optimizer state is
migrated for surviving rows and initialized at zero for new rows.

Adaptive strand sampling uses the accepted absolute physical-length and
curvature linear formula. Only a minimum representation count remains. There is
no initialization-derived sampling reference, configured maximum, or final
upper clamp; sufficiently long or complex hair receives more segments.

The active regularization is relative and surface-aware: guide-field
smoothness, effective-groom smoothness, strand-shape smoothness, clean-flow
guide anchoring, and residual concentration. The length residual term keeps the
accepted mean absolute prior and adds the population-stable concentration term
`L4 - L2` during unlock.

R051 stores root/tip color only on sparse primary guide roots. The existing
topology-safe guide support interpolates those colors to every render root;
there is no trainable render-root color or local child-color term. Guide colors
use the guide graph smoothness already applied to the parameter field. They
learn the main fur color through 10k and are frozen thereafter.

The view-independent Gaussian RGB profile from R050 remains. Each generated
Gaussian samples that profile at its normalized segment midpoint. It is exactly
inactive through 10k and ramps with the common schedule to full strength at
20k. It has no TV or smoothness loss because its role is to absorb remaining
high-frequency image evidence. Pure-fur asset export intentionally omits it.

## Representation Lineage

The retired directional decomposition and old gravity/sag controls are no
longer present. R043 uses only length, normalized 3D endpoint direction, and
guide-owned brush stiffness for the ordinary base centerline. Curl/frizz and
clump remain separate explicit controls; curl/frizz have zero effective scale
in this short-fur baseline.
R035 keeps R034's accepted segment repair, positive unbounded guide length, and
semantic opacity. It replaces direct render-root width learning with a complete
guide/render width hierarchy and removes the absolute root-width range. R036
extends the same ownership contract to child spread and removes its active
physical endpoint. The formal and local R036 code both retain the historical
1k-to-7k child-spread coverage ramp. R037's proposal to move every coverage
field onto that ramp was deferred and is not active. The checkpoint schema
persists local length, width, and child-spread references but no segment
reference, maximum-segment field, or absolute endpoint for those positive
physical fields. Each schema-changing candidate must train from zero and
intentionally does not load the preceding strict-schema checkpoint.

Any later candidate replaces the active structural reference only after:

1. from-zero training completes with the strict current schema;
2. full-resolution train/test composite metrics are recorded;
3. canonical pure-fur renders are inspected as single large images;
4. lifecycle, memory, optimizer-state, and checkpoint integrity checks pass.

### Decoder-range evidence

The direct-3D 30k checkpoint establishes the audit order. These values are
measured from the active checkpoint, not inferred from source defaults:

- effective render length is already positive and unbounded; its observed
  range is `0.00765` to `0.11377`, so the retired `0.105` endpoint is not an
  effective render-length cap;
- guide length was actually sigmoid-decoded through the formal
  `[0.010, 0.220]` range, despite older documentation claiming
  `[0.012, 0.105]`; `9.19%` of guide roots lie in its lowest one percent;
- in the completed R033 checkpoint, `68.75%` of render tip-width ratios and
  `82.61%` of width-taper coordinates lie in the highest one percent of their
  configured intervals; R034 replaced those intervals with semantic tip ratio
  and positive unbounded taper coordinates;
- R034 then shows that removing endpoints is not sufficient: `62.15%` of
  render roots learn tip/root ratio at least `0.99`, so R035 moves the complete
  width profile into the guide/render hierarchy rather than restoring caps;
- `97.83%` of render opacity coordinates lie in the highest one percent of the
  actual formal `[0.05, 0.98]` interval. The padded endpoints are not
  justified, so R033 uses the semantic `[0, 1]` domain;
- root width is not materially saturated; curl and frizz have zero effective
  scale in this baseline and therefore cannot explain current geometry.

R033 replaced the inconsistent old normalization with an initialization-relative
spacing rule, but that changed mean segments from `10.928` to `20.102` while
physical length changed only about 5-7%. R034 instead keeps the old absolute
linear calibration and removes its upper clamps. On the fixed R033 100k export,
the corrected mean/max are `11.041/17`.

R036 removes the active child-radius bound. Curl and frizz retain old bounded
definitions but have zero effective scale in the current route; changing their
inactive representation is intentionally not mixed into this experiment.
Numerical epsilon clamps, normalized directions, RGB/opacity domains, tip
ratios, and clump interpolation weights are representation or semantic domains
rather than animal-specific physical thresholds; they remain.

The active representation design is defined in
[`brush_curve_representation.md`](brush_curve_representation.md): straight
root-to-tip length, one guide-owned direction-aware quadratic curve, and
final-curve Gaussian allocation. R038 and R039 remain historical structural
evidence in
[`r038_brush_curve_and_9k_lifecycle.md`](r038_brush_curve_and_9k_lifecycle.md),
and
[`r039_one_turn_brush_centerline.md`](r039_one_turn_brush_centerline.md).
R040 establishes independent dense render roots, R041 accelerates their exact
surface graph, R042 accelerates exact lifecycle selection, and R043 restores
density-matched physical support to the independent render field. The active
result is recorded in
[`r043_density_matched_render_support.md`](r043_density_matched_render_support.md).

## Execution Memory Contract

The model and loss are unchanged across accelerator sizes. On GPUs with at
most 48 GiB, training uses activation recomputation for strand-to-Gaussian
construction and strand-shape consistency. Larger GPUs retain the direct
autograd path. Forward Gaussian values and one-step parameter updates were
compared from the same 30k checkpoint; the maximum FP32 difference was
`4.77e-7`.

Different views retain different numbers of depth-clipped Gaussians, so the
process enables PyTorch expandable CUDA allocator segments before importing
PyTorch. This prevents incompatible cached blocks from growing until WDDM
pages CUDA memory into system RAM. The memory guard also checks total device
usage because per-process `nvidia-smi` accounting is unavailable under WDDM.

The original memory repair was measured locally at a historical 30k population
(`322,222` render roots, `14.21M` preclip Gaussians):

- old second-step backward: `216.22 s`; full card usage: about `31.99 GB`;
- repaired 30-step multi-view steady state: `1.638 s/iteration`;
- repaired peak live PyTorch allocation: `18.68 GB`;
- repaired reserved allocation: `23.82 GB`;
- repaired total device use: about `26.41 GB`, stable across the run.

The remaining steady-state cost is model work rather than allocator failure. A
synchronized 30k step spends `1.31 s` in backward (including recomputation),
while a rasterization forward is about `0.04 s`; strand-to-Gaussian generation
and its backward dominate. At the 9k checkpoint, recomputation changes one-step
time only from `2.09 s` to `2.13 s` under synchronized instrumentation, while
reducing peak live allocation from `13.07` to `9.09 GB` and total device use
from `18.45` to `12.34 GB`.

No image resolution, root/strand/Gaussian count, segment budget, renderer
setting, loss, or optimization schedule was reduced for this repair.

The formal R043 H100 run validates the same execution contract at `469620`
independent render roots. It completes in `17388.655 s` with `5295653`
training-metric Gaussians and `19733.46 MB` peak allocated CUDA memory, without
OOM, restart, fallback, or use of a second GPU. Exact K32 graph construction is
cached and fast; the remaining cost is repeated per-iteration traversal of the
larger edge set. The next execution candidate must preserve exact R043 losses
and gradients rather than reducing K or model capacity.
