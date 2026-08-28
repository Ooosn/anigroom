# Current Route

Status date: 2026-08-28.

This is the only source of truth for active Stage 1 behavior. The recovery
ledger records measured experiments, but it does not define executable schema.

R068 is the accepted current single-sample method baseline. R067 remains the
exact no-frizz/crossing-enabled control, R066 remains the learned-turn parent,
and R065 remains the earlier exact crossing reference. R043 remains
the underlying structural/lifecycle baseline, R050 remains the accepted
Gaussian-RGB appearance reference, and R062 remains the direct no-penetration
control. These roles are distinct and are not interchangeable.

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
`14.89` to `2.73` degrees. The exact current target is tracked under
`baseline_inputs/v7_surface_direction/`. V6 is its immutable fixed-axis parent,
V5 is the completed trained rollback, and V4 remains the older clean-axis
ablation.

A 2026-08-27 Panda view-27 investigation attributes two remaining local
discontinuities to multiview raw-axis fusion, not curl or training. The
V6 `trusted-view-cluster` mode gives tangent ownership to robust multiview
clusters, then fits only a guarded nonnegative normal/tangent ratio from direct
selected-shell evidence. It supersedes old whole-vector final consensus. Panda
shoulder/upper-back local jumps become `24.77/29.22 deg`; matched white-tiger
view-27 and direct all-view evidence both improve. V6 was then reopened when a
directed audit found that `abs(dot)` metrics hid local arrow-head reversals:
Panda final directed-negative observed edges were `1754 / 54244`, with a
`179.93 deg` maximum.

V7 keeps the V6 tangent axis, computes exact multiview `+/-` sign evidence,
solves canonical trusted connected blocks over the parallel-transport graph,
then refits only the nonnegative normal/tangent ratio with a directed
sequential guard. The method has no species, region, root, view-index, or image
coordinate rule. Formal Panda/white target generation at source commit
`0712587e2c32c621f5566b7a8706c9dc061fc85b` preserves `4194/4407` observed
roots, changes `112/62` global signs, accepts `527/490` post-sign ratio updates,
resolves `322/174` severe directed edges, and introduces zero.

Both samples pass fixed original-resolution views `00/09/18/27`. A matched
64-step graph-streamline audit reduces Panda/white severe selected transitions
`113 -> 75` and `144 -> 113`; Panda view-27 crop transitions become `13 -> 0`
and two-cycles `2 -> 0`, without reducing median path length or increasing
P99 convergence. V7 is the accepted initialization target; V6 remains the
fixed-axis parent and V5 remains the completed trained rollback. Its first
Panda R068 cross-sample run is now complete and recorded below.

The white-tiger V6 baseline is tracked under
`baseline_inputs/v6_surface_direction/` with SHA-256
`29d07139d6214cf9540e814a8f872128ad29999890221e8afe0b2c5599586dd1`;
the matched Panda target SHA-256 is
`b3f49317dbf9d09a2d3981dc02b48cf4dff5e67b19f900efbf0268ac270d8e29`.
The accepted V7 white/Panda hashes are respectively
`f009af820560adf19b6eedbb8bf2c5d29df00cca576be13161b4ee2ebaed6510`
and
`6a220f52b15ca996c88e71802d3309f9499ade442f79dc72300f1af12b5fa56f`.
See `docs/v7_global_directed_flow.md` and
`docs/panda_v5_multiview_discontinuity_20260827.md`.

R050 is the accepted appearance checkpoint. It keeps R049's 20k secondary
geometry field and adds only a normalized arc-length Gaussian RGB residual
profile. Final/best test composite reaches `32.12111/32.20936`, improving R049
by `+0.51791/+0.46836` dB. In the same final checkpoint, disabling only the
residual loses `0.88-2.73` dB over eight fixed full-resolution views. The fixed
100k-strand audit retains R049's structural advantage: local relative-length
mean/P95 is `0.02047/0.07741`, local direction P95 is `11.2959` degrees, and no
backward segment appears. R049 remains the residual-free structural control;
R043 remains the independent-root RGB metric control. See
`docs/r050_gaussian_rgb_residual.md`.

R055 is the accepted scheduling parent for optional curl/frizz handoff and the
exact parent of R057, not the default Stage 1 route. It first
ramps smooth primary-guide curl/frizz together with Gaussian RGB residual from
20k to 25k, then ramps a zero-centered secondary-guide relative residual from
25k to 30k. Compared with
R054 it reduces backward strands `375 -> 159`, local relative-length mean
`0.02417 -> 0.02226`, and local-turn P95 `57.30 -> 50.03 deg`, while losing
`0.132 dB` on the fixed eight-view mean. R050 therefore remains the strict
structural/appearance reference; R055 is the next controlled shape branch. See
`docs/r055_staged_primary_secondary_shape.md`.

R057 is the accepted gradient-ownership parent of R059. It changes no forward render,
loss source, weight, schedule, interpolation, lifecycle, capacity, or learning
rate from R055. It only prevents the existing RGB-derived flow backward from
updating root/tip color, optional child color, and Gaussian RGB residual. The
formal from-zero run preserves reconstruction (`+0.01589 dB` final test versus
R055), and the Gaussian residual retains a `+1.64675 dB` mean gain over eight
fixed views. Its sparse extreme foldback tail is not better, so R057 is a
gradient-ownership correction rather than a structural solution. See
`docs/r057_rgb_flow_no_color_grad.md`.

R059 is the immutable absolute-amplitude comparison for advanced geometry. It
inherits the complete R057 behavior contract and changes only the R058
curl/frizz forward geometry. Its formal run reaches final/best test composite
`32.25537/32.33647`, fixed eight-view mean `33.32501`, and 34 strict foldbacks
in one compact head-crown patch. See
`docs/r059_redesigned_groom_geometry_training.md`.

R060 is the accepted current advanced-geometry baseline. It retains the
complete R059 training contract but stores curl radius and frizz amplitude as
positive dimensionless ratios to current strand length. Physical offsets are
formed only in `build_strands` as `length * ratio`; guide interpolation,
secondary/render residuals, smoothing, and lifecycle inheritance all use the
same ratio semantics. The strict from-zero run reaches final/best test
`32.23912/32.32348`, within `0.0163/0.0130 dB` of R059, while matched 100k
strict foldbacks fall `34 -> 0`. Fixed eight-view mean is `33.29358`, and the
Gaussian RGB residual still contributes `+1.60593 dB`. R050 remains the
near-straight appearance reference, and R043 remains the structural/lifecycle
base. See `docs/r060_relative_shape_amplitudes.md`.

R061 is the frozen direct appearance control for R062. It changes
only `LOCAL_CHILD_COLOR_SUPPORT=0`, deleting the obsolete per-render-root color
delta so smooth root/tip color owns strand appearance and the generated-Gaussian
RGB residual is the sole high-frequency outlet. Its strict from-zero run reaches
final/best test composite `32.21457/32.30076`, within `0.02455/0.02272 dB` of
R060. Fixed eight-view mean is `33.23565`; the Gaussian residual contribution
increases from R060 `+1.60593` to `+1.82087 dB` with only `2.10%` saturation.
The matched 100k-strand audit retains zero backward segments and no length above
`0.12`; canonical assets show no material geometry regression. R060 remains the
local-render-color control. See `docs/r061_gaussian_only_appearance.md`.

R062 is the frozen direct validity control for R065.
It is a strict single-variable child of R061 and adds only the differentiable
continuous-strand mesh no-penetration loss. The loss queries a reviewed
mesh-local SDF, excludes surface roots, rotates deterministically over 16,384
render roots per iteration, and sends gradients to groom geometry and root
barycentric coordinates but not global mesh calibration. Its strict from-zero
run reaches final/best test composite `32.19214/32.28517`; fixed eight-view mean
is `33.21203`, only `0.02361 dB` below R061. The final all-root audit reduces
penetrating point fraction `0.134272% -> 0.023592%`, penetrating-root fraction
`0.675571% -> 0.416470%`, mean depth by `84.59%`, and maximum depth by
`51.16%`. Matched 100k-strand QA retains zero backward strands, no length above
`0.12`, and no canonical visual regression. R061 remains the immutable direct
control. See `docs/mesh_no_penetration.md`.

R065 is the accepted current advanced-geometry/appearance/validity/crossing
baseline. It keeps R062's rendering, reconstruction, no-penetration,
interpolation, lifecycle, appearance, and groom geometry contracts. It adds the
exact continuous 3D crossing active set from R063/R064, but routes crossing
gradients only through active dense zero-centered local direction, curl-radius,
and frizz-amplitude residuals. Primary guides, length, root placement, width,
and appearance receive no crossing gradient. The uninterrupted from-zero run
reaches final/best test composite `32.19859/32.28711`; fixed eight-view mean is
`33.22230`. Exact contacts at least 45 degrees fall from R062 `230` to `198`,
while sampled, primary-guide, and effective lengths all remain below `0.12`.
The no-penetration audit remains matched. R063 and R064 are rejected ownership
controls; R062 remains the immutable direct control. See
`docs/r065_local_crossing_residual.md`.

R066 is the accepted learned-turn parent, with R065 retained
as its exact parent and crossing reference. R066 keeps the R065 configuration
snapshot unchanged and adds only a direct signed, zero-initialized primary-guide
turn coordinate with explicit phase zero. Turns are interpolated from primary
guides to render roots; secondary residuals do not own a turns field. The
strict schema-8 loader rejects schema 7 without migration or aliasing.

The formal R066 evidence is tied to training commit
`46672fab4b1d6317fcdc041af067a955cb99f12b`, postprocess commit
`d912ef2fdedbcd47ccdafacb1562fbec1d2e2d53`, and checkpoint SHA-256
`21e0e3a66907067215ceb3f0232432c4cd9d0ef4bea0826a62ebd6e2d1410f06`. It
passes `184` pre-training tests and `186` postprocess tests, exits zero after
an uninterrupted 30k run, and records final train/test composite
`33.149204/32.107651`, roots/Gaussians `471605/5391612`, peak allocation
`20392.39 MB`, and wall time `15267.73 s`. The fixed eight-view mean is
`33.125197` versus R065 `33.222302` (`-0.097105 dB`).

R066 learned curl-only cumulative turn P50/P95 is `2.10359/20.48548` degrees
versus its matched fixed-1.2 control `24.60057/129.16371`. Final cumulative
P50/P95 is `15.294/68.646` versus R065 `34.179/119.282`; final arc/chord P95/P99
is `1.01402/1.03095` versus R065 `1.03666/1.08083`. Backward strands and full
foldbacks are zero. R066 has `217` versus R065 `198` contacts at least 45
degrees; crossing is secondary and is not treated as solved. Final local-turn
P99/max remains `18.973/45.387` degrees and the top extremes are
frizz-dominated. The next target is frizz, not crossing or another turns
change; frizz is not claimed solved.

Remote R066 postprocess:
`/home/wangyy/anigroom-r066-learned-curl-turns-runtime-20260824/postprocess/r066_protocol_20260825`.
Remote manifest:
`/home/wangyy/anigroom-r066-learned-curl-turns-runtime-20260824/postprocess/r066_protocol_20260825/r066_postprocess_manifest.json`.
Local acceptance:
`D:/RTS/_tmp/r066_acceptance_20260825/postprocess/r066_learned_curl_turns`.
Local quantitative comparison:
`D:/RTS/_tmp/r066_acceptance_20260825/postprocess/r066_learned_curl_turns/analysis/r066_vs_r065_metrics.json`.
Signed guide maps are under the local acceptance tree's
`attributes_view00`, `attributes_view09`, and `attributes_view32` directories;
the fixed-1.2 NPZ/report are under `components`. R065 parent/crossing and its
learned/fixed closeups remain under
`D:/RTS/_tmp/r065_acceptance_20260815/postprocess/r065_local_crossing_residual`
and `D:/RTS/_tmp/r065_curl_frizz_component_20260816`.

R067 is the frozen crossing-enabled no-frizz control. It removes frizz
cleanly from the differentiable reconstruction path while retaining the R066
learned-turn parent, the R065 crossing ownership, Gaussian RGB appearance,
clean-flow initialization, lifecycle, and mesh-SDF systems. The core
implementation is frozen: schema 9 is strict, no frizz key survives model,
config, optimizer, checkpoint, or postprocess metadata, and the standalone
`frizz_backbone` utility is disconnected from reconstruction.

The formal R067 evidence is tied to training commit
`8c010f09576f671df92ff40cdabff5886648c55e`, postprocess commit
`18217be56dc468fdb8e1fffc9f0c9c39689ddce1`, and checkpoint SHA-256
`2433812f8ab784f9b04d94c88a782121fc3c11ea9522f1053b8e5f7e150b5729`. It passes
`204` pre-training and `210` postprocess tests; uninterrupted full-resolution
30k training exits 0. Final train/test composite is
`33.101788/32.069145`; fixed eight-view mean is `33.077009` versus R066
`33.125197` (`-0.048189 dB`); roots/Gaussians are `471673/5382959`; peak
allocation is `19825.54 MB`; wall time is `15775.028 s`.

R067 final curl cumulative turn P50/P95 is `2.07234/21.20436` degrees versus
R066 final `15.29414/68.64623`; local-turn P99/max is `2.32353/3.60025` versus
`18.97307/45.38681`; arc/chord P95/P99 is `1.00553/1.02621` versus
`1.01402/1.03095`. Backward strands and full foldbacks are zero. Crossing
pairs are `14872` versus R066 `15418`, and crossing remains secondary. Direct
root-opacity/tip-ratio/tip-opacity means are
`0.9909846485/0.9454077401/0.9412825014`, versus R066
`0.9909962107/0.9456069964/0.9414174664`. Gaussian RGB residual absolute
mean/RMS/saturation are `0.0497083994/0.0790892018/0.0202916788`, versus R066
`0.0502568274/0.0796196752/0.0207626478`; there is no compensation evidence.

Remote R067 checkpoint, log, strict validation, and postprocess manifest:

- `/home/wangyy/anigroom-r067-no-frizz-runtime-20260825/outputs/r067_no_frizz_0_30k_h100_20260825/checkpoint_030000.pt`
- `/home/wangyy/anigroom-r067-no-frizz-runtime-20260825/logs/r067_no_frizz_0_30k_h100_20260825.log`
- `/home/wangyy/anigroom-r067-no-frizz-runtime-20260825/contracts/r067_postrun_strict_validation.json`
- `/home/wangyy/anigroom-r067-no-frizz-runtime-20260825/postprocess/r067_protocol_20260825/r067_postprocess_manifest.json`

Local R067 acceptance and quantitative comparison:

- `D:/RTS/_tmp/r067_acceptance_20260825/postprocess/r067_no_frizz`
- `D:/RTS/_tmp/r067_acceptance_20260825/postprocess/r067_no_frizz/r067_postprocess_manifest.json`
- `D:/RTS/_tmp/r067_acceptance_20260825/postprocess/r067_no_frizz/rgb_views/render_report.json`
- `D:/RTS/_tmp/r067_acceptance_20260825/postprocess/r067_no_frizz/analysis/r067_vs_r066_metrics.json`
- `D:/RTS/_tmp/r067_acceptance_20260825/postprocess/r067_no_frizz/analysis/r067_vs_r066_metrics.md`

Full assets, corrected closeups, signed guide maps, and opacity maps are under
the same acceptance root's `assets_blender_protocol_20260825`,
`assets_blender_rump_closeup_ortho082_20260825`, `attributes_view00`,
`attributes_view09`, and `attributes_view32` directories. The frozen method
figure is `D:/RTS/_tmp/anigroom-r067-no-frizz/paper/method/fig_parametric_groom_controls.pdf`.

A completed final-checkpoint R067 all-root no-penetration audit is recorded at
`D:/RTS/_tmp/r067_acceptance_20260825/postprocess/r067_no_frizz/no_penetration_final/report.json`
with SHA256 `da1f104c9a4f796720e72beff269525cda960c984bc46a2d240b382e527a3083`.
Its matched R065 comparison is validity evidence, not an improvement claim.
Paper readiness, including the remaining external/generalization evidence, is recorded in
`docs/paper_readiness_20260825.md`.

R068 is the accepted current single-sample method baseline. It is an exact
child of R067 that disables crossing support, weight, and refresh while adding
an output- and active-gradient-equivalent fast path for the exact-zero curl
phase. No schedule, K, loss outside crossing, resolution, lifecycle, SDF,
learning rate, root initialization, or capacity setting changes.

The uninterrupted from-zero 30k H100 run uses training commit
`3dfff62e621a91ba0d30764fdf780b2d1e247672`; validator-only commit `a7f3601`
corrects the post-run distinction between train/test metric rows and lifecycle
JSON rows. Final train/test composite is `33.117367/32.083080`; fixed eight-view
mean is `33.101055`. Roots/pre-step Gaussians are `471482/5380775`, peak
allocation is `15869.80 MB`, and wall time is `11885.196 s`. Relative to R067,
R068 is `24.66%` faster, uses `19.95%` less peak allocated memory, and improves
fixed-view mean by `0.024046 dB` without reducing capacity materially.

The matched 100k-strand audit keeps backward/foldback at zero. Local relative
length P95 changes `0.082026 -> 0.081259`, direction P95 changes
`11.4372 -> 11.4079` degrees, and local-turn max changes `3.6002 -> 4.3594`
degrees. Canonical side, opposite-side, and top/front assets show no crossing,
curl, or length regression. Exact crossing pairs change `14872 -> 14983`;
contact-axis pairs at least 45 degrees change `171 -> 208`, while chord-axis
pairs at least 45 degrees remain `157`. This small offline diagnostic change
does not justify default crossing training cost. Crossing remains an offline
diagnostic or optional refinement, not an active R068 loss.

Remote R068 checkpoint, validation, and postprocess:

- `/home/wangyy/anigroom-r068-no-crossing-zero-curl-runtime-20260826/outputs/r068_no_crossing_zero_curl_0_30k_h100_20260826/checkpoint_030000.pt`
- `/home/wangyy/anigroom-r068-no-crossing-zero-curl-runtime-20260826/contracts/r068_postrun_strict_validation.json`
- `/home/wangyy/anigroom-r068-no-crossing-zero-curl-runtime-20260826/postprocess/r068_protocol_20260826/r068_postprocess_manifest.json`

Local acceptance root:
`D:/RTS/_tmp/r068_acceptance_20260826/postprocess/r068_no_crossing_zero_curl`.
See `docs/r068_no_crossing_zero_curl.md` and
`docs/paper_readiness_20260826.md`.

The first formal cross-sample run applies the exact R068 behavior to Panda with
the accepted V5 flow target and identity mesh alignment. It completes from zero
through 30k and confirms that V5 initialization, lifecycle, Gaussian RGB
residual, and no-penetration transfer without an execution fallback. The fixed
eight-view mean composite PSNR is `29.42054`; Gaussian RGB residual contributes
`+1.25195 dB` on average. The backbone has zero backward strands and no length
above `0.12`.

This run also exposes the current generalization blocker: primary-guide curl is
about five times stronger than the Tiger control and creates visible shoulder
and leg waves while adding `+1.36 dB`. The checkpoint is accepted as
cross-sample evidence, not as a generalized structure baseline. See
`docs/panda_r068_v5_cross_sample.md`.

The matched Panda R068+V7 run then changes only the accepted clean-flow target
and trains strictly from zero through 30k at source commit
`58bba7b7ea66745cf79346aa8e7046b08b9ea3a5`. Final train/test composite is
`29.815145/28.763426`; roots/preclip Gaussians are `669143/7891276`; peak
allocated CUDA memory is `22407.746 MB`; and the final checkpoint SHA-256 is
`fb8c52ab50c7a879e6f18d2d1b2fd12475b276be89b194913c0507a843dc0ec2`.
The test metric is effectively unchanged from V5 (`-0.0096 dB`), so this run
validates the corrected directed initialization rather than claiming a PSNR
gain.

Checkpoint-native audits at 9k/20k/30k find no large transparent interior hole
in fixed views 09/27 at 20k or 30k. At 30k, alpha below `0.5` after an 8-pixel
mesh-interior erosion is `0%` in view 09 and `0.00365%` in view 27, and is zero
in both after a 16-pixel erosion. However, concentrated low-length and
low-opacity root bands grow after the 20k geometry unlock; the largest final
low-length component covers `5.42%/6.89%` of visible roots in attribute views
09/32. Original-resolution user review rejects this as a visually valid groom:
the upper-back basin is a real noisy bald patch, not merely healthy short fur.
Primary-guide attribution finds `74 / 4500` guides below one quarter of their
stored length reference while those same guides average `9.37x` reference
width; guide log length and width ratios correlate at `-0.737`. Secondary
length/width residuals are too small to cause the defect. The run remains an
accepted completed execution and V7-direction artifact, but it fails visual
coverage acceptance. R069 guide-support-gauge work is a separate pending
training-method candidate; it does not reopen V7 flow. See
`docs/panda_r068_v7_training_20260828.md` and
`docs/r069_guide_support_gauge.md`.

## Active Entry Points

- training: `tools/train_white_tiger_stage1.py`
- frozen R036 metric-control configuration: `configs/stage1_baseline.env`
- active R043 behavior configuration:
  `configs/r043_density_matched_render_support_0_30k.env`
- active R043 lock: `configs/r043_density_matched_render_support.lock.json`
- active R049 geometry-parent configuration:
  `configs/r049_secondary_guide_resume16k_30k.env`
- active R050 appearance checkpoint configuration:
  `configs/r050_gaussian_rgb_residual_0_30k.env`
- latest R055 staged shape research configuration:
  `configs/r055_staged_primary_secondary_shape_0_30k.env`
- accepted R057 staged-shape/gradient-ownership parent:
  `configs/r057_rgb_flow_no_color_grad_0_30k.env`
- frozen R059 absolute-amplitude comparison configuration:
  `configs/r059_redesigned_groom_geometry_0_30k.env`
- accepted R060 advanced-geometry configuration:
  `configs/r060_relative_shape_amplitudes_0_30k.env`
- accepted R061 Gaussian-only appearance configuration:
  `configs/r061_gaussian_only_appearance_0_30k.env`
- accepted R062 mesh-validity configuration:
  `configs/r062_mesh_no_penetration_0_30k.env`
- accepted R065 local crossing-residual configuration:
  `configs/r065_local_crossing_residual_0_30k.env`
- accepted R066 learned-curl-turns configuration:
  `configs/r066_learned_curl_turns_0_30k.env`
- accepted R067 no-frizz configuration:
  `configs/r067_no_frizz_0_30k.env`
- accepted R068 no-crossing configuration:
  `configs/r068_no_crossing_zero_curl_0_30k.env`
- historical R038/R039 configurations remain evidence, not fallbacks
- server launcher: `scripts/server/run_white_tiger_stage1.sh`
- strand export: `tools/export_white_tiger_checkpoint_strands.py`
- Gaussian export: `tools/export_white_tiger_checkpoint_gaussians_ply.py`
- checkpoint rendering: `tools/render_white_tiger_stage1_checkpoint_views.py`
- groom diagnostics: `tools/visualize_white_tiger_groom_attributes.py`
- optional sparse flow annotation: `tools/groom_flow_annotator.py` and
  `anigroom/flow_annotations.py` (input acquisition only; not yet an active
  R068 training dependency)
- fixed-protocol strand audit: `tools/audit_strand_structure.py`
- mesh SDF construction: `anigroom/collision/mesh_sdf.py`
- differentiable SDF query/loss: `anigroom/collision/sdf.py`
- checkpoint penetration audit: `tools/diagnose_checkpoint_no_penetration.py`

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
- dimensionless curl-radius ratio, turns, and phase
- child radius and clump strength
- root and tip color
- root and tip opacity

The current R068 method baseline has no differentiable frizz-amplitude field or
persistent frizz seed. `frizz_backbone` is retained only as a standalone
procedural post-edit utility and is disconnected from reconstruction. The
historical R043-R066 frizz-bearing contracts remain evidence and parent
references, not current R068 state.

Brush stiffness controls one quadratic normal-to-groom transition while
preserving the root, tip, straight length, and 3D endpoint direction. The
effective value is brush stiffness multiplied by the continuous tangential
difference between the normal and endpoint direction. There is no second
interior deformation field. R058 directly replaces the optional curl/frizz
forward geometry with one local-3D-frame implementation: physical curl radius
and turns plus independent band-limited frizz. No legacy formula, normal-mode
switch, or checkpoint alias remains. R060 further decodes the two physical
amplitudes from current strand length, so equal controls produce equal
normalized geometry across short and long fur. Curl stays active in the current
R068 method baseline, while frizz is absent from its differentiable state. The
active route has one strand per render root
(`child_count=1`); density comes from independent render roots and their finite
lifecycle, not deterministic child expansion.

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

R050 retains smooth guide root/tip color and the existing local render-root
color term, then adds one view-independent RGB profile per render root. Each
generated Gaussian samples that profile at its normalized segment midpoint.
The profile is exactly inactive through 10k and ramps with the common schedule
to full strength at 20k. It has no TV or smoothness loss because it is the
explicit high-frequency appearance outlet; pure-fur asset export intentionally
omits it.

## Representation Lineage

The retired directional decomposition and old gravity/sag controls are no
longer present. R043 uses only length, normalized 3D endpoint direction, and
guide-owned brush stiffness for the ordinary base centerline. Its historical
frizz and clump controls remain evidence only for current R068; R068 retains
active curl and removes frizz from the differentiable reconstruction path.
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
