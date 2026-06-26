# Stage 1 Audit 2026-06-26

This audit freezes the current understanding of the white-tiger Stage 1 state.
It is not a training plan and it is not a new method proposal.

## Immediate Conclusion

The repository was not in a trustworthy training state before this cleanup.

The main Stage 1 implementation is mostly untracked by git, several old tracked
files are deleted but not committed, and server/local experiment directories mix
valid baselines, diagnostics, interrupted runs, and partial image pulls.

The local code has now been restored to the `20260626015952` single-layer Stage
1 snapshot, which is the best recovery baseline found so far.

Do not start more training until this restored baseline is committed and pushed.

## Dirty Working Tree

Tracked files currently modified or deleted:

- `.gitignore`
- `README.md`
- `scripts/verify_white_tiger_server_env.sh`
- deleted old UV/asset scripts and old `anigroom/mesh.py`, `preview.py`, `stage_a.py`

Formal Stage 1 files are currently untracked:

- `tools/train_white_tiger_stage1.py`
- `configs/white_tiger_stage1_formal.env`
- `scripts/server/run_white_tiger_stage1.sh`
- `anigroom/grooming/`
- `anigroom/roots/`
- `anigroom/projection/`
- `anigroom/data/`
- `anigroom/evaluation/`

This means `git diff` cannot reliably explain the current Stage 1 behavior.

## Core File Versions

Three code/config snapshots are different:

| Snapshot | Meaning | Status |
|---|---|---|
| `server_pull/anigroom_code_20260626015952.clean.tar.gz` | Good single-view run family | Keep as recovery baseline |
| `server_pull/anigroom_code_20260626031918.white.tar.gz` | Cleaned-flow / anchor experiments | Diagnostic only |
| `_handoff/server_code_sync.tgz` and current files | Later handoff/current line | Suspect until revalidated |

Hash comparison showed all three snapshots differ for:

- `tools/train_white_tiger_stage1.py`
- `configs/white_tiger_stage1_formal.env`
- `scripts/server/run_white_tiger_stage1.sh`

## Restored Formal Config

Current `configs/white_tiger_stage1_formal.env` has been restored from
`server_pull/anigroom_code_20260626015952.clean.tar.gz`:

```text
ROOT_COUNT=20000
CHILD_COUNT=8
SAMPLES=48
MIN_SEGMENTS=6
MAX_SEGMENTS=22
MESH_DEPTH_CLIPPING=1
MESH_BACKING_COMPOSITING=1
DENSIFY_WARMUP=500
DENSIFY_INTERVAL=100
DENSIFY_UNTIL=12000
PRUNE_START=1200
PRUNE_INTERVAL=100
```

There is no `ORIENTATION_TARGET_MODE` switch in the restored formal config. The
formal path uses the default raw orientation map loading and confidence weighting
from the restored training script.

The formal config is not using a 1000-step densification interval. The 1000-step
number is from evaluation/logging or stale run naming, not the current formal
config.

## Orientation / Flow State

The previous flow/orientation state was mixed:

- The good `20260626015952` baseline snapshot did not have the new
  `ORIENTATION_TARGET_MODE` switch.
- The `20260626031918` experiments added `ORIENTATION_TARGET_MODE=cleaned`.
- The later current-line config had set `ORIENTATION_TARGET_MODE=raw`.

The later current-line code supported both:

```text
raw:
  target = original orientation map
  confidence = orientation confidence * foreground mask * mask-edge confidence

cleaned:
  target = local double-angle trend
  confidence = confidence * coherence * agreement * hard thresholds
```

This is one source of confusion. The exact orientation map used by each result
must be read from that run's config/metrics, not from memory.

Cleanup action:

- removed cleaned-flow / anchor / multilevel flow diagnostic entry scripts from
  `tools/`;
- restored the formal config and server run script to the default orientation
  path;
- kept only `tools/analyze_white_tiger_projection.py` as the mesh/camera
  projection diagnostic.

## Root / Multi-Level Root State

Current formal `tools/train_white_tiger_stage1.py` is still a single root field
with deterministic child strands:

```text
root -> groom parameters -> guide strand -> child strands -> Gaussians
```

The later guide/render multi-level root idea is not cleanly integrated into the
formal Stage 1 training path.

Therefore the drop after the "multi-level root" discussion should not be
interpreted as proof that multi-level root is bad. It means the transition was
not isolated or verified.

Cleanup action:

- multi-level root is not part of the current formal code path;
- the current path is single-layer root plus differentiable child strands;
- future guide/render root work must be introduced as a separate branch/module
  after the restored baseline is reproduced.

Child strand differentiability:

- `child_count` is a discrete structural setting, not differentiable;
- `child_radius` and `clump_strength` are continuous groom parameters and are
  differentiable;
- root/tip width control strand thickness, not child count;
- child strand positions, widths, colors, and opacities are generated in PyTorch
  before gsplat, so gsplat gradients propagate back to the groom parameters.

## Densification / Pruning State

Current root lifecycle code uses real gsplat evidence:

- `info["radii"] > 0` for Gaussian visibility.
- Gaussian mean/scale gradients accumulated by root id.
- Root barycentric gradient accumulated separately.
- Need score is not percentile-normalized:

```text
gaussian_grad = gaussian_grad_abs_sum / gaussian_contrib_sum
root_grad = root_grad_abs_sum / visible_count
need = gaussian_grad + root_grad + residual
```

Split placement is topology-local:

```text
selected parent root
-> nearby mesh face candidates
-> locally empty candidates by distance to existing roots
-> child attributes interpolated from parent + neighboring roots
-> parent can be pruned/replaced
```

This matches the intended direction better than the earlier smoke-test
densification, but it is still only a single-root lifecycle. It has not been
revalidated after the later flow and range changes.

## Mesh Depth / Backing State

Current code clips per Gaussian outside gsplat before rasterization:

```text
gaussians -> mesh_depth_clip_gaussians -> gsplat rasterization
```

It is not doing root-level clipping in `render_view`.

Current backing composition:

```text
fur image from gsplat with black raster background
image = fur + (1 - alpha) * backing_image
```

`backing_image` is built from mesh depth and a random mesh/body color during
training. This is still a Python-side approximation, not the final ideal CUDA
depth-aware backing implementation.

## Valid / Useful Results

Best useful single-view results found in local metrics:

| Run | Iter | PSNR | Composite PSNR | Notes |
|---|---:|---:|---:|---|
| `white_tiger_single_view09_rootclip_child4_reconweights_3000_20260625` | 3000 | 32.475 | n/a | old stronger single-view result |
| `white_tiger_single_view09_rootclip_child4_rawmetric_resume5500_lowlr_fix_20260625` | 5500 | 32.055 | 33.152 | old stronger single-view result |
| `white_tiger_single_view09_rootclip_child4_rawmetric_5000_20260625` | 5000 | 31.924 | 32.981 | old stronger single-view result |
| `white_tiger_single_view09_exactdepth_resume6000_20260625` | 6000 | 31.890 | 32.942 | exact-depth line |
| `server_20260626015952_view09_rgb_iter3000` | 3000 | 31.558 | 32.689 | current best server-side recovery baseline |

The later drop to 25/26 is not normal and should be treated as a bug or an
uncontrolled code/config change.

## Invalid / Diagnostic Runs

These were diagnostic or invalid and were removed from local `_downloads` /
`server_pull` after recording this audit:

- `server_202606261320_rgb_alpha_decouple_iter3000`
- `server_202606261250_fixed_calib_3000_iter3000`
- `server_202606261235_width_guard_*`
- `server_20260626121532_iter*`
- `202606261350_albedo_geo`
- `202606261410_re_eval_old5500_current_code`
- `202606261455_restore_verified_objective`
- `202606261505_restore_verified_direct3000`
- `202606261520_restore_ranges_direct3000`

Reasons:

- partial local pulls without `metrics.jsonl`;
- interrupted runs;
- configs changed during debugging;
- not isolated against the known good baseline.

Removed local diagnostic families:

- `white_tiger_multiview_cleaned_flow_*`
- `white_tiger_flow_cleaning_*`
- `white_tiger_anchor_field_*`
- `white_tiger_multilevel_projected_init_*`
- `synthetic_orientation_convention_*`
- polluted package `server_pull/anigroom_code_20260626031301.clean.tar.gz`

## Today's Actual Pollution

Main code changes made after the good baseline, then removed from the current
formal path, include:

- added raw/cleaned orientation target switch;
- added local orientation trend cleaning and thresholds;
- added RGB geometry-gradient routing;
- added albedo/geometry decoupling loss path;
- changed groom range defaults around `root_width`, `tip_width_ratio`, and
  `tip_opacity_ratio`;
- changed server script to require/pass the new parameters;
- created multiple server/local diagnostic runs without a clean run manifest.

These changes may be useful individually, but none should be assumed safe until
isolated against `server_20260626015952_view09_rgb_iter3000`.

The restored code path intentionally does not include the cleaned-flow switch,
multi-level root, RGB geometry-gradient routing, or albedo/geometry decoupling.

## Recovery Order

1. Freeze current directory. No training.
2. Make one git-tracked baseline commit containing only the formal Stage 1
   files needed to reproduce that baseline.
3. Push the baseline commit.
4. Server must `git pull` this commit before any training.
5. Re-run one deterministic single-view view09 recovery check.
6. Only then test changes one at a time:
   - orientation raw vs cleaned;
   - groom range patch;
   - RGB geometry-gradient routing;
   - albedo/geometry decoupling;
   - multi-level guide/render root.
7. Multi-level root must be integrated as an isolated module:
   - guide/anchor roots own stable flow/groom trend;
   - render roots interpolate from guide roots and provide dense visible fur;
   - color ownership must be explicit;
   - orientation loss target must be explicit;
   - densification must state whether it updates guide roots, render roots, or
     both.

Until this recovery order is done, more training is likely to produce misleading
results.
