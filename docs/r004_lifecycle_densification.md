# R004 Lifecycle Densification

## Goal

R004 tests one change on top of the R003 baseline: render-root densification no
longer inserts residual pixels directly as new roots. Instead, roots must earn
densification through accumulated root evidence, then pass a mesh-neighborhood
local-maximum filter before they are split.

This is the recovered lifecycle line. It is not an overlong, dark-stroke,
screen-footprint, luma, or animal-specific route.

## What Stays Identical To R003

- V4 clean-flow target and surface-direction interpolation.
- 100k render-root initialization.
- Guide roots from the clean-flow target.
- RGB-flow losses, smooth losses, random mesh backing, mesh-depth clipping, and
  evaluation metric.
- Phase A `0 -> 9000`, Phase B `9000 -> 30000`.
- Root statistics: gsplat visibility from radii, Gaussian mean/scale absolute
  gradients accumulated to root, root-position gradients kept separately, and
  optional residual evidence accumulated per root.

## Single Algorithmic Change

R003 inherited `DENSIFY_PARENT_SELECTION=target_direct`.

That route does this:

1. Compute an image residual.
2. Backproject high-residual pixels to the mesh.
3. Insert those points directly as child roots.
4. Keep the old parent roots.

R004 uses `DENSIFY_PARENT_SELECTION=evidence_local_max`.

That route does this:

1. Accumulate evidence over the same training window:
   `need = gaussian_grad + root_grad + residual`.
2. Apply absolute thresholds:
   `raw_visibility >= visibility_threshold`, `need >= grad_threshold`, and
   optional residual threshold.
3. Convert valid roots to their mesh faces.
4. For every valid face, compute the maximum valid root evidence on that face.
5. For each candidate root, compare its evidence against the maximum evidence
   in a topology-ring neighborhood around its face.
6. Keep only local maxima, then sort by evidence and apply the event budget.
7. Split each selected parent into child roots using the existing topology-local
   child placement.
8. Delete the selected parent, so this is a true root split rather than a direct
   pixel insertion.

In R003, `target_direct` consumes the residual image for direct mesh insertion,
so the lifecycle report shows zero root-level residual. In R004, the same
residual image is also projected back to roots and accumulated into
`RootStats.residual_sum`, making residual part of the parent-selection evidence.

## Exact Evidence Statistics

The lifecycle window accumulates statistics after each backward pass. It is
reset immediately after every structure update, so the next densification/prune
decision uses a fresh window.

Visibility is computed from gsplat `info["radii"]`: a Gaussian is visible when
its radius is positive. For the H100 gsplat layout where radii can have two
values per Gaussian, the implementation uses `amax(dim=1) > 0`. The visible
Gaussian counts are then scattered back to their owning render roots through
`gaussians.root_indices`.

The root-level evidence uses three terms:

```text
mean_grad      = abs(d loss / d gaussian_mean_xyz).sum_xyz
scale_grad     = abs(d loss / d gaussian_scale).sum_channels
gaussian_grad  = scatter_sum(mean_grad + 0.25 * scale_grad, root_id)
contribution   = scatter_sum(visible_gaussian * opacity, root_id)
root_grad      = abs(d loss / d root_xyz).sum_xyz
residual       = residual_per_root, if available
```

With the current default `score_mode=raw`, the normalized terms are:

```text
gaussian_grad_term = gaussian_grad_abs_sum / gaussian_contrib_sum
root_grad_term     = root_grad_abs_sum / visible_count
residual_term      = residual_sum / visible_count
need               = gaussian_grad_term + root_grad_term + residual_term
```

So the Gaussian term is not divided by plain visible count; it is divided by
opacity-weighted visible contribution. The separate `visible_count` is still
used as an absolute eligibility threshold, and it normalizes the direct root
gradient and residual terms.

## Budget Normalization

R003 `target_direct` inserts children and keeps parents. With
`MAX_SPLITS_PER_EVENT=512` and `SPLIT_CHILDREN_PER_PARENT=2`, it can add 1024
roots per event.

R004 deletes selected parents. To keep the maximum net growth comparable, the
R004 configs use `MAX_SPLITS_PER_EVENT=1024`. With two children per selected
parent, the maximum net growth is again 1024 roots per event.

## Expected Signal

R004 should reduce redundant growth where an entire high-residual surface patch
would otherwise be inserted at once. If it works, lifecycle logs should show
fewer adjacent parent clusters, and pure-fur visualization should have less
patchy overgrowth without losing coverage.

If R004 loses coverage or PSNR sharply, the first thing to inspect is whether
the local-max face neighborhood is too broad or the evidence threshold is too
strict for the current root density. Do not compensate with screen/luma/overlong
hardcoded routes in this experiment.

## 30k Result

Run:

```text
job: 124598861
commit: 30f0554d88e6aa60c80f3eaeb18fb3d991d06b86
output: /home/wangyy/anigroom-r004-localmax/outputs/r004_localmax_30f0554_20260721_051239
log: /home/wangyy/logs/r004_localmax_30f0554_20260721_051239/job.log
```

The job finished successfully on H100 with `exit_status=0`.

Metric comparison against the locked R003 refresh:

| Iteration | R003 test composite | R004 test composite |
| ---: | ---: | ---: |
| 9000 | 24.3561 | 24.0019 |
| 10000 | 29.1621 | 29.2422 |
| 12000 | 30.6575 | 30.6393 |
| 16000 | 31.7910 | 31.7875 |
| 20000 | 32.3734 | 32.3642 |
| 24000 | 32.5708 | 32.5649 |
| 29000 | 32.7361 | 32.7423 |
| 30000 | 32.5895 | 32.6023 |

Final R004 state:

```text
roots: 197280
generated Gaussians: 8780735
peak allocated CUDA memory: 15.5 GB
wallclock: 27301 s
```

Conclusion: R004 is an equivalent-quality lifecycle candidate, not a PSNR
improvement. It recovers from a slower Phase A and matches R003 after Phase B.
The topology-local split/delete rule does not damage the accepted baseline, but
its speed is worse in this run: recent normal training segments are around
`2.7-3.1 it/s`, while long progress gaps reduce end-to-end throughput. Before
making R004 the default, compare canonical pure-fur visualizations and identify
whether the long gaps come from lifecycle selection, diagnostic output, or HGC
node/runtime jitter.

## Files

- `anigroom/roots/lifecycle.py`
- `tools/train_white_tiger_stage1.py`
- `tools/test_root_lifecycle_local_max.py`
- `configs/r004_evidence_localmax_0_9k.env`
- `configs/r004_evidence_localmax_9k_30k.env`
- `scripts/server/run_r004_from_zero.sh`
