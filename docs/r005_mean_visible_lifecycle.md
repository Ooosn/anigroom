# R005 Mean-Visible Lifecycle Densification

## Goal

R005 keeps the R004 topology-local split/delete lifecycle, but replaces the
parent-selection evidence with a cleaner geometric signal:

```text
need = scatter_sum(abs(d loss / d gaussian_mean_xyz).sum_xyz, root_id)
       / root_visible_gaussian_count
```

This is the direct version of the densification rule we originally wanted:
roots split when the Gaussians that belong to them repeatedly receive high
geometry pressure. It removes two ambiguities in R004 parent selection:

- opacity-weighted contribution is no longer used as the Gaussian-gradient
  denominator;
- residual/root-position terms are not mixed into the default parent evidence.

## Relationship To R004

R004:

```text
parent selection: evidence_local_max
score mode:       raw
need:             gaussian_grad / opacity_weighted_contribution
                  + root_grad / visible_count
                  + residual / visible_count
split rule:       topology-local split/delete
```

R005:

```text
parent selection: evidence_local_max
score mode:       mean_visible
need:             gaussian_mean_xyz_grad / visible_count
split rule:       topology-local split/delete
```

Everything else should remain inherited from R004/R003 unless explicitly logged
as a later experiment.

## Exact Definitions

The statistics are accumulated after each backward pass and reset immediately
after every structure update.

Visibility is still formal gsplat visibility:

```text
visible_gaussian = info["radii"] > 0
```

For H100 gsplat layouts with two radii values per Gaussian, visibility uses:

```text
visible_gaussian = info["radii"].reshape(N, -1).amax(dim=1) > 0
```

The R005 term is:

```text
mean_grad_per_gaussian = abs(d loss / d gaussian_mean_xyz).sum_xyz
gaussian_mean_grad_abs_sum[root] =
    scatter_sum(mean_grad_per_gaussian, gaussian.root_id)

visible_count[root] =
    scatter_sum(visible_gaussian, gaussian.root_id)

need[root] =
    gaussian_mean_grad_abs_sum[root] / max(visible_count[root], 1)
```

`root_grad_abs_sum`, `gaussian_contrib_sum`, and `residual_sum` are still
recorded for diagnostics and pruning, but they do not enter R005 parent
selection.

## Why This Matters

R003 can grow roots directly from residual pixels. That is effective but less
like a true root lifecycle. R004 makes lifecycle more principled, but its
evidence mixes geometry pressure, root-position pressure, opacity contribution,
and residual pressure. R005 isolates the root split decision to the most direct
geometry signal: visible Gaussian mean gradients.

If R005 is worse than R004, the likely conclusion is that residual/root terms
are actually useful for coverage. If R005 matches R004, it should become the
cleaner densification baseline because the rule is easier to explain and less
coupled to RGB residual behavior.

## Files

- `anigroom/roots/statistics.py`
- `anigroom/roots/lifecycle.py`
- `tools/train_white_tiger_stage1.py`
- `tools/test_root_lifecycle_mean_visible.py`
- `configs/r005_mean_visible_localmax_0_9k.env`
- `configs/r005_mean_visible_localmax_9k_30k.env`
- `scripts/server/run_r005_from_zero.sh`

## Status

Prepared. No 30k result yet. The R005 configuration now explicitly sets
`MAX_SPLITS_PER_EVENT=0`; in lifecycle selection this means no event budget.
A positive value is allowed only as a separate safety experiment and must be
named as such.

## Local Threshold Sanity Check

Earlier short local run:

```text
output: D:/petsgaussianhair-accept-line/outputs/local_r005_threshold_1200_20260721
iterations: 1200
GPU: RTX 4080 SUPER
score mode: mean_visible
threshold: 2.5e-5
```

Lifecycle summary:

| Iteration | Roots Before | Selected | Inserted | Pruned | Roots After | Global Need P50 | Global Need P95 | Selected Need P50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 600 | 100000 | 1024 | 2048 | 1024 | 101024 | 2.50e-5 | 6.07e-5 | 8.45e-5 |
| 700 | 101024 | 1024 | 2048 | 1024 | 102048 | 2.81e-5 | 7.65e-5 | 1.06e-4 |
| 800 | 102048 | 1024 | 2048 | 1024 | 103072 | 2.86e-5 | 7.59e-5 | 1.02e-4 |
| 900 | 103072 | 1024 | 2048 | 1024 | 104096 | 2.82e-5 | 7.51e-5 | 1.00e-4 |
| 1000 | 104096 | 1024 | 2048 | 1024 | 105120 | 2.69e-5 | 7.34e-5 | 9.75e-5 |
| 1100 | 105120 | 1024 | 2048 | 1024 | 106144 | 2.78e-5 | 7.01e-5 | 9.33e-5 |
| 1200 | 106144 | 1024 | 2048 | 1024 | 107168 | 2.80e-5 | 7.11e-5 | 9.41e-5 |

This run is no longer a valid R005 behavior check because it silently inherited
R004's `MAX_SPLITS_PER_EVENT=1024`. It is kept only as evidence that the old cap
hid the true candidate count.

## No-Cap Sanity Check

Short local runs after removing the event budget:

```text
output: D:/petsgaussianhair-accept-line/outputs/local_r005_nocap_1200_20260721
iterations: 1200
GPU: RTX 4080 SUPER
score mode: mean_visible
threshold: 2.5e-5
parent budget: none
```

Lifecycle summary:

| Iteration | Roots Before | Threshold Candidates | Local-Max Parents | Inserted | Pruned | Roots After |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 600 | 100000 | 49968 | 4728 | 9456 | 4728 | 104728 |
| 700 | 104728 | 58737 | 5176 | 10352 | 5176 | 109904 |
| 800 | 109904 | 60203 | 5113 | 10226 | 5113 | 115017 |
| 900 | 115017 | 58895 | 4932 | 9864 | 4932 | 119949 |
| 1000 | 119949 | 56217 | 4879 | 9758 | 4879 | 124828 |
| 1100 | 124828 | 58252 | 4898 | 9796 | 4898 | 129726 |
| 1200 | 129726 | 57951 | 4895 | 9790 | 4895 | 134621 |

Conclusion: R005 now exposes the real lifecycle behavior. The inherited
`2.5e-5` threshold admits roughly half the roots before local-max suppression.
Topology-local maxima reduce this to about 4.9k parent splits per event, so the
growth is not an immediate explosion, but it is much more aggressive than the
old hidden +1,024/event cap. The next full or medium run must monitor the same
`threshold_candidate_count` and `local_max_candidate_count` fields past 9k. If
growth is unstable, the fix is a named threshold/evidence ablation, not
reintroducing a silent event budget.

## Spatial Distribution Check

The 1200-step diagnostic was repeated with selected-parent spatial histograms.
The selection is compared against the distribution of all active render roots
using 10 bins per world axis.

| Iteration | Threshold Fraction | Local-Max Fraction | Selected Parents | X L1 From All | Y L1 From All | Z L1 From All |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 600 | 0.500 | 0.047 | 4728 | 0.051 | 0.136 | 0.151 |
| 700 | 0.561 | 0.049 | 5175 | 0.056 | 0.111 | 0.131 |
| 800 | 0.547 | 0.047 | 5122 | 0.075 | 0.112 | 0.139 |
| 900 | 0.513 | 0.043 | 4934 | 0.051 | 0.119 | 0.153 |
| 1000 | 0.468 | 0.040 | 4842 | 0.059 | 0.110 | 0.143 |
| 1100 | 0.466 | 0.039 | 4883 | 0.056 | 0.106 | 0.124 |
| 1200 | 0.447 | 0.038 | 4906 | 0.071 | 0.112 | 0.144 |

Adjacent selected-parent distribution deltas are also small:

| Event Pair | X L1 Delta | Y L1 Delta | Z L1 Delta |
| --- | ---: | ---: | ---: |
| 600->700 | 0.043 | 0.041 | 0.045 |
| 700->800 | 0.024 | 0.031 | 0.046 |
| 800->900 | 0.028 | 0.024 | 0.041 |
| 900->1000 | 0.035 | 0.029 | 0.026 |
| 1000->1100 | 0.022 | 0.020 | 0.063 |
| 1100->1200 | 0.037 | 0.021 | 0.053 |

Interpretation: current R005 is not exploding in count, but the evidence is too
broad. Because the threshold accepts nearly half the active roots, local-max
selection behaves close to regular surface sampling: every event inserts a
similar number of roots with a similar whole-body spatial distribution. This is
not yet the intended targeted root densification.
