# R032: Density-Invariant Intrinsic Guide Smoothness

Status: accepted after the complete matched H100 9k-30k comparison. R032
replaces R030 as the Stage 1 continuation baseline and retains R031's
surface-consistent guide lifecycle with the corrected guide-length operator.

## General Failure Behind The Tail Artifact

The remaining distal-tail strands are not a tail-specific failure. At 16k,
the largest one percent of centered guide-length drift lies mostly on the
torso. The tail exposes the same drift more clearly because it is a thin
silhouette endpoint with little overlapping fur to hide a coherent length
error.

R031 increases guide roots from 5,332 to 11,875, but the accepted guide
smoothness is the unweighted mean of neighboring value differences. For the
same continuous field, shorter edges created by densification produce smaller
differences, so the loss becomes weaker as the representation becomes denser.

Measured at 16k:

| Quantity | R030, 5,332 guides | R031, 11,875 guides |
| --- | ---: | ---: |
| accepted relative edge loss | 0.004578 | 0.002936 |
| area-weighted intrinsic log-gradient | 13.758 | 22.264 |
| distal-tail intrinsic log-gradient | 28.211 | 69.807 |

The accepted loss reports a 36% improvement while physical surface roughness
is 62% worse globally and 147% worse at the distal tail.

The complete 30k control confirms that this is persistent rather than a brief
densification transient. R031 improves final test composite PSNR over R030 by
`0.15002 dB`, but its accepted guide loss falls from `0.003626` to `0.002454`
while global intrinsic roughness rises from `10.723` to `18.335` and distal-tail
roughness rises from `23.845` to `52.972`.

This does not imply that smoothness explains every tail error. A matched audit
of the original 4,500 guides found a second, independent conditioning effect.
Within the distal-tail diagnostic group, learned/target length drift correlates
with clean-flow multi-view count at about `-0.47` and with axis evidence
confidence at about `-0.57` at 16k. The same relationship is absent globally.
The tail is both a thin silhouette and weakly constrained in some views, so it
visually exposes geometry drift that overlapping body fur can hide.

R032 intentionally does not alter this uncertainty contract. It tests only
whether the guide smoothness remains physically equivalent after guide
densification. If R032 removes local spikes but leaves a coherent tail-wide
drift, the next candidate must use general multi-view observability to control
geometry freedom; it must not introduce a tail mask, body-part coefficient, or
absolute length rule.

## Single Method Change

R032 replaces only the guide-length term inside the existing guide smoothness:

1. represent positive length as `u = log(length)`;
2. divide neighboring differences by intrinsic mesh-surface distance;
3. average each source neighborhood;
4. integrate sources with a local surface-area proxy given by squared intrinsic
   spacing;
5. keep the initial guide spacing as a fixed reference scale after guide
   densification.

For source root `i`, intrinsic neighbors `N(i)`, surface distance `d_ij`, local
area proxy `A_i`, and initial reference spacing `h_0`:

```text
g_i^2 = mean_{j in N(i)} ((u_i - u_j) / d_ij)^2
L_length = (h_0^2 / 4) * sum_i A_i g_i^2 / sum_i A_i
```

The factor `h_0^2 / 4` preserves the scale of the accepted symmetric-relative
edge term at the initial guide sampling. Both `h_0` and `A_i` are derived from
the intrinsic source graph. There is no length bound, body label, tail mask,
percentile rule, selected-root budget, or animal-specific threshold.

All non-length guide terms, render-root losses, lifecycle evidence and
thresholds, schedules, initialization, and visualization protocols remain
unchanged.

## Pre-Training Verification

On the same analytic positive field sampled by 4x4 and 8x8 guide grids:

- accepted edge loss ratio, dense/coarse: `0.2049`;
- R032 intrinsic loss ratio, dense/coarse: `1.0148`.

The existing loss loses almost 80% of its strength after densification; R032
changes by 1.5%. The focused surface and checkpoint regression suites pass.

## Formal Result

The formal run started from the exact R030 9k checkpoint with optimizer state
reset, matching R031 in every option except
`GUIDE_LENGTH_SMOOTH_MODE=intrinsic_density_invariant`. It completed at 30k
without restart, fallback, OOM, NaN, or a post-20k lifecycle event.

| Quantity | R030 | R031 control | R032 |
| --- | ---: | ---: | ---: |
| final test composite PSNR | 32.33575 | 32.48577 | 32.47268 |
| best test composite PSNR | 32.53251 | 32.68041 | 32.66519 |
| final guide roots | 5,332 | 11,875 | 12,067 |
| final render roots | 318,963 | 318,898 | 318,928 |
| global intrinsic roughness | 10.723 | 18.335 | 7.833 |
| distal-tail intrinsic roughness | 23.845 | 52.972 | 14.596 |

Relative to R031, R032 reduces global intrinsic guide-length roughness by
`57.3%` and distal-tail roughness by `72.4%`, while changing final/best test
composite by only `-0.0131/-0.0152 dB`. Relative to the accepted R030 route,
R032 improves final/best test composite by `+0.1369/+0.1327 dB` and lowers
global/tail roughness by `27.0%/38.8%`.

This is not global coat shrinkage. The distal-tail guide median length changes
from `0.02606` in R030 and `0.02655` in R031 to `0.02713` in R032, while its
local jumps fall. The full length distribution remains broad and the fixed
side, opposite-side, and front asset views show no transferred short-hair or
curl-back artifact.

The stored reference spacing remains exactly `0.0315867625` from the initial
guide graph through the final 12,067-guide checkpoint. The final peak allocated
CUDA memory is `24272.92 MB`.

## Acceptance

- Exact-parent and single-variable checks passed.
- Fixed-reference-spacing and checkpoint-resume checks passed.
- Matched physical-roughness, composite-PSNR, lifecycle, memory, and canonical
  three-view asset checks passed.
- R032 is accepted because it corrects the guide-density discretization bug
  without global shortening, artifact transfer, or material reconstruction
  loss.

Formal artifacts:

- HGC output:
  `/home/wangyy/anigroom-r032-density-invariant-20260802/outputs/r032_density_invariant_guide_smoothness_9k_30k_20260802_h100`
- Local final checkpoint: `D:/RTS/_tmp/r032_30k_final/checkpoint_030000.pt`
- Checkpoint SHA-256:
  `5f142dac1f0ef725943182e329090114165ddcd76d187cace156b05e874997cf`
- Quantitative comparison:
  `D:/RTS/_tmp/r032_30k_final/guide_smoothing_comparison.json`
- Side asset:
  `D:/RTS/_tmp/r032_30k_final/r032_030000_asset_side_y_v11_protocol.png`
- Opposite-side asset:
  `D:/RTS/_tmp/r032_30k_final/r032_030000_asset_side_y_pos_v11_protocol.png`
- Front structural asset:
  `D:/RTS/_tmp/r032_30k_final/r032_030000_asset_front_z_v11_protocol.png`

## Remaining General Boundary

R032 fixes local guide-length discretization, not all uncertainty in the input
evidence. For the original guide roots in the distal-tail diagnostic group,
final learned/target length drift still correlates with multi-view count,
axis-view confidence, and axis-evidence confidence by approximately
`-0.395/-0.442/-0.444`; the same pattern is absent or weak globally. Any next
candidate must therefore test a general observability-conditioned geometry
freedom rule. It must remain independent of body labels, tail masks, absolute
lengths, percentiles, and animal-specific thresholds.
