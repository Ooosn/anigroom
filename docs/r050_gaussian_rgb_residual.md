# R050 Gaussian-Level RGB Residual

## Status

Completed and frozen as the appearance diagnostic parent. Its numerical gain
and structural result remain valid, but a later same-checkpoint layer ablation
rejects the claim that it achieved a clean low-frequency/high-frequency color
decomposition. R049 remains the matched structural control and R043 remains the
independent-render-root RGB metric control.

## Method Question

R049 proves that a 20k secondary geometry field removes most local length
patchiness and geometric texture fitting, but it remains about 1.1 dB below
R043. R050 tests whether the missing high-frequency RGB evidence can be routed
through appearance rather than returned to length or direction.

## Single Variable

Keep R049's geometry, 20k secondary guides, K8 render interpolation, K4
secondary smoothness, lifecycle, losses, optimizer rates, random mesh backing,
resolution, and schedules unchanged. Add one appearance field:

```text
base Gaussian RGB
  = interpolated root/tip color
  + existing local render-root color

final Gaussian RGB
  = clamp(base Gaussian RGB
          + scheduled_scale * Gaussian RGB profile(sample position), 0, 1)
```

Each render root owns a 36-control RGB profile over normalized strand arc
length. Every generated Gaussian samples the profile at its own segment
midpoint. This is per-Gaussian under the active discretization while remaining
well-defined when adaptive segment counts change. It is not a constant
per-strand color and is not a dynamically resized Parameter tied to a transient
flattened Gaussian list.

The decoded residual is `tanh(raw) * 0.20 * multiplier`. The multiplier is an
exact zero through 10k and ramps to one at 20k using the same schedule function
as the existing handoff controls. When it is zero, residual evaluation is
skipped, so no gradient or Adam state is allocated before activation.

New render roots created during lifecycle receive a zero RGB profile. They do
not inherit the parent's image noise. Surviving rows and their Adam state are
migrated exactly. The candidate fails fast unless `child_count=1`.

No L2, TV, or strand smoothness is applied to this residual: its purpose is to
store the high-frequency appearance that the smooth groom should not absorb.
It is view-independent RGB in this first controlled experiment; SH or a
decoder would introduce a second variable.

## Acceptance Gate

1. Zero profile and zero multiplier are bit-exact no-ops.
2. Gradients reach only the two arc-length controls supporting each Gaussian.
3. Adaptive segment counts preserve normalized-position semantics.
4. Lifecycle keeps surviving rows/state and initializes new rows to zero.
5. Old support-off checkpoints still load strictly; R050 checkpoints roundtrip.
6. One-H100 30k training completes without fallback, OOM, or topology change.
7. Compare full-resolution composite PSNR and per-view diffs with R049/R043.
8. Repeat the fixed 100k-strand structural QA. RGB recovery is rejected if it
   restores R043-style length discontinuity, long strands, or curl-back.

## Result

The one-H100 from-zero run completed without fallback, OOM, topology drift, or
runtime error:

```text
source:     /home/wangyy/anigroom-r050-gaussian-rgb-residual-20260808
runtime:    /home/wangyy/anigroom-r050-gaussian-rgb-residual-runtime-20260808
output:     /home/wangyy/anigroom-r050-gaussian-rgb-residual-runtime-20260808/outputs/r050_gaussian_rgb_residual_0_30k_h100_20260808
checkpoint: /home/wangyy/anigroom-r050-gaussian-rgb-residual-runtime-20260808/outputs/r050_gaussian_rgb_residual_0_30k_h100_20260808/checkpoint_030000.pt
sha256:     21f5ff68461c69cca1b182a01cbd0a6dbd01b09c29128f77e82150eb7cc535ee
```

| Measurement | R044 full-run control | R049 structural control | R050 |
| --- | ---: | ---: | ---: |
| final train composite PSNR | 32.234222 | 32.262772 | 33.253601 |
| final test composite PSNR | 31.574749 | 31.603195 | 32.121105 |
| best test composite PSNR | 31.712448 | 31.741003 | 32.209358 |
| best iteration | 29k | 29k | 29k |
| render roots | 469,402 | 469,402 | 469,757 |
| generated Gaussians | 5,319,491 | 5,323,659 | 5,358,403 |
| peak allocated CUDA memory | 10,699.64 MB | 10,319.53 MB | 11,327.52 MB |
| from-zero elapsed time | 9,133.44 s | continuation | 9,243.02 s |

R050 improves final/best test composite over R049 by `+0.517910/+0.468355`
dB. It recovers more than half of the final metric gap from R049 to R043 while
remaining `0.390485` dB below R043's final metric. The extra profile and Adam
state add about 0.6 GiB over the matched full-run control without introducing
the historical retained-graph memory failure.

### Same-checkpoint appearance ablation

The fixed eight-view renderer loads the final checkpoint once, renders it with
the residual enabled, then sets only its runtime multiplier to exactly zero and
renders again. This isolates the direct appearance contribution from geometry,
topology, camera, and evaluation differences.

| View | Full PSNR | Residual-off PSNR | Direct gain |
| ---: | ---: | ---: | ---: |
| 00 | 32.494854 | 31.612377 | +0.882477 |
| 05 | 32.606186 | 30.215857 | +2.390329 |
| 09 | 32.539284 | 29.819038 | +2.720245 |
| 14 | 33.637882 | 31.487896 | +2.149986 |
| 18 | 33.130547 | 32.175415 | +0.955132 |
| 21 | 34.262756 | 31.890663 | +2.372093 |
| 27 | 33.060146 | 30.333593 | +2.726553 |
| 32 | 33.798496 | 31.086613 | +2.711884 |
| mean | 33.191269 | 31.077682 | +2.113587 |

The final decoded residual has absolute mean `0.061171`, RMS `0.092599`, and
`5.480%` near-saturated values. The residual images are confined to the fur
support and restore stripe contrast and fine fur/shadow variation; they do not
change alpha, roots, or exported asset geometry.

### Four-layer decomposition audit

The two-layer ablation above proves that the Gaussian residual matters, but it
does not prove that it alone stores noise. A second same-checkpoint audit also
disables the existing local render-root color and the complete local color
stack:

| Variant | Mean eight-view PSNR | View09 PSNR |
| --- | ---: | ---: |
| full | 33.1913 | 32.5393 |
| Gaussian residual off | 31.0777 | 29.8190 |
| local render color off | 28.3788 | 27.1601 |
| root/tip only | 27.1539 | 25.8152 |

The decoded local render color has absolute mean `0.12466`, RMS `0.12960`, and
absolute maximum `0.14988` under a `0.15` scale. It is not a small smooth
correction. Frequency-separated error gives total/low/high RMSE of
`0.04879/0.03002/0.03238` for the full model,
`0.06476/0.04350/0.03776` with Gaussian residual off, and
`0.09268/0.07799/0.03761` with local render color off.

The user-visible result is consistent with these numbers: removing the
Gaussian residual increases noise. The correct conclusion is that it performs
useful cancellation, while the base and local color fields themselves also
contain image noise. R050's layers are entangled.

### Fixed structural QA

R043, R049, and R050 use the same deterministic 100k-strand, 32-sample,
child-one export and the same three Blender cameras.

| Statistic | R043 | R049 | R050 |
| --- | ---: | ---: | ---: |
| local 4NN relative length difference mean | 0.103637 | 0.021087 | 0.020467 |
| local 4NN relative length difference P95 | 0.315815 | 0.079554 | 0.077408 |
| local chord-direction difference P95 | 11.3489 deg | 11.3911 deg | 11.2959 deg |
| arc/chord ratio P95 | 1.005887 | 1.003271 | 1.006726 |
| maximum local turn P95 | 0.869957 deg | 0.627962 deg | 0.954593 deg |
| maximum local turn maximum | 2.733589 deg | 2.062634 deg | 3.187551 deg |
| maximum arc length | 0.131546 | 0.119080 | 0.128519 |
| strands longer than 0.12 | 6 | 0 | 5 |
| strands with a backward segment | 0 | 0 | 0 |

R050 preserves and slightly improves the secondary field's local length and
direction continuity. Its five sampled lengths above `0.12` are part of one
coherent head/cheek long-hair patch already present in R049's upper tail, not
distributed isolated spikes. The three canonical views contain no body spike,
crossing cluster, foldback, or curl-back.

Local fixed-protocol outputs:

```text
D:\RTS\_tmp\r050_30k_final\rgb_views\render_report.json
D:\RTS\_tmp\r050_30k_final\r050_030000_asset_side_y_v11_protocol.png
D:\RTS\_tmp\r050_30k_final\r050_030000_asset_side_y_pos_v11_protocol.png
D:\RTS\_tmp\r050_30k_final\r050_030000_asset_front_z_v11_protocol.png
D:\RTS\_tmp\r050_30k_final\r043_r049_r050_strand_audit.json
```

## Decision

Keep the existing R050 tag and checkpoint as immutable diagnostic evidence. It
validates the Gaussian RGB residual mechanism and preserves the secondary
geometry field, but it does not validate the intended color separation. Do not
promote its layer ownership as the method. R051 tests the corrected contract:
sparse guide-owned main color, no competing render-root local color, and the
Gaussian RGB residual as the sole high-frequency outlet.

Implementation verification also completed:

- `109 passed` in the full test suite;
- Python compilation and launcher syntax passed;
- launcher dry-run resolves the R049 geometry contract with `child_count=1`,
  secondary-guide K4, and the exact 10k-to-20k Gaussian RGB ramp;
- pre-R050 support-off checkpoint configuration defaults remain strict and
  residual-free;
- residual statistics are computed exactly in root chunks, avoiding a full
  decoded-profile temporary allocation during evaluation.

Before the formal run, the explicitly non-reportable
`configs/r050_gaussian_rgb_residual_fullres_preflight.env` executes one real
1920x1080 view09 optimizer step with the full 400k-root model and residual
multiplier one. Its sole purpose is to verify CUDA forward/backward, nonzero
profile gradients/state, checkpoint reload, and peak memory before committing
the held H100 to 30k.
