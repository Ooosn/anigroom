# R038 Brush Curve And Finite Render Lifecycle

Status: accepted as a historical structural/lifecycle Stage 1 baseline after a
complete strict-schema 0-30k H100 run. R036 remains frozen as the higher-PSNR
metric control.

## Question

Can the model represent ordinary brushed fur without using legacy lateral bend
as the base shape, while ending render-root densification cleanly at 9k instead
of paying lifecycle cost through the late appearance/shape stages?

R038 was trained from zero and did not load an R036 checkpoint. Its formal
source is frozen by tag `stage1-r038` and
`configs/r038_brush_curve.lock.json`; the original training process ran from
source commit `a4e1e581d0c35a5f1c8391d94205266101949c24`.

## Representation Change

For root `P0`, outward normal `n`, normalized learned 3D endpoint direction
`d`, straight length `L`, and guide-owned brush strength `c in [0,1]`:

```text
delta = L d
delta_n = dot(delta, n) n
delta_t = delta - delta_n
F_n(s,c) = s + c s(1-s)
F_t(s,c) = s - c s(1-s)
B(s) = P0 + F_n delta_n + F_t delta_t
```

`c=0` is the exact straight root-to-tip segment. Increasing `c` accumulates
normal displacement earlier and tangent displacement later. Root and tip are
unchanged, so length remains the straight endpoint distance and 3D direction
retains its existing meaning.

`brush_curve_strength` belongs only to guide roots. Render roots receive the
intrinsically interpolated guide value; no render-root brush residual exists.
It uses the same guide graph smoothing and optimizer lifecycle as the other
guide-owned base fields.

The optional bend is now an unbounded signed, non-periodic interior offset:

```text
w(s) = 16 s^2 (1-s)^2
P(s) = B(s) + L bend w(s) side
```

The envelope and its first derivative vanish at both endpoints. Bend therefore
does not change the root, tip, or endpoint tangents and cannot duplicate the
endpoint direction. The legacy `tanh/atanh` bend interval has been removed from
guide controls, render residuals, lifecycle interpolation, and priors.

Curl and frizz stay disabled in R038. They are not silently replaced or tuned.

## Gaussian Allocation

Strands are constructed first. Existing adaptive allocation then measures the
final arc and turning complexity, so a genuinely curved brush strand receives
more samples than the same straight endpoint segment. No maximum segment count
or animal-specific length threshold is introduced.

## Lifecycle Change

R038 uses one render-root lifecycle only:

- warmup starts at iteration 600;
- one evidence-driven event every 100 iterations;
- the final event is iteration 9000;
- guide-root densification is disabled for this candidate;
- pruning remains disabled.

Iteration 9000 still retains per-Gaussian/root gradients, adds the final
window, and applies the event. From iteration 9001 onward the trainer no longer
retains lifecycle-only gradients, builds residual evidence, or accumulates
root visibility/evidence statistics. This is a code-path stop, not only a
schedule value that leaves hidden work active.

The trainer emits `lifecycle_statistics_state` at the transition and records
`lifecycle_statistics_active` in evaluation metrics, so the 9000/9001 boundary
is verified from the formal run rather than inferred from the config.

The score threshold, visibility/contribution evidence, local-max parent
selection, surface child placement, attribute interpolation, and optimizer
state migration are unchanged from the accepted implementation. This isolates
the representation and lifecycle horizon rather than retuning density to one
animal.

## Candidate Configuration

`configs/r038_brush_curve_0_30k.env` differs from frozen R036 only in:

- `DENSIFY_UNTIL: 20000 -> 9000`;
- guide densification is disabled;
- iteration 9000 is added to stage checkpoints;
- candidate comments identify the strict from-zero route.

All image resolution, root count, child count, evidence thresholds, losses,
learning rates, renderer settings, and memory guard values remain unchanged.

## Local Verification

- full repository tests: `73 passed`;
- R036 lock is verified against Git tag `stage1-r036`, not the mutable candidate
  worktree;
- tests cover exact straight behavior, fixed endpoints, gradient flow,
  normal-first/tangent-later motion, smooth interior bend, adaptive sampling,
  guide ownership, strict lifecycle migration, and the 9000/9001 boundary;
- Python compilation and launcher/config preflight must pass before launch.

## Formal H100 Run

The strict from-zero run used the frozen R036 data, camera, loss, resolution,
and evaluation contract. It started on one H100 at
`2026-08-05T22:16:07+09:00` and completed normally at
`2026-08-06T01:55:15+09:00` with exit code zero. The HGC checkout passed all
73 tests before model construction.

Final measured evidence:

| Measurement | R038 | Frozen R036 control |
| --- | ---: | ---: |
| Final train composite PSNR | `33.03637` | `33.42397` |
| Final test composite PSNR | `32.34588` | `32.66322` |
| Best test composite PSNR | `32.51677` at 29k | `32.83977` at 29k |
| Render roots | `209220` | `317245` |
| Generated Gaussians | `9496145` | `14215421` |
| Elapsed training time | `12345.53 s` | `21718.13 s` |
| Peak allocated CUDA memory | `16767.55 MB` | about `24.10 GiB` |

R038 is `0.31734 dB` below R036 at the final test metric. In exchange it uses
`34.0%` fewer render roots and `33.2%` fewer Gaussians, and finishes `43.2%`
faster. This is not recorded as a PSNR improvement; it is a representation and
finite-lifecycle result.

The final checkpoint is `120369618` bytes with SHA-256
`994578210640f7e586f3c2cbdfb0eced6b962680945ea8ca1b82c6444e1cdf41`.
The saved metrics SHA-256 is
`37aa24d9a619e690a30c6e7bb5ea019728f5d893173c02afdc50451da77adc6d`.

## Lifecycle Audit

The measured lifecycle boundary and event behavior are correct:

- 85 events run from iteration 600 through 9000 inclusive;
- iteration 9000 selects 917 parents, inserts 1,834 children, and leaves
  209,220 render roots;
- iteration 9001 emits `lifecycle_statistics_active=false`;
- no later event changes root count or accumulates lifecycle evidence;
- every event reports `parent_budget=-1` and `budget_saturated=0`;
- selected parents vary across 83 distinct counts, with
  min/median/mean/max `702 / 1304 / 1284.94 / 1736`;
- inserted children vary with the evidence, with min/median/mean/max
  `1404 / 2608 / 2569.88 / 3472`.

The population is therefore not produced by a hidden fixed quota, and no
lifecycle-only gradient/statistics path remains active after the last event.

## Structural QA

The fixed V11 asset protocol uses child expansion, deterministic 100k-strand
sampling, 32 curve samples, the same three cameras, 1920x1080 Cycles rendering,
and the same mesh/material settings as R036. Both the exact post-lifecycle 9k
checkpoint and the final 30k checkpoint were exported with this protocol.

The final 100k-strand numeric audit reports:

- backward segment fraction: `0`;
- strands containing any backward segment: `0`;
- arc/chord ratio P95/P99/max: `1.03003 / 1.06014 / 1.28102`;
- maximum local turn P95/P99/max:
  `4.50 / 7.21 / 16.15` degrees.

The exact post-lifecycle 9k checkpoint, before late guide/shape optimization,
also has zero backward segments. Its arc/chord P95/P99/max is
`1.03845 / 1.04007 / 1.04040`, and its maximum local-turn P95/P99/max is
`2.59 / 2.63 / 2.72` degrees. This confirms that the finite lifecycle itself
produces a coherent brushed field rather than relying on late-stage residuals
to repair loops or foldbacks.

R036's arc/chord P99/max are `1.00658 / 1.02890`; R038 is therefore genuinely
curved rather than visually reproducing the old straight construction. The
zero backward-segment result distinguishes that controlled curvature from a
loop or foldback.

Canonical three-view inspection found no loop, inward fold, isolated long
spike, width collapse, or local geometry failure. The brush-strength map is a
low-frequency guide-owned field; bend remains spatially coherent, while curl
and frizz are exactly zero. Full-resolution RGB errors remain concentrated in
head/cheek fine stripes, silhouette detail, and high-frequency coat appearance,
not a newly blurred body region.

The remaining structural limitation is root-level speckle in the effective
length field, which is also visible in R036. It is recorded for a later
isolated field-regularity experiment rather than hidden by changing R038.

## Decision And Artifacts

R038 passes its formal acceptance checks and becomes the active
structural/lifecycle baseline. R036 remains immutable as the higher-PSNR metric
control, so future work must report against both where efficiency and metric
quality trade off.

Machine-readable identity:

- source tag: `stage1-r038`;
- lock: `configs/r038_brush_curve.lock.json`;
- formal output:
  `/home/wangyy/anigroom-r038-runtime-20260805/outputs/r038_from_zero_h100_20260805`.

Local fixed-protocol evidence:

- 9k asset:
  `D:/RTS/_tmp/r038_9k_final/r038_009000_asset_side_y_v11_protocol.png`;
- 30k side asset:
  `D:/RTS/_tmp/r038_30k_final/r038_030000_asset_side_y_v11_protocol.png`;
- 30k opposite-side asset:
  `D:/RTS/_tmp/r038_30k_final/r038_030000_asset_side_y_pos_v11_protocol.png`;
- 30k third canonical view:
  `D:/RTS/_tmp/r038_30k_final/r038_030000_asset_front_z_v11_protocol.png`;
- full-resolution RGB views and diffs:
  `D:/RTS/_tmp/r038_30k_final/postprocess_030000/rgb_views`;
- canonical attribute maps:
  `D:/RTS/_tmp/r038_30k_final/postprocess_030000/attributes_view09`.
