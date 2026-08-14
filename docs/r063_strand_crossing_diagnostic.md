# R063 Strand-Crossing Diagnostic

## Status

This is a diagnostic checkpoint derived from frozen `stage1-r062`. It does not
change training, geometry, losses, interpolation, lifecycle, or rendering.
R062 remains the accepted baseline until a crossing treatment passes a strict
from-zero comparison.

## Question

The remaining visual crossing problem must be separated into three cases:

1. one strand folds back on itself;
2. two strands physically intersect in 3D at a high angle;
3. two valid strands overlap only after projection into a camera.

The existing structure audit already covers case 1. A camera-space line
intersection cannot distinguish cases 2 and 3, so it is not a valid training
target by itself.

## Exact 3D Protocol

`tools/diagnose_strand_crossings.py` evaluates all 100,000 fixed-protocol
strands and all 32 samples per strand. It does not subsample strands or
segments.

Each polyline segment uses its learned transverse Gaussian scale as a one-sigma
capsule radius. A midpoint KD-tree supplies an exact conservative broad phase;
exact segment-to-segment distance and linearly interpolated endpoint widths
then determine contact. Same-strand pairs are excluded because self-foldback is
already audited separately.

For each contacting pair, the continuous crossing score is

```text
overlap_fraction * sin(axis_angle)^2
```

This makes parallel dense fur approach zero while high-angle penetration
remains large. There is no world-space distance threshold, body-part rule,
strand-length rule, or camera projection in the contact decision. The
15/30/45/60-degree counts are reporting and visualization bins only; they are
not training thresholds.

The formal protocol distinguishes a projected crossing from a physical one:
two curves that cross in an image but whose 3D capsule envelopes do not touch
are not reported.

## Frozen-Baseline Results

| metric | R061 | R062 |
|---|---:|---:|
| unique one-sigma contact pairs | 16,424 | 16,291 |
| pairs at least 15 degrees | 3,446 | 3,152 |
| pairs at least 30 degrees | 898 | 748 |
| pairs at least 45 degrees | 256 | 230 |
| pairs at least 60 degrees | 71 | 75 |
| strands involved at least 45 degrees | 390 | 379 |
| crossing-score P95 | 0.12410 | 0.10885 |
| crossing-score max | 0.96093 | 0.89609 |

R062 does not introduce a new crossing regression. It reduces the 45-degree
pair count by 10.2% and the crossing-score P95 by 12.3% relative to its direct
R061 control while preserving the accepted reconstruction result.

## R062 Attribution

Among the 230 R062 pairs at or above 45 degrees:

- 154 also have chord-axis disagreement at or above 45 degrees;
- 42 have chord-axis disagreement between 15 and 45 degrees;
- 34 have chord-axis disagreement below 15 degrees and are therefore more
  consistent with local curl/frizz deformation than with the main direction
  field;
- only 12 pairs contact with both points in the root quarter;
- 159 pairs have at least one contact point in the tip quarter.

The crossing strands are not simply the widest strands. Their mean-width
percentile has median 52.6%. Length contributes to the tail but does not explain
the complete failure set: involved-strand length percentile has median 63.5%,
while its P90 is 99.1%.

The dominant failure is therefore sparse nonlocal tip convergence driven by
effective 3D direction/chord disagreement. A global increase in local KNN
smoothness would penalize the whole groom while missing many root pairs that
are not local neighbors. Curl/frizz is a minority source, not the main target.

## Canonical Evidence

- report:
  `D:/RTS/_tmp/r063_crossing_baseline_20260814/strand_crossing_report.json`;
- diagnostic NPZ:
  `D:/RTS/_tmp/r063_crossing_baseline_20260814/strand_crossing_diagnostic.npz`;
- side highlight:
  `D:/RTS/_tmp/r063_crossing_baseline_20260814/r062_crossing45_side_y.png`;
- opposite-side highlight:
  `D:/RTS/_tmp/r063_crossing_baseline_20260814/r062_crossing45_side_y_pos.png`;
- front/top highlight:
  `D:/RTS/_tmp/r063_crossing_baseline_20260814/r062_crossing45_front_z.png`.

The three fixed-camera renders highlight the same 379 strands. Visible red
clusters occur mainly near the face/ears, groin recesses, sparse upper-body
convergence zones, and tail tip. Most of the torso and limbs remain unmarked.

## Next Isolated Candidate

The evidence supports a differentiable active-set crossing term that uses the
same continuous score as the diagnostic:

```text
L_cross = mean(
    relu((r_i + r_j) - distance_ij) / (r_i + r_j)
    * (1 - abs(dot(t_i, t_j))^2)
)
```

The learned widths define the contact envelope but must be detached in this
loss so the optimizer cannot remove a crossing by making fur artificially thin.
Candidate discovery may be detached and refreshed as an active set; the loss
itself must remain differentiable to strand geometry. No camera, anatomical
mask, absolute length, or body-specific distance enters the method.

Before a from-zero R063 run, the implementation must pass exact synthetic
crossing/parallel/projection-overlap tests, gradient ownership checks, a
full-resolution memory/runtime preflight, and a measured active-set refresh
benchmark. The loss weight must be calibrated from observed gradient scale,
not chosen to fit one white-tiger region.
