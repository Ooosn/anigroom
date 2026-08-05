# R021 Unbounded Log-Length Coordinate

## Status

`rejected as a baseline`; formal H100 9k-14k calibration completed.

R018 remains the accepted full baseline. R019 established that explicit direct
3D render-root residuals improve direction continuity but exposed an invalid
additive length coordinate. R020 removed the absolute clamp with a bounded
log-ratio, but its inherited scale made length learning roughly one order of
magnitude weaker than R019.

## Isolated Question

Can render-root length detail be learned as its natural dimensionless
coordinate, `log(effective_length / guide_length)`, without absolute length
bounds, relative saturation, or a coordinate-dependent duplicate smoothness
term?

## Representation

R021 uses:

```text
effective_length = guide_length * exp(residual_ramp * log_length_residual)
```

The learned render-root value is the log length ratio itself. Therefore:

- zero residual follows the guide field exactly;
- effective length is strictly positive without a minimum or maximum clamp;
- equal residuals mean equal relative changes across different animal scales;
- the coordinate has no `tanh` saturation;
- densification copies/interpolates a meaningful zero-centered coordinate;
- continuity is measured on the final effective length field by the existing
  scale-free surface-graph loss, rather than also smoothing the raw coordinate.

Guide-root decoding is unchanged in this experiment. Removing guide-level
absolute decoder bounds remains a separate question.

## Calibration Contract

R021 starts from the verified R020 Phase-A 9k checkpoint. R020 and R019 are
identical before render residual unlock, and the R020 9k trajectory matches
R019 within 0.01 dB. Optimizer state is reset exactly as in the formal Phase-B
protocol.

Only the following method-level changes are active relative to R020:

```text
RENDER_GEOMETRY_PARAMETERIZATION=zero_centered_unbounded_log_length_residual
GUIDE_LENGTH_RESIDUAL_SCALE=1.0
raw residual-length graph term disabled
```

The last line removes duplicate regularization in a representation-dependent
coordinate. The final effective groom length remains regularized with the same
surface graph and weight as R020.

The first gate is 9k-14k. A successful calibration may continue from 14k to
30k with optimizer state preserved; a failed calibration is stopped without
spending a complete run.

## Acceptance Gates

1. No absolute length-bound occupancy, non-finite values, OOM, or reload error.
2. At 12k-14k the learned log-ratio must move materially beyond R020's nearly
   zero field without saturating or producing an extreme tail.
3. Test composite PSNR must recover the R019/R020 trajectory rather than losing
   reconstruction capacity.
4. Surface-neighbor effective-length ratios must not regress from R020.
5. Canonical pure-fur diagnostics must not introduce isolated long hairs,
   short-hair speckle, curled-back strands, or density holes.
6. Only a candidate passing these gates may replace R020 and continue to 30k.

## Calibration Result

R021 completed at 1920x1080 without OOM, non-finite values, fallback, or
capacity mismatch. It recovered most of the reconstruction capacity lost by
R020:

```text
14k test composite PSNR
R019: 31.1100
R020: 30.4001
R021: 30.9736
```

The central length distribution also remained coherent. The structural tail,
however, failed the acceptance gate:

```text
14k effective/guide ratio P01/P50/P99: 0.791 / 1.000 / 1.411
14k effective/guide ratio min/max:     0.043 / 14.214
14k neighbor length jump P95/P99:      0.242 / 0.487
fraction below the historical floor:   3.26%
```

The failure is localized rather than global. R021's neighbor P95 is slightly
better than R019, while its P99 and extrema are worse. The bounded symmetric
relative difference used by the final effective-length smoothness saturates as
the length ratio becomes extreme, so its gradient cannot recover a small set
of escaped raw log residuals. R021 is therefore not continued to 30k. R022
tests raw log-residual surface smoothness as the isolated correction.
