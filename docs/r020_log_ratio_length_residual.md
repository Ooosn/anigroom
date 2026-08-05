# R020 Log-Ratio Length Residual

## Status

`completed as an intermediate calibration; superseded`.

The bounded `tanh` log-ratio removed the additive physical floor failure, but
its residual P95 at 12k was about 25 times smaller than R019's normalized
coordinate and its 14k test composite lagged by about `0.41 dB`. R021 tested
the corresponding unbounded raw log coordinate, and R023's `asinh` log-ratio
became the retained decoder. R020 is not an executable baseline.

R018 remains the accepted baseline. R019 remains a completed representation
ablation: its direct local-frame 3D direction residual is retained, while its
additive full-decoder-span length residual is rejected.

## Isolated Question

Can a render root learn local length detail as a true zero-centered residual
without depending on an absolute length minimum or maximum, while preserving
the guide field's spatial coherence?

R020 changes only the length composition used by the R019 explicit residual
field. Direction and all other training, lifecycle, renderer, data, and
evaluation settings remain identical to R019.

## Representation

R019 used an absolute physical displacement:

```text
length = clamp(guide_length + multiplier * scale * decoder_span * tanh(raw))
```

That gave short and long guide strands the same absolute displacement. Short
strands therefore crossed the decoder floor and accumulated at the clamp.

R020 uses a dimensionless log-ratio:

```text
log_ratio       = multiplier * scale * tanh(raw)
effective_length = guide_length * exp(log_ratio)
```

Properties of this representation:

- zero raw residual follows the guide exactly;
- positive guide length remains positive without an absolute clamp;
- equal residuals produce equal length ratios, not equal metric offsets;
- the residual meaning is independent of the configured guide decoder range;
- render-root smoothing acts on a dimensionless local log-ratio field;
- densification transports the same zero-centered residual coordinates.

The guide field still uses the accepted R018 decoder in this isolated test.
Removing guide-level absolute bounds is a separate later experiment and is not
silently combined with R020.

## Fixed Experiment Contract

The R020 phase configs are flattened copies of R019. Their only assignment
difference is:

```text
RENDER_GEOMETRY_PARAMETERIZATION=zero_centered_log_length_residual
```

Formal protocol:

1. train from zero through iteration 9000;
2. strictly load that R020 checkpoint, reset optimizer state, and continue to
   iteration 30000;
3. use the same V4 clean-flow target, 1920x1080 data, view split, lifecycle,
   losses, random mesh backing, renderer, and evaluation as R018/R019.

## Acceptance Gates

R020 can advance only if all gates pass:

1. Phase A matches the accepted R018/R019 trajectory before residual unlock;
2. no OOM, non-finite value, reload mismatch, or lifecycle state mismatch;
3. no point mass at the historical `0.01` length floor;
4. local relative-length jumps improve substantially over R019 and are
   competitive with R018;
5. direction continuity keeps the R019 improvement;
6. canonical pure-fur renders contain no new salt-and-pepper length field,
   isolated long hair, curl-back, density hole, or blurred patch;
7. RGB remains competitive, but RGB gain cannot override a fragmented groom
   field.

If R020 fails, the next R-series candidate must change only the positive length
residual map; it must not retune losses or schedules to hide the failure.
