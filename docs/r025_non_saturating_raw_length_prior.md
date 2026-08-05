# R025 Non-Saturating Raw Length Prior

## Status

`completed; improved but rejected as the final fix`.

## Evidence From R024

R024 replaced the historical `tanh(raw)` prior coordinate with
`asinh(raw)`, matching the actual log-ratio used by the decoder. It reduced
the transient 19k maximum effective length from `0.3629` to `0.2128` without
changing the ordinary P95 or reconstruction score. At 24k, however, the
maximum rose again to `0.4436` with `raw=5.3195`.

The remaining failure is explained by the gradient of the R024 coordinate:

```text
d asinh(raw) / d raw = 1 / sqrt(1 + raw^2)
```

It is much better than the saturated `tanh` coordinate, but it still weakens
as the residual moves into the tail.

## Isolated Change

R025 retains the exact R024 decoder, schedule, weights, lifecycle, data, and
evaluation. Only the coordinate consumed by the existing length prior changes:

```text
R024 prior coordinate = asinh(raw)
R025 prior coordinate = raw
```

For the asinh decoder the physical composition remains:

```text
effective_length = guide_length * exp(scale * asinh(raw))
```

It is still positive, scale-relative, and unbounded. R025 does not add a
physical length minimum, maximum, percentile gate, region rule, or fallback.
The raw-coordinate L1 prior has a non-vanishing tail gradient, while the guide
field remains responsible for coherent regional length.

## Gate

1. Match R023/R024's P50/P95 and reconstruction trajectory through 20k.
2. Prevent the 24k-30k connected long-hair cluster seen in R023/R024.
3. Keep test composite competitive with R018/R019 and R023.
4. Pass matched graph diagnostics and canonical 100k-strand Blender QA.

## Result

R025 completes without traceback or OOM at train/test composite PSNR
`32.8130/32.0910`. Relative to R024, test PSNR rises by `0.0338 dB`, P95 length
falls from `0.036873` to `0.036749`, maximum effective length falls from
`0.7801` to `0.5057`, and the largest connected top-0.1% raw-residual component
falls from `81` roots to `32`.

The late tail is reduced but not eliminated. Canonical assets show a visible
tail-tip line by 25k and a clear outlier at 27k/30k. The top residual root is
opaque and lies inside a coherent high-residual neighborhood, so pruning
low-opacity roots or adding a guide-only smoothness term would not address the
observed failure. R025 is retained as the better residual coordinate and used
as R026's parent, but it is not locked as the final baseline.
