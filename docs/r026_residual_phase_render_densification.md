# R026 Residual-Phase Render Densification

## Status

`stopped at 16k; diagnostic rejected because optimizer state reset confounds the method change`.

The formal H100 run reached iteration 16k and saved a verified checkpoint
(`SHA-256 2ce7d5910d30a7d840d2a856f464c7d2b86210d6e3528fdc30c0116e4944c512`).
At 16k, R026 has `282095` render roots and test composite PSNR `31.1301`,
versus R025's `217036` roots and `31.2422`. The extra `65059` roots therefore
do not recover the optimizer-reset cost. The run was stopped after preserving
the 16k evidence; the H100 allocation was retained for R027.

## Evidence From R025

R025 completes at test composite PSNR `32.0910`, compared with R024
`32.0572`. It lowers the maximum effective length from `0.7801` to `0.5057`
and the largest connected top-0.1% residual component from `81` roots to `32`.
Canonical 100k-strand assets nevertheless show a visible tail-tip outlier from
25k onward.

The outlier is not an invisible or transparent root. The top residual is
approximately `guide_length=0.0465`, `ratio=10.87`,
`effective_length=0.5057`, and `opacity=0.9785`. Its local neighbors also have
large residuals. The remaining failure is therefore a coherent render-root
residual cluster, not a guide-length maximum or low-opacity strand.

## Schedule Mismatch

The accepted schedule stops render-root densification at 10k, exactly when
render geometry residuals begin to unlock. From 10k onward, unresolved local
coverage can only be represented by changing existing roots. At the tail tip,
that creates a competition in which length inflation is cheaper than adding
the missing short strands.

## Isolated Change

R026 keeps R025's representation, losses, evidence threshold, split placement,
interval, uncapped event size, data, evaluation, and optimizer contract. It
changes only:

```text
DENSIFY_UNTIL: 10000 -> 20000
```

This overlaps evidence-driven render-root split with the complete 10k-20k
residual ramp, then leaves 10k iterations after the last split for continuous
optimization. Split selection remains gradient/evidence based. It does not use
length, animal region, percentile, or a physical length bound.

## Gate

1. Root growth must be driven by evidence and must not explode or hit the 30 GB
   memory guard.
2. P50/P95 length and reconstruction must remain competitive with R025.
3. The late top-residual connected component and maximum effective length must
   fall materially.
4. Canonical 100k-strand 25k/27k/30k assets must remove the visible tail-tip
   long line without introducing density clumps elsewhere.
