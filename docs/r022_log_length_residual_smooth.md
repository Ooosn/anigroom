# R022 Log-Length Residual Smoothness

## Status

`rejected after continuation`; the 9k-14k calibration was coherent, but a
sparse render-root length tail escaped when the residual ramp reached 16k.

## Evidence From R021

R021 recovered most of R019's reconstruction capacity while removing the
absolute effective-length clamp. At 14k its test composite PSNR was `30.974`,
versus `31.110` for R019 and `30.400` for R020.

The central field was coherent, but the unbounded coordinate developed a small
extreme tail:

```text
effective/guide ratio P01/P50/P99: 0.791 / 1.000 / 1.411
effective/guide ratio min/max:     0.043 / 14.214
neighbor length jump P95/P99:      0.242 / 0.487
```

The final effective-length regularizer used a bounded symmetric relative
difference. Its gradient tends to zero as a neighboring length ratio becomes
extreme, so it cannot reliably recover a residual after it escapes.

## Isolated Change

R022 retains the R021 representation:

```text
effective_length = guide_length * exp(residual_ramp * log_length_residual)
```

It restores surface-graph smoothness on the learned raw log-ratio coordinate:

```text
(log_length_residual_i - log_length_residual_j)^2
```

This is scale-relative and introduces no absolute length minimum, maximum, or
animal-specific threshold. The existing final effective-groom smoothness is
unchanged. The two terms have separate roles: the residual term keeps local
multiplicative deviations coherent, while the effective term regularizes the
actual strand controls after guide interpolation.

## Calibration Gate

R022 restarts from the same verified 9k Phase-A checkpoint as R019-R021 with a
reset optimizer and runs to 14k. It is accepted only if:

1. test composite PSNR remains on the R019/R021 trajectory;
2. effective/guide ratio tails contract materially from R021;
3. neighbor length-jump P99 improves without degrading P50/P95;
4. the canonical 100k pure-fur render removes isolated long/short hairs;
5. no absolute clamp, non-finite state, OOM, or hidden fallback is introduced.

## Measured Result

R022 reached test composite PSNR `30.8502` at 14k. This was below R019
`31.1100`, but its central length field and canonical 100k-strand render were
coherent enough to continue without changing the method.

The 14k-30k continuation was stopped immediately after the 16k checkpoint
exposed a sparse tail failure:

```text
iteration   test composite   effective max   effective/guide max
14000       30.8502          0.1768          7.596
15000       31.2907          0.2113          -
16000       31.2161          2.1655          44.965
```

A same-root 14k/15k/16k diagnostic reconstructed the exact surface graph. The
largest failures include both local clusters and isolated roots whose graph
neighbors remain normal. Root motion is negligible. The graph is therefore
present and correct; the failure is the exponential decoder
`guide * exp(ramp * raw)` amplifying sparse RGB-driven residual coordinates
faster than a global-mean smoothness loss can suppress their tail.

R022 remains evidence that raw-coordinate graph smoothness is useful, but the
raw exponential decoder is rejected for the full residual ramp.
