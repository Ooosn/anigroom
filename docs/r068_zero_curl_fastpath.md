# R068 Zero-Curl Fast Path

## Scope

This candidate adds an explicit `enable_curl` boolean to `build_strands`. The
default remains enabled, so the existing differentiable curl path is unchanged.
Stage 1 disables the deformation call only when both
`shape_detail_multiplier` and `shape_curl_scale` are exactly on their frozen
zero side. The disabled path returns the brush backbone directly and uses the
same flag for mesh no-penetration local strands.

Decoded groom fields, sampling, outputs, active-field gradients, statistics,
crossing, SDF queries, and lifecycle behavior are unchanged. This is a runtime
optimization, not a method change.

## Boundary

The optimization applies only during the exact-zero frozen phase. It removes
the differentiable local-frame and trigonometric curl work while curl is
disabled. After curl unlocks, the full path is used and there is no speed gain
from this fast path.

## Validation

Focused tests cover exact default equivalence, nonzero curl outputs and
gradients, exact brush-backbone identity, model flag routing, disabled curl
gradient ownership, and the mesh no-penetration shape/gradient contract.

The CUDA benchmark uses synthetic R067 scale (`471673` roots and `64`
samples), runs forward plus backward for at least 20 repetitions, and reports
median/P95 time, peak allocation, scalar/output errors, and gradient errors:

```text
python tools/benchmark_r068_zero_curl_fastpath.py
```

The benchmark CLI is exercised on both the local RTX 4080 SUPER and the held
HGC H100 allocation; neither run changes a checkpoint or training state.

## Measured Benchmarks

The exact R067-size benchmark uses `471673` roots, `64` samples, five warmup
repetitions, and twenty measured forward/backward repetitions.

| Device | Existing zero-radius full path median / P95 | Explicit disabled median / P95 | Peak allocation full / disabled |
| --- | ---: | ---: | ---: |
| RTX 4080 SUPER | `232.428 / 234.039 ms` | `32.585 / 33.218 ms` | `9.238 / 4.569 GB` |
| H100 80GB HBM3 | `52.556 / 52.714 ms` | `7.601 / 7.635 ms` | `9.221 / 4.558 GB` |

Both devices report exact zero scalar, output, length-gradient,
direction-gradient, and brush-stiffness-gradient differences. The disabled
path intentionally has no curl-field gradient; in formal training the same
fields are already frozen behind an exact zero shape-detail multiplier.

At the H100 median, the avoided work is `44.955 ms` for each exact-zero
iteration. Applying that measured kernel delta to the first `14000` frozen
iterations gives an upper-bound estimate of `629.4 s` (`10.5 min`) saved. It
does not accelerate the post-unlock phase and therefore cannot by itself
explain or remove the full R055-to-R067 wall-time increase.

## Decision Boundary

This is a small, exact, and directly causal optimization worth retaining as an
isolated candidate. It is not accepted into the R067 baseline and no formal
30k run is launched until the broader module-cost review decides whether the
current advanced-geometry route remains the intended default.
