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

The benchmark is a local CUDA CLI and is not launched through HGC here.
