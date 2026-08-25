# R068 Candidate 1: Exact Packed Appearance Smoothness

## Status

This document records the acceptance contract for the R068 candidate. The
implementation is pending the focused CPU checks and the CUDA speed gate.

## Exact Runtime Change

When secondary guides are enabled, root-graph smoothness is appearance-only on
the render-root graph. The packed path decodes only root color, tip color,
root opacity, and tip opacity from their raw parameters, packs them into eight
channels, and applies the unchanged per-channel coefficients:

- root RGB: `0.25 / 3` per channel;
- tip RGB: `0.15 / 3` per channel;
- root opacity: `0.50`;
- tip opacity: `0.25`.

Confidence edge weights are detached once per graph state as
`0.25 + (1 - min(conf[src], conf[dst]))`, with the clamped denominator cached
alongside them. The cache is rebuilt at initial graph setup and after each
lifecycle graph or root-confidence update. Non-secondary and non-appearance
paths retain the existing reference implementation and coefficients.

## Acceptance Gates

1. `tests/test_r068_packed_appearance_smoothness.py` passes on CPU, covering
   scalar equality, all four appearance gradients, cached and uncached weights,
   empty graphs, cache refresh, direct raw decoding, and zero geometry gradients.
2. `python -m py_compile tools/train_white_tiger_stage1.py
   tools/benchmark_r068_packed_appearance_smoothness.py` passes.
3. The benchmark runs on CUDA with synthetic R067-size inputs: `471673` roots,
   directed K32 edges, warmup, and at least 20 measured forward/backward
   repetitions. It reports median and P95 time, peak allocation, and scalar and
   gradient maximum errors against the explicit reference.
4. A diff check confirms that only the scoped training tool, R068 tests,
   optional benchmark, and this document changed.

## Benchmark Artifact

The benchmark writes its JSON report to
`outputs/r068_packed_appearance_smoothness/benchmark.json` by default. The
command is:

```text
python tools/benchmark_r068_packed_appearance_smoothness.py
```
