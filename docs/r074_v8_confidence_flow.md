# R074 Panda V8 Confidence-Flow Gate

Status date: 2026-08-30.

Status: launcher prepared; formal V8 target acceptance and H100 execution are
pending. No R074 checkpoint is accepted by this document yet.

## Question

R073's 3k checkpoint uses the accepted V7 target. Its guide-direction residual
multiplier is zero, yet the pure-fur upper-back asset already contains opposing
flow. The corrected target audit proves that this is an initialization seam,
not a learned R073 ownership failure. Does replacing only the external
clean-flow target with formal V8 remove that seam without changing R073's
training behavior or reintroducing its early sparse/noisy coverage problem?

## Single Method Change

R074 inherits R073 exactly:

- equal-owner-budget trusted view gating;
- the same 30 train / 6 test cameras;
- the same guide/render populations and lifecycle;
- the same reconstruction, mask, geometry, appearance, and regularization
  losses;
- the same learning rates and 0-3k schedule;
- the same Panda data, mesh, SDF, and identity alignment.

The only changed input is `CLEAN_FLOW_TARGET`: V7 is replaced by the formally
generated V8 confidence-guided Panda target. The config intentionally changes
no executable variable:

`configs/r074_v8_confidence_flow_0_3k_gate.env`

The H100 launcher verifies the exact source commit and target SHA-256, V8 NPZ
schema, enabled summary report, finite arrays, and zero-new-severe invariant
before training:

`scripts/server/run_panda_r074_v8_confidence_flow.sh`

## 0-3k Gate

The first run stops at 3k. Acceptance requires:

1. full-suite and full-resolution view-09 preflight success;
2. finite gradients, Adam state, and checkpoint reload;
3. no lifecycle or memory failure;
4. the same expected-gradient view budget as R073;
5. a canonical pure-fur side-positive-Y asset, not only a training camera;
6. no opposing flow in the previously marked upper-back crop;
7. no blue sparse/bald noise patch in that crop, and no material density loss
   relative to R073 under the same asset protocol.

The R073 metric reference at 3k is train/test composite
`18.96358 / 19.13590`, peak allocation `9832.7 MB`, and checkpoint SHA-256
`c37bb87fe30745f64dad9fc57c113476f9a4366bfe52ae42ddd6f5263095f836`.
R074 is not required to improve this early metric; the gate is directional
coherence and asset coverage under an otherwise identical run.

A later 6k/9k continuation is not automatic. It is authorized only after the
3k asset and crop are visually accepted. A 30k run must start from zero with
the final accepted target rather than resume the frozen R073 checkpoint.
