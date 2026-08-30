# R074 Panda V8 Confidence-Flow Gate

Status date: 2026-08-30.

Status: formal V8 target accepted; from-zero H100 0-3k run completed. The
checkpoint/reload gate passed; 100k/full-strand/3DGS asset acceptance is in
progress, so no later training stage is authorized yet.

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

The active run uses source
`bc8265716fb1493af73dc5a165a885df6e4aa915` and Panda target SHA-256
`5cb76945adb034e9666bfc98ae05647062d7ac4e3609e68162e561e4eebd54b1`.
Its runtime is `/home/wangyy/panda-r074-v8-runtime-20260830`.

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

## H100 3k Result

The uninterrupted run completed at `2026-08-30T19:00:02+09:00` without OOM,
traceback, restart, or schema fallback:

- train/test composite PSNR: `18.826702 / 18.988163`;
- difference from R073: `-0.136879 / -0.147737 dB`;
- final render roots: `480292` from `400000` initial roots;
- 25 lifecycle events through 3k, net `+80292` roots;
- training/reload preclip Gaussians: `5128628 / 5160579`;
- peak allocated/reserved CUDA memory: `9761.45 / 13880 MB`;
- checkpoint SHA-256:
  `fcd62694663a7ab9383ff0250fa6a44544b7bafff1ebc96ffd7a2e05ad8d013e`.

The exact V8 target and view ownership are verified. Guide reconstruction uses
4488 direct roots, fills 4490, and retains 4449 anchors. Relative to R073, the
direction reconstruction change improves from mean/P95
`4.33902/17.37752 deg` to `4.02535/16.20302 deg`; mean reliability changes
`0.60406 -> 0.59668`. Owner support is identical: 30 requested training views,
28 trusted, 3995 guides with an owner, and 505 zero-owner guides.

The early metric cost is recorded, not hidden. It is small enough to proceed
to the mandatory asset comparison, but the checkpoint is not promoted until
the user-region flow and full-population coverage are inspected.

A later 6k/9k continuation is not automatic. It is authorized only after the
3k asset and crop are visually accepted. A 30k run must start from zero with
the final accepted target rather than resume the frozen R073 checkpoint.

## Asset-Coverage Correction

The earlier R073 Blender image is not a complete checkpoint asset: it samples
100,000 of 484,442 render roots (`20.64%`) and compensates with width scale
`1.65`. It must be retained only as a matched comparison, not presented as the
full groom. R074 postprocess must additionally export every render strand at
physical width scale `1.0` and every training Gaussian as a full 3DGS PLY.

R073's sampled strand arc-length median/max is `0.01117/0.01422`. This is not
an optimizer accident. The formal Panda target's reliable shell-height
5%-95% interval is `0.02524-0.04706`; inherited
`CLEAN_FLOW_LENGTH_INIT_SCALE=0.30` maps it to `0.00757-0.01412`, matching the
asset. R074 intentionally preserves this value to isolate flow direction. If
the full-population asset is still visually too short, the next experiment
must change only this scale to the data-identity value `1.0`; it must not mix a
length repair into R074 or hide sparse export with thicker curves.

R074 postprocess exports all three required representations:

- matched 100k strands: 45,771,861 bytes, SHA-256
  `0712bbbdbff580360cda59f2661dc1b4a41d9dc59684ac5451a0fa7d01ebc24b`;
- all 480,292 render strands: 217,828,226 bytes, SHA-256
  `907460d5eac02a157520d40b277941fd97917feca66df97e11297923ddb69c8c`;
- all 5,160,579 training Gaussians as 3DGS PLY: 1,279,825,124 bytes,
  SHA-256
  `3e8d65e9080e094b9a9f1dc0f91964692c678f28b60a264c9fb88eea494e8367`.

The matched 100k Blender scene validates 100,000 splines and one body mesh. A
240k density-control render at physical width scale `1.0` completes and is
finer/denser than the historical 100k/`1.65` view, proving that export
subsampling and width compensation contribute to the sparse/coarse look. It
still remains visibly short, agreeing with the independent 0.30 length-scale
diagnosis. A single Blender scene containing all 480,292 32-sample curves hits
the Blender/Cycles allocator near 33 GB and crashes before producing an image;
that failure is retained and is not mislabeled as an asset success. The full
NPZ and full 3DGS PLY are complete and hash-verified.

Most importantly, the trained 100k asset re-runs the original upper-back crop
at guide and render all/front/back negative counts `0/0/0`, >120-degree counts
`0/0/0`, and render-chord-versus-guide negative count `0`. R074 therefore
passes the direction-transfer gate. It does not pass the desired coat-length
gate; R075 isolates the data-identity length scale.
