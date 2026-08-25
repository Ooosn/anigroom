# R068 No-Crossing + Zero-Curl Fast Path

Status: accepted on 2026-08-26 as the current single-sample method baseline.
R068 is a strict child of R067. It removes crossing from default training and
adds one exact runtime fast path while preserving the R067 method output.

## Method Question

Does removing the exact strand-crossing active-set refresh and loss, while
retaining R067's no-frizz schema and the already implemented zero-curl
fastpath, preserve the accepted reconstruction quality and mesh validity? The
question is isolated: no schedule, neighborhood K, loss, resolution, root
count, lifecycle, SDF, or learning-rate change is allowed outside crossing
removal.

The zero-curl path is a runtime path selection only. `build_strands` retains
the explicit `enable_curl` flag, and Stage 1 forwards the exact frozen-zero
condition to both ordinary and mesh-no-penetration strands. After curl is
enabled, the existing differentiable path remains selected.

## Exact Config Contract

`configs/r068_no_crossing_zero_curl_0_30k.env` sources the exact
`configs/r067_no_frizz_0_30k.env` file, then changes only:

```json
{
  "STRAND_CROSSING_SUPPORT": {"r067": "1", "r068": "0"},
  "STRAND_CROSSING_WEIGHT": {"r067": "0.001", "r068": "0"},
  "STRAND_CROSSING_REFRESH_INTERVAL": {"r067": "2000", "r068": "0"}
}
```

`STRAND_CROSSING_QUERY_BATCH=50000` and
`STRAND_CROSSING_EXACT_PAIR_BATCH=250000` remain unchanged and inert. The
strict runner snapshots both resolved environments and rejects every other
delta. Resolution remains `1920x1080`; the run remains uninterrupted from
zero through `30000` iterations.

## Formal Runtime Gate

`scripts/server/run_r068_no_crossing_zero_curl.sh` requires:

- a full 40-character `EXPECTED_COMMIT` and a clean checkout;
- a new runtime root and output directory, with no reuse;
- the verified `mygs` interpreter and frozen data, mesh, and mesh-SDF paths;
- checkpoint schema 9, `stage1_full`, and no R067 frizz keys;
- unlimited virtual memory and the complete pytest suite before training;
- the full input preflight at native `1920x1080`, with no reduced batch
  preflight or preflight-only stop;
- no resume checkpoint, optimizer resume, view override, fallback, or reduced
  resolution;
- the exact candidate config and source-level `enable_curl` checks;
- an uninterrupted from-zero `0-30k` training run.

The final checkpoint must contain `strand_crossing_support=false`, zero
crossing weight and refresh interval, no active set, refresh iteration zero,
and an empty crossing history. The formal log must show disabled crossing
state with zero pairs and zero history, and must contain no crossing refresh
event. The metrics log must retain the crossing fields with zero active pairs
and zero last-refresh iteration through the final record.

## Acceptance Evidence

Acceptance requires the runtime gate to pass and the final output to be
compared with R067 under the same fixed protocol. The comparison must cover:

- **Runtime:** exit status, wall time, peak allocation, exact commit, schema 9,
  resolved config delta, and the zero-curl source proof.
- **Quality:** train/test and fixed-view composite metrics, child-strand and
  clump structure, cleaned flow or RGB-to-flow/edge-style behavior, local
  turn and foldback diagnostics, and no-frizz key absence.
- **Assets:** the same fixed views, resolution, sampling, root mode, camera,
  strand count, line style, and scalar ranges used for R067; no visual
  regression in the canonical assets.
- **Exact crossing postprocess:** run the exact continuous 3D crossing audit
  on the final R068 checkpoint and compare its contacts, high-angle contacts,
  involved strands, backward segments, foldbacks, and length ownership with
  R067. Training must show no active-set history even though the postprocess
  diagnostic may independently measure crossings.
- **No-penetration postprocess:** run the all-root matched mesh-SDF audit with
  the frozen R067 SDF and compare penetrating-point/root fractions, mean depth,
  maximum depth, and canonical no-penetration assets against R067. This is a
  validity comparison, not permission to change the SDF or collision loss.

## Formal Result

Training used commit
`3dfff62e621a91ba0d30764fdf780b2d1e247672`. The post-run metric validator
initially rejected the 85 lifecycle JSON records because those records do not
carry per-evaluation crossing fields. All 31 actual train/test metric records
did carry the expected zero crossing state. Commit `a7f3601` fixes the
validator to distinguish those schemas; the saved 30k run was then validated
without rerunning or changing training.

| Measure | R067 | R068 | R068 - R067 |
| --- | ---: | ---: | ---: |
| wall time | `15775.028 s` | `11885.196 s` | `-3889.832 s` |
| train composite | `33.101788` | `33.117367` | `+0.015579 dB` |
| test composite | `32.069145` | `32.083080` | `+0.013935 dB` |
| fixed eight-view composite mean | `33.077009` | `33.101055` | `+0.024046 dB` |
| render roots | `471673` | `471482` | `-191` |
| pre-step metric Gaussians | `5382959` | `5380775` | `-2184` |
| peak allocated CUDA memory | `19825.54 MB` | `15869.80 MB` | `-3955.74 MB` |

R068 is `24.66%` faster (`1.327x`) and uses `19.95%` less peak allocated
memory. Root and Gaussian changes are both about `-0.041%`; the result does
not obtain speed by reducing reconstruction capacity.

The zero-curl path is active only while shape detail is exactly zero. The
resolved R067/R068 schedule confirms that this is iterations 0 through 20k,
not the obsolete 14k estimate in the first runtime note. At 20k R068 had
already saved `2672.434 s`; after curl and appearance fully unlock, the normal
R067 differentiable curl path is selected automatically.

## Structural And Validity Result

The matched deterministic protocol uses 100k render-root strands, child count
1, 32 samples, seed 29, and the same three 1920x1080 Blender cameras as R067.

- local relative-length mean/P95 changes
  `0.021676/0.082026 -> 0.021636/0.081259`;
- local direction mean/P95 changes
  `3.8717/11.4372 -> 3.8785/11.4079` degrees;
- arc/chord P95/P99 changes
  `1.005528/1.026208 -> 1.005447/1.028950`;
- local-turn P99/max changes
  `2.3235/3.6002 -> 2.5830/4.3594` degrees;
- backward strands and strict foldbacks remain zero;
- final curl cumulative-turn P50/P95 changes
  `2.0723/21.2044 -> 2.0025/20.8904` degrees.

The local-turn maximum rises by `0.7591` degree but remains small, with no
backward or foldback failure and no visible regression in any canonical asset.

The exact crossing diagnostic shows the limited value of the removed loss:

- unique intersecting pairs: `14872 -> 14983` (`+111`, `+0.75%`);
- strands with any contact: `23158 -> 23287` (`+129`);
- contact-axis pairs at least 45 degrees: `171 -> 208` (`+37`);
- chord-axis pairs at least 45 degrees: `157 -> 157`;
- contact-axis pairs at least 60 degrees: `71 -> 61`.

This small mixed diagnostic change is not visible in the matched side,
opposite-side, or top/front asset renders and does not justify the default
training cost. Crossing remains available as an offline diagnostic or an
optional local refinement; it is no longer part of the default reconstruction
objective.

No-penetration remains valid and slightly improves under the matched all-root
audit: penetrating point/root fractions change
`0.00021423/0.00402397 -> 0.00019944/0.00384532`, mean depth falls `7.96%`,
and maximum depth falls `1.17%`.

## Count Audit

The iteration-30000 metric is a pre-optimizer-step render count of `5380775`.
The persisted post-step checkpoint reconstructs `5380705`; the delta of 70 is
the same documented timing distinction as R067, not missing PLY rows. Three
complete reconstructions produce identical counts and root/segment order
hashes. The audit status is pass.

## Accepted Artifacts

Remote:

- checkpoint:
  `/home/wangyy/anigroom-r068-no-crossing-zero-curl-runtime-20260826/outputs/r068_no_crossing_zero_curl_0_30k_h100_20260826/checkpoint_030000.pt`;
- strict validation:
  `/home/wangyy/anigroom-r068-no-crossing-zero-curl-runtime-20260826/contracts/r068_postrun_strict_validation.json`;
- postprocess manifest:
  `/home/wangyy/anigroom-r068-no-crossing-zero-curl-runtime-20260826/postprocess/r068_protocol_20260826/r068_postprocess_manifest.json`;
- Gaussian-count audit:
  `/home/wangyy/anigroom-r068-no-crossing-zero-curl-runtime-20260826/postprocess/r068_protocol_20260826/analysis/r068_gaussian_count_audit.json`.

Local acceptance root:

`D:/RTS/_tmp/r068_acceptance_20260826/postprocess/r068_no_crossing_zero_curl`

Canonical asset images are under its
`assets_blender_protocol_20260826` directory. The checkpoint SHA-256 is
`949eaf2a71fc4f55c29591e8574905e77d2e55646f6d7416f75364f456c80c4b`;
the Gaussian-count audit SHA-256 is
`66878ba6d17053b4d9d89234f6a7b4024747f16083559bc88674ab8b4307e9bb`.

## Decision

R068 passes the runtime, no-frizz/schema-9, fixed-view RGB, deterministic
strand, curl/foldback, exact crossing, no-penetration, Gaussian-count, and
canonical-asset gates. It replaces R067 as the current single-sample method
baseline. The packed K32 experiment remains rejected: it adds code and memory
for a much smaller isolated gain and does not explain the historical slowdown.
