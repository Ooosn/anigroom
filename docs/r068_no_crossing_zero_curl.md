# R068 No-Crossing + Zero-Curl-Fastpath Candidate

Status: formal candidate prepared; no HGC run or acceptance claim has been
made. The candidate is a strict child of the accepted R067 no-frizz baseline.

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

R068 is accepted only if the runtime contract passes, the no-frizz/schema-9
contract is preserved, the no-crossing state is proven in checkpoint and log,
and the matched quality and asset review gives a defensible answer to the
method question. Preparing these files does not constitute that acceptance.
