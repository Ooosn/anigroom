# R075 Panda Full-Shell Data-Identity Length Gate

Status date: 2026-08-30.

Status: from-zero H100 0-3k run completed and checkpoint/reload gates passed.
Matched/full-strand and full-3DGS asset export is pending; no later training
stage is authorized yet.

## Question

R074 preserved the accepted V8 confidence-guided flow target and the existing
reliable shell-height 5%-95% clamp, but its full-population evidence remained
short. R075 asks whether restoring the data-identity length scale improves
coverage and physical strand length without changing the V8 flow result or
any other R074 behavior.

## Exact single change

R075 inherits R074 exactly. The only executable change is:

`CLEAN_FLOW_LENGTH_INIT_SCALE: 0.30 -> 1.0`

The existing reliable shell-height q05-q95 clamp remains unchanged. No clean-
flow target, training schedule, loss, lifecycle, root population, camera,
Panda data, mesh, SDF, identity alignment, width, sampling, or asset
parameter changes are part of this gate.

The owned files are:

- `configs/r075_full_shell_length_0_3k_gate.env`
- `scripts/server/run_panda_r075_full_shell_length.sh`
- `tests/test_r075_full_shell_length_gate.py`
- `docs/r075_full_shell_length.md`

The config sources
`r074_v8_confidence_flow_0_3k_gate.env` and assigns only the scale override
after that source. The launcher retains R074's explicit source-commit,
runtime, target, target-hash, CUDA, V8-schema, full-test, view-09 full-
resolution preflight, Panda data/mesh/SDF/identity, from-zero 3k, checkpoint
reload/render, and hash contracts.

## R074 evidence and data identity

R074's matched strand evidence is arc length `0.00743-0.01428`, with median
approximately `0.01117`. The formal reliable shell-height q05/q95 is
`0.02524/0.04706`. At the R075 scale of `1.0`, that reliable physical range
is expected to remain `0.02524-0.04706`; the launcher logs the target-derived
values and the resolved preflight values when available without turning those
measurements into a new acceptance rule.

## 0-3k acceptance

Acceptance is focused on data identity and coverage while checking for a flow
regression against R074:

1. the full test suite and full-resolution view-09 preflight pass;
2. gradients, optimizer state, checkpoint reload, lifecycle, and memory remain
   valid under the inherited R074 contract;
3. full-population or matched asset evidence shows improved physical length
   and no sparse/bald coverage defect relative to R074;
4. the reliable length distribution is consistent with the formal shell-height
   q05/q95 evidence and does not introduce an out-of-range tail; and
5. V8 flow direction/coherence does not regress, including no new opposing-flow
   defect in the previously marked upper-back region.

The 3k gate is not a request for a new flow method or a later training stage.
Any continuation requires this coverage/length gate to pass first.

## H100 3k result

Retry2 job `127371170` completed from zero with `failed=0`, `exit_status=0`,
and no OOM/traceback. The exact result is:

- train/test composite PSNR: `21.963490 / 21.913940`;
- gain over R074: `+3.136788 / +2.925777 dB`;
- effective length q05/q50/q95:
  `0.029179 / 0.036531 / 0.042835`;
- effective length min/max: `0.024463 / 0.046553`;
- checkpoint/reload roots: `447462 / 449482`;
- training/reload Gaussians: `5690498 / 5716077`;
- peak allocated CUDA memory: `9644.07 MB`;
- checkpoint SHA-256:
  `779e5a18f852bfa24d927cbe1410090d22cfee036bcd6404ce60a293beb42193`.

The measured length interval lies inside the formal reliable shell evidence
and removes R074's ~`0.01111` mean short-coat state; measured arc mean is
`0.037064`. Direction reconstruction is unchanged from R074 at mean/P95
`4.02535/16.20302 deg`, and the equal-owner-budget view gate remains exact.
The metric gain, lower peak memory, and finite lifecycle clear the numerical
gate. Visual/full-asset coverage and the upper-back flow regression remain the
final acceptance boundary.

## HGC attempt ledger

The first qsub attempt, job `127370930`, was accepted with
`h100=1,mem_req=16G` and no `s_vmem`, but the H100 qsub environment did not
inject `CUDA_VISIBLE_DEVICES`. The main launcher's safety requirement therefore
exited in `0.454 s` before importing torch or creating the runtime
(`exit_status=1`, `ru_maxrss=18448`). No training occurred.

The correction is a qsub-only wrapper that reads the live `$JOB_ID` with
`qstat -j`, extracts and deduplicates `granted_devices=/dev/nvidiaN`, requires
exactly one physical device, exports that index, and only then enters the
unchanged main launcher. It contains no hard-coded GPU index, qdel, kill,
release, or scheduler-resource mutation. The failed log is retained; retry
uses a new log and runtime.

Retry1, job `127371058`, proved a second scheduler boundary: `qstat` granted
physical `/dev/nvidia2`, while the qsub container exposed that sole device as
local ordinal `0`. Validating `nvidia-smi -i 2` therefore exited before the
main launcher, again with no runtime or training. The wrapper now records the
single physical grant for audit, independently requires exactly one
container-visible `nvidia-smi` index, and exports that local index. It still
contains no fixed GPU ordinal.

Retry2, job `127371170`, verified the final mapping: physical GPU2 was the sole
scheduler grant and local CUDA ordinal 0 was the sole container-visible
device. The job completed all tests, preflight, 3k training, checkpoint reload,
and view-09 render on that isolated H100. The original qlogin job `127350058`
and its separate GPU1 workload remained untouched.
