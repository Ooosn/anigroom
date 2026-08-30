# R075 Panda Full-Shell Data-Identity Length Gate

Status date: 2026-08-30.

Status: prepared/pending. The strict R075 config, launcher, contract tests, and
acceptance record are prepared; the from-zero H100 0-3k run and asset gate are
pending.

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
