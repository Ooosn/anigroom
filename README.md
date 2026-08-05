# anigroom

Editable animal grooming from mesh-rooted parametric fur.

## White Tiger Mainline

The active route is intentionally small:

- Stage 1 trains mesh-rooted fur grooming parameters.
- Stage 1 may use random colored mesh backing only to prevent white-fur transparency during reconstruction training.
- Stage 2 is not part of the current active run.
- UV maps are storage and visualization; geometric smoothness should be enforced on mesh/root neighborhoods.

Current planning documents:

- `docs/current_route.md`
- `docs/stage1_baseline_r036.md`
- `docs/r_series_evolution.md`
- `docs/anigroom_module_map.md`
- `docs/accept_line_recovery_ledger.md`

R036 is the frozen measured Stage 1 baseline. Its formal 30k result is
train/test composite PSNR `33.42397 / 32.66322`, with best test composite
`32.83977` at 29k. The only runnable contract is the from-zero
`configs/stage1_baseline.env`, launched through
`scripts/server/run_white_tiger_stage1.sh`. Verify the frozen source and input
contract with `python tools/verify_stage1_baseline.py`.
