# R027 Lifecycle Optimizer State

## Status

`accepted as the current Stage 1 baseline`.

The formal H100 run completed at 30k with request exit code 0. It preserves
R026's representation, losses, data, schedules, lifecycle evidence, and split
placement. The isolated method change is optimizer-state migration across
render/guide lifecycle updates.

## Problem

R026 extends render-root densification through the 10k-20k render-residual
unlock. Its structure update interpolates child attributes and removes each
selected parent correctly, but the trainer then rebuilds Adam without
transferring state. Every 100-iteration render split therefore clears moments
for all surviving render roots and unchanged guide parameters. Guide
densification has the same problem.

At 16k, R026 has `282095` render roots versus R025's `217036`, but test
composite falls from `31.2422` to `31.1301`. Capacity and optimizer reset are
confounded, so R026 cannot test the intended lifecycle schedule.

## Isolated Repair

1. Optimizer state is addressed by canonical parameter name.
2. Every unchanged parameter and surviving root row retains Adam state.
3. Removed parent rows are removed from state in model-row order.
4. Newly inserted child rows start with zero first/second moments.
5. Adam's scalar step is retained.
6. Render-root and guide-root updates use the same migration contract.
7. An unknown shape transition raises an error; there is no fallback.

The first formal render event at 9.1k closed exactly: `205279` old roots,
`1186` removed parents, `204093` retained rows, `2372` zero-moment children,
and `206465` new roots. Adam state was restored for 24 parameters; 15
root-indexed parameters used row migration.

## Formal Results

| Iteration | R025 test composite | R026 test composite | R027 test composite |
| ---: | ---: | ---: | ---: |
| 10k | 29.6091 | 29.6134 | 29.7848 |
| 12k | 30.2885 | 30.1307 | 30.3573 |
| 14k | 30.8239 | 30.6643 | 30.8947 |
| 16k | 31.2422 | 31.1301 | 31.3245 |
| 18k | 31.6408 | - | 31.7585 |
| 20k | 31.9013 | - | 32.0771 |
| 25k | 31.9737 | - | 32.1660 |
| 27k | 32.2278 | - | 32.5032 |
| 29k | 32.2842 | - | **32.5517** |
| 30k | 32.0910 | - | 32.3811 |

R026 was intentionally stopped at 16k after the reset confound was proved.
R027 remains above R025 at every listed matched gate. At 30k:

- train/test composite: `33.1937 / 32.3811`;
- render roots: `318532`;
- guide roots: `5332`;
- generated Gaussians: `13791691`;
- effective length P95/max: `0.036316 / 0.330164`;
- peak allocated CUDA memory: `26.23 GB`.

The final render lifecycle event occurs at 20k. Post-event root count is
`318532` and remains unchanged through 30k. Formal metrics contain 110 unique,
strictly increasing lifecycle events from 9.1k through 20k.

## Structural QA

Canonical deterministic assets at 12k, 16k, 18k, 20k, 25k, 27k, and 30k use
the same 100k child-expanded strands, 32 curve samples, mesh alignment, camera,
material, lighting, and Blender settings.

- Body, head, belly, legs, and tail retain coherent flow.
- No global curl-back, spiral, or long-line collapse appears.
- A sparse length-residual tail remains localized around the tail tip. At 27k
  one or two isolated strands are visible; they shorten by 30k.
- The length P95 remains stable while the maximum is non-monotonic, confirming
  that this is a sparse tail rather than global coat inflation.
- The 30k RGB test view's remaining errors concentrate on silhouette, cheek
  strands, and high-frequency stripe sharpness. The side/rear train view has no
  large blurred patch or depth-clipping failure.

R027 therefore fixes optimizer continuity and improves both matched PSNR and
late tail statistics, but does not claim to eliminate every sparse outlier.

## Runtime Incidents

Two runs after the verified 18k checkpoint were stopped by diagnostics, not by
the model:

1. The CUDA guard callback passed `stage` twice. The report path was fixed and
   covered by tests.
2. The guard treated `nvidia-smi` process memory as live model memory, thereby
   counting PyTorch reserved cache a second time. At the false stop, current
   allocation was about `2.6 GB`, historical allocated peak about `25.5 GB`,
   reserved cache about `30.3 GB`, and process memory about `31.1 GB`.

The accepted guard tracks:

```text
max(current allocated, historical allocated peak)
+ max(0, nvidia process memory - current PyTorch reserved memory)
```

It therefore retains external-allocation protection without treating the
allocator cache as additional live tensors. The final run passes the 30 GB
guard and remains stable after the 20k capacity boundary.

The two aborted continuations are preserved under the R027 diagnostics
directory; their metric rows were removed from the formal trajectory before
the successful resume.

## Checkpoint Integrity

The original resume code started a fresh in-memory `lifecycle_history`, so
post-18k checkpoints initially contained only events 18.1k-20k. This did not
affect weights, gradients, optimizer state, RNG, metrics, or training behavior,
but it made checkpoint metadata incomplete.

The loader now restores and validates checkpoint history: records must be
mappings, strictly increasing, and no later than the checkpoint iteration.
Local and HGC focused suites both pass `38/38` tests plus the surface-graph
regression.

The complete formal `metrics.jsonl` history exactly matches the 18k checkpoint
prefix and the post-resume checkpoint suffix. Checkpoints 20k, 21k, 22k, 24k,
25k, 27k, and 30k were atomically repaired from 20 to 110 history records.
Every non-history field was deep-compared and remained bit-identical in memory.
Original files and hashes are archived at:

`diagnostics/checkpoint_history_pre_repair_20260730`

Final 30k repaired SHA-256:

`9e6d3082001e3a54cd3e23489045d42a7b733814345e2caf3ae1b466545b0b79`

## Artifacts

- HGC output:
  `/home/wangyy/anigroom-r027-optimizer-state-20260730/outputs/r027_lifecycle_optimizer_state_9k_30k_20260730_h100`
- Local final checkpoint:
  `D:/RTS/_tmp/r027_30k_visuals/checkpoint_030000.pt`
- Local final canonical asset:
  `D:/RTS/_tmp/r027_30k_visuals/r027_030000_asset_side_y_v11_protocol.png`
- Local 30k test prediction:
  `D:/RTS/_tmp/r027_30k_visuals/view_00_eval_pred.png`
- Local 30k side/rear training prediction:
  `D:/RTS/_tmp/r027_30k_visuals/view_14_train_pred_fixed_bg.png`

The held H100 qlogin job remains allocated for the next explicit instruction.
