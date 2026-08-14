# R062 Mesh No-Penetration

Status date: 2026-08-14.

Status: completed and accepted as the current
advanced-geometry/appearance/validity baseline. R061 remains the immutable
single-variable control. The authoritative algorithm and acceptance details
are in `docs/mesh_no_penetration.md`.

## Single Variable

R062 inherits the complete R061 route and adds only:

```text
mean(ReLU(-SDF(x)) / diagonal(SDF bounds))
```

The loss covers non-root continuous-strand samples from a deterministic cyclic
block of 16,384 current render roots per step with weight 256. It contains no
body-part mask, length condition, penetration tolerance, fallback, or separate
schedule.

## Acceptance

- Native 1920x1080 H100 forward/backward preflight: passed.
- Gradient ownership: guide/secondary geometry and root barycentrics receive
  collision gradients; global translation and scale do not.
- Strict uninterrupted from-zero 30k training: passed.
- Final/best test composite: `32.19214/32.28517`, only
  `0.02243/0.01559 dB` below R061.
- Fixed eight-view mean: `33.21203`, `0.02361 dB` below R061.
- Penetrating point fraction: `0.134272% -> 0.023592%` (`82.43%` reduction).
- Penetrating-root fraction: `0.675571% -> 0.416470%` (`38.35%` reduction).
- Mean/maximum normalized penetration depth reduction: `84.59%/51.16%`.
- Matched 100k-strand audit: zero backward strands, zero lengths above `0.12`,
  and no canonical visual regression.

## Identity

```text
source commit:
  100f7223ede6975862cbc6c30b27f29709f68147
checkpoint SHA256:
  d1f23c92f68b250f00ac8771f6435c63af2baf686a9329696de3b34c0cc72900
SDF SHA256:
  766e177fbeeb89fc779292f56662c7c6b256f7d4365415baa366cef04af10530
formal output:
  /home/wangyy/anigroom-r062-no-penetration-runtime-20260814-v2/outputs/r062_mesh_no_penetration_0_30k_h100_20260814
local QA:
  D:/RTS/_tmp/r062_acceptance_20260814
```

R062 is frozen after this acceptance. Crossing/intersection work starts from
R062 as a separate experiment and must not silently change this result.
