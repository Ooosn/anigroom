# R051 Primary-Guide Color Decomposition

## Question

Can the structured root/tip base color be reduced to the 4,500 primary guide
roots while the accepted Gaussian RGB residual remains the only high-frequency
appearance outlet?

## Controlled Change

R051 branched from R050. Geometry, root lifecycle, losses, resolution, and the
Gaussian residual schedule were unchanged. The old render-root/local color path
was replaced by root/tip colors on the primary guide set. Those base colors
learned through iteration 10,000 and then froze without Adam momentum updates.

## Result

The experiment was stopped at 10,000 because the structured base had already
failed its acceptance gate.

| Iteration | R050 test composite | R051 test composite | R051 - R050 |
|---:|---:|---:|---:|
| 1 | 17.75204 | 17.62229 | -0.12975 |
| 1,000 | 20.95175 | 20.60291 | -0.34884 |
| 3,000 | 21.57691 | 21.15748 | -0.41943 |
| 6,000 | 21.93293 | 21.47092 | -0.46201 |
| 9,000 | 22.14972 | 21.67379 | -0.47593 |
| 10,000 | 29.86556 | 26.83279 | -3.03277 |

Root lifecycle matched the control at 9k (`469,183` versus `468,529` roots),
so the failure is not missing geometry capacity. The fixed full-resolution
view09 render is blurred and carries broad cross-region stripe smearing. A
4,500-node field is too sparse to own the white-tiger's mid-frequency color
regions. Allowing the Gaussian residual to reconstruct that missing base would
repeat the color entanglement the experiment is intended to remove.

## Decision

Reject R051. Preserve it only as experimental evidence. Do not merge its
primary-guide color representation into the accepted line. R052 starts again
from R050 and tests the existing 20k secondary guide layer instead.

Local evidence:

```text
D:\RTS\_tmp\r051_checkpoints\checkpoint_010000.pt
D:\RTS\_tmp\r051_10k_view09\view_09_pred.png
D:\RTS\_tmp\r051_10k_view09\view_09_gt.png
```
