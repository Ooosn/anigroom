# R054 Primary-Guide Shape Ownership

## Status

Formal from-zero 30k validation completed. R054 is retained as the next
shape/appearance-handoff checkpoint, while R050 remains the strict structural
reference. R054 does not replace R050 as a fully disentangled final baseline.

## Evidence From R053

R053 proves that Gaussian RGB residual is useful: it improves final test
composite by `+0.6771 dB` over the matched residual-disabled control and reduces
backward strands from `762` to `560`. It nevertheless fails structural
acceptance. Both branches learn tiger-stripe-correlated curl, and the treatment
raises mean/P95 frizz while worsening maximum local-turn P95.

The failure exposes an ownership problem. R053 gives curl/frizz to both the
4,500-root primary guide field and the 20,000-root secondary geometry residual.
The second field is dense enough to follow local RGB texture, so Gaussian RGB
residual is not the only high-frequency outlet.

## Single Change

R054 keeps the complete R053 treatment except for two assignments:

```text
GUIDE_CURL_RESIDUAL_SCALE=0.0
GUIDE_FRIZZ_RESIDUAL_SCALE=0.0
```

Consequently:

- primary guides remain the sole owners of curl radius and frizz amplitude;
- those guide fields remain trainable and use the existing surface-graph
  smoothness;
- their initial physical amplitude equals the guide root-width reference. This
  is a scale-consistent near-neutral initialization rather than the saturated
  lower range endpoint; the shape gate still renders exact zero through 10k;
- render roots receive curl/frizz only by the existing physical surface
  interpolation from primary guides;
- secondary-guide curl/frizz residual tensors remain at the semantic neutral
  value, are absent from the optimizer, and cannot affect rendered geometry;
- length, direction, width profile, child spread, lifecycle, losses, root
  counts, interpolation K, resolution, and all schedules remain unchanged;
- Gaussian RGB residual remains active on the same 10k-to-20k ramp as shape
  detail.

This is an attribute-ownership test, not a smaller learning rate, a stronger
penalty, a physical clamp, or an animal-specific threshold.

## Required Validation

1. The full-resolution active-path preflight must show nonzero finite Adam
   state for primary-guide curl, primary-guide frizz, and Gaussian RGB
   residual.
2. Secondary-guide curl/frizz residuals must be absent from optimizer ownership.
3. The candidate must train from zero through 30k on one H100.
4. Compare final and best composite PSNR against R050 and R053 treatment.
5. Render the same fixed eight RGB views and same-checkpoint shape/residual
   ablations.
6. Export the deterministic child1 100k-strand field and render the same three
   canonical Blender views.
7. Audit backward strands, local relative-length tails, arc/chord ratio, local
   turn tails, curl/frizz distributions, and attribute maps.

R054 can replace R050 only if curl/frizz remain visibly coherent and structural
tails stay near the R050 field while Gaussian residual retains useful RGB
evidence. PSNR alone is not sufficient.

## Active-Path Preflight

The first full-resolution preflight exposed a real initialization defect rather
than a training-schedule problem. Primary-guide curl/frizz were initialized at
the saturated lower endpoint of their sigmoid decoder, leaving their Adam
updates effectively zero (`3.10e-10` curl and `9.81e-12` frizz). The physical
initial amplitude was changed to the scale-consistent guide-width reference;
the existing shape gate still renders exact zero through iteration 10k.

After the repair, the same full-resolution forward/backward produced finite
nontrivial Adam updates:

- primary-guide curl: `1.859e-7` maximum update;
- primary-guide frizz: `2.415e-8` maximum update;
- Gaussian RGB residual: `2.870e-7` maximum update;
- secondary-guide curl/frizz: absent from optimizer ownership;
- peak allocated CUDA memory: about `7.9 GB` for the preflight step.

The complete local suite passed (`113 passed`) before the formal run.

## Formal 30k Result

The run started from zero on one held H100 and completed without resume,
fallback, parameter changes, or OOM. Densification ended at iteration 9k; the
shared shape/appearance handoff ramp ran from 10k to 20k.

```text
final train composite: 33.564434
final test composite:  32.369957
best test composite:   32.437790 at 29k
render roots:          469,893
generated Gaussians:   5,865,061
peak allocated CUDA:   16,254 MB
elapsed:               9,244 s (about 2 h 34 min)
```

Relative to R050, R054 gains `+0.24885 dB` final test and `+0.22843 dB`
best test. Relative to the R053 treatment, it gives back `0.16624 dB` final
test and `0.16257 dB` best test while improving the structural tails.

Across the fixed eight-view postprocess, Gaussian RGB residual raises mean
composite PSNR from `31.28880` to `33.47170` (`+2.18290 dB`). Its mean rendered
absolute contribution is `0.00534`. The no-residual images retain the tiger's
main white/black color organization and stripe layout; the residual primarily
sharpens local stripe boundaries, dark regions, and high-frequency appearance.
It is useful, but it is not yet a perfectly isolated noise-only channel.

## Structural Audit

All values below use the same deterministic child-1, 100k-strand, 32-sample
protocol.

| Metric | R050 | R053 treatment | R054 |
| --- | ---: | ---: | ---: |
| Local 4NN relative-length mean | 0.02047 | 0.02632 | 0.02417 |
| Local 4NN relative-length P95 | 0.07741 | 0.09376 | 0.09122 |
| Backward strands | 0 | 560 | 375 |
| Arc length above 0.12 | 5 | 9 | 0 |
| Arc/chord P95 | 1.00673 | 1.19097 | 1.15280 |
| Maximum local-turn P95 | 0.9546 deg | 58.8473 deg | 57.2972 deg |

Fixed canonical renders confirm the numeric result. Compared with R053,
R054 removes the obvious large curl-back/long-hair failures and makes the body,
belly, and tail field visibly more coherent. R050 is still substantially
cleaner in local turning. R054 retains sparse top/head outliers and several
shoulder/hip direction transitions.

Primary-guide curl/frizz remain small relative to mean strand length, but their
maps are still partly correlated with image stripes. On visible view09 roots,
mean/P95 curl radius is `0.000711/0.002272`, mean/P95 frizz is
`0.000548/0.001633`, and mean strand length is `0.024026`. Removing dense
secondary ownership reduces the failure but does not by itself make shape and
appearance fully independent.

## Decision

R054 passes as a useful shape/appearance-handoff checkpoint because it preserves
a real Gaussian-residual appearance gain while recovering much of the structure
lost by R053. It does not satisfy the stronger requirement of R050-like local
turn tails, so R050 remains the structural reference and rollback point.

The next method question is now isolated: curl/frizz should receive structural
evidence without receiving unrestricted RGB pressure. Do not tune an
animal-specific curl cap or add another dense residual field. Any continuation
must keep Gaussian RGB residual as the appearance outlet and test loss/gradient
ownership as one controlled variable.

## Files

- `configs/r054_primary_guide_shape_gaussian_residual_0_30k.env`
- `configs/r054_primary_guide_shape_gaussian_residual_fullres_preflight.env`
- `scripts/server/run_r054_primary_guide_shape.sh`
- H100 output:
  `/home/wangyy/anigroom-r054-primary-guide-shape-runtime-v2-20260809/outputs/r054_primary_guide_shape_gaussian_residual_0_30k_h100_20260809`
- local fixed QA: `D:/RTS/_tmp/r054_h100_postprocess_20260809`
