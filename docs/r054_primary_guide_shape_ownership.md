# R054 Primary-Guide Shape Ownership

## Status

Implementation prepared. R050 remains the accepted baseline until the formal
from-zero 30k run passes fixed RGB and structural QA.

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

## Files

- `configs/r054_primary_guide_shape_gaussian_residual_0_30k.env`
- `configs/r054_primary_guide_shape_gaussian_residual_fullres_preflight.env`
- `scripts/server/run_r054_primary_guide_shape.sh`
