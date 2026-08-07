# R050 Gaussian-Level RGB Residual

## Status

Implementation and verification in progress. R049 is the structural parent;
R043 remains the RGB metric control.

## Method Question

R049 proves that a 20k secondary geometry field removes most local length
patchiness and geometric texture fitting, but it remains about 1.1 dB below
R043. R050 tests whether the missing high-frequency RGB evidence can be routed
through appearance rather than returned to length or direction.

## Single Variable

Keep R049's geometry, 20k secondary guides, K8 render interpolation, K4
secondary smoothness, lifecycle, losses, optimizer rates, random mesh backing,
resolution, and schedules unchanged. Add one appearance field:

```text
base Gaussian RGB
  = interpolated root/tip color
  + existing local render-root color

final Gaussian RGB
  = clamp(base Gaussian RGB
          + scheduled_scale * Gaussian RGB profile(sample position), 0, 1)
```

Each render root owns a 36-control RGB profile over normalized strand arc
length. Every generated Gaussian samples the profile at its own segment
midpoint. This is per-Gaussian under the active discretization while remaining
well-defined when adaptive segment counts change. It is not a constant
per-strand color and is not a dynamically resized Parameter tied to a transient
flattened Gaussian list.

The decoded residual is `tanh(raw) * 0.20 * multiplier`. The multiplier is an
exact zero through 10k and ramps to one at 20k using the same schedule function
as the existing handoff controls. When it is zero, residual evaluation is
skipped, so no gradient or Adam state is allocated before activation.

New render roots created during lifecycle receive a zero RGB profile. They do
not inherit the parent's image noise. Surviving rows and their Adam state are
migrated exactly. The candidate fails fast unless `child_count=1`.

No L2, TV, or strand smoothness is applied to this residual: its purpose is to
store the high-frequency appearance that the smooth groom should not absorb.
It is view-independent RGB in this first controlled experiment; SH or a
decoder would introduce a second variable.

## Acceptance Gate

1. Zero profile and zero multiplier are bit-exact no-ops.
2. Gradients reach only the two arc-length controls supporting each Gaussian.
3. Adaptive segment counts preserve normalized-position semantics.
4. Lifecycle keeps surviving rows/state and initializes new rows to zero.
5. Old support-off checkpoints still load strictly; R050 checkpoints roundtrip.
6. One-H100 30k training completes without fallback, OOM, or topology change.
7. Compare full-resolution composite PSNR and per-view diffs with R049/R043.
8. Repeat the fixed 100k-strand structural QA. RGB recovery is rejected if it
   restores R043-style length discontinuity, long strands, or curl-back.

## Result

Formal H100 result pending. Local implementation verification completed:

- `109 passed` in the full test suite;
- Python compilation and launcher syntax passed;
- launcher dry-run resolves the R049 geometry contract with `child_count=1`,
  secondary-guide K4, and the exact 10k-to-20k Gaussian RGB ramp;
- pre-R050 support-off checkpoint configuration defaults remain strict and
  residual-free;
- residual statistics are computed exactly in root chunks, avoiding a full
  decoded-profile temporary allocation during evaluation.

Before the formal run, the explicitly non-reportable
`configs/r050_gaussian_rgb_residual_fullres_preflight.env` executes one real
1920x1080 view09 optimizer step with the full 400k-root model and residual
multiplier one. Its sole purpose is to verify CUDA forward/backward, nonzero
profile gradients/state, checkpoint reload, and peak memory before committing
the held H100 to 30k.
