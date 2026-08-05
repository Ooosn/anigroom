# R038 Brush Curve And Finite Render Lifecycle

Status: implemented and locally verified; formal from-zero training pending.

## Question

Can the model represent ordinary brushed fur without using legacy lateral bend
as the base shape, while ending render-root densification cleanly at 9k instead
of paying lifecycle cost through the late appearance/shape stages?

R036 remains the frozen measured baseline. R038 is a strict-schema from-zero
candidate and does not load an R036 checkpoint.

## Representation Change

For root `P0`, outward normal `n`, normalized learned 3D endpoint direction
`d`, straight length `L`, and guide-owned brush strength `c in [0,1]`:

```text
delta = L d
delta_n = dot(delta, n) n
delta_t = delta - delta_n
F_n(s,c) = s + c s(1-s)
F_t(s,c) = s - c s(1-s)
B(s) = P0 + F_n delta_n + F_t delta_t
```

`c=0` is the exact straight root-to-tip segment. Increasing `c` accumulates
normal displacement earlier and tangent displacement later. Root and tip are
unchanged, so length remains the straight endpoint distance and 3D direction
retains its existing meaning.

`brush_curve_strength` belongs only to guide roots. Render roots receive the
intrinsically interpolated guide value; no render-root brush residual exists.
It uses the same guide graph smoothing and optimizer lifecycle as the other
guide-owned base fields.

The optional bend is now an unbounded signed, non-periodic interior offset:

```text
w(s) = 16 s^2 (1-s)^2
P(s) = B(s) + L bend w(s) side
```

The envelope and its first derivative vanish at both endpoints. Bend therefore
does not change the root, tip, or endpoint tangents and cannot duplicate the
endpoint direction. The legacy `tanh/atanh` bend interval has been removed from
guide controls, render residuals, lifecycle interpolation, and priors.

Curl and frizz stay disabled in R038. They are not silently replaced or tuned.

## Gaussian Allocation

Strands are constructed first. Existing adaptive allocation then measures the
final arc and turning complexity, so a genuinely curved brush strand receives
more samples than the same straight endpoint segment. No maximum segment count
or animal-specific length threshold is introduced.

## Lifecycle Change

R038 uses one render-root lifecycle only:

- warmup starts at iteration 600;
- one evidence-driven event every 100 iterations;
- the final event is iteration 9000;
- guide-root densification is disabled for this candidate;
- pruning remains disabled.

Iteration 9000 still retains per-Gaussian/root gradients, adds the final
window, and applies the event. From iteration 9001 onward the trainer no longer
retains lifecycle-only gradients, builds residual evidence, or accumulates
root visibility/evidence statistics. This is a code-path stop, not only a
schedule value that leaves hidden work active.

The trainer emits `lifecycle_statistics_state` at the transition and records
`lifecycle_statistics_active` in evaluation metrics, so the 9000/9001 boundary
is verified from the formal run rather than inferred from the config.

The score threshold, visibility/contribution evidence, local-max parent
selection, surface child placement, attribute interpolation, and optimizer
state migration are unchanged from the accepted implementation. This isolates
the representation and lifecycle horizon rather than retuning density to one
animal.

## Candidate Configuration

`configs/r038_brush_curve_0_30k.env` differs from frozen R036 only in:

- `DENSIFY_UNTIL: 20000 -> 9000`;
- guide densification is disabled;
- iteration 9000 is added to stage checkpoints;
- candidate comments identify the strict from-zero route.

All image resolution, root count, child count, evidence thresholds, losses,
learning rates, renderer settings, and memory guard values remain unchanged.

## Local Verification

- full repository tests: `73 passed`;
- R036 lock is verified against Git tag `stage1-r036`, not the mutable candidate
  worktree;
- tests cover exact straight behavior, fixed endpoints, gradient flow,
  normal-first/tangent-later motion, smooth interior bend, adaptive sampling,
  guide ownership, strict lifecycle migration, and the 9000/9001 boundary;
- Python compilation and launcher/config preflight must pass before launch.

## Formal Acceptance

R038 is accepted only after a complete 0-30k run records:

1. full-resolution train/test composite metrics;
2. render-root and Gaussian population through the final 9k event;
3. proof that lifecycle statistics are inactive after 9k;
4. canonical single-image, 100k-strand pure-fur renders at 9k and 30k;
5. brush-strength, length, direction, and bend maps from the canonical groom
   visualizer;
6. no new loop, inward fold, sparse spike, width collapse, or local blur.

Until those checks pass, `stage1-r036` remains the baseline.
