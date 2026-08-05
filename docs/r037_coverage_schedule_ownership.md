# R037: Coverage Schedule Ownership

Status: deferred; never launched as a formal run and not part of the active
mainline.

## Decision

The implementation proved that schedule ownership can be separated cleanly,
but the project is not moving every coverage field onto a separate early
schedule. Frozen R036 keeps only child spread on its already measured 1k-7k
coverage ramp; width profile, length, direction, and the remaining render-root
geometry residuals keep the 10k-20k geometry unlock.

This document remains as rejected/deferred evidence so the proposal is not
silently rediscovered. Schedule specialization may be revisited only after the
hard-bound-free representation and densification behavior are fixed across
samples.

## Base And Scope

The deferred R037 proposal kept R034's accepted uncapped absolute segment allocation, R035's
hierarchical width profile, and R036's positive unbounded hierarchical child
spread. It changes no parameter domain, loss weight, learning rate,
densification threshold, image resolution, or lifecycle rule.

Its only change was schedule ownership. Four render-root controls determine
early strand coverage rather than late centerline geometry:

- root width;
- tip/root width ratio;
- width taper;
- child spread.

They now share one declared coverage-field set, one 1k-7k coverage ramp, and
one freeze exemption. Length, direction, bend, clump, and optional hair-style
controls retain the 10k-20k geometry schedule.

## Evidence For The Change

R035's formal H100 run exposed a clean ownership mismatch:

- at 9k its test composite PSNR was `23.39643`, versus R034 `24.73936`;
- root width, tip ratio, and taper remained exactly at their guide defaults
  through 9k, even though the configured coverage ramp reached one at 7k;
- after the guide field unlocked, R035 reached `30.15367` at 10k, above R034's
  matched `29.62538`.

This shows that the hierarchical decoder is viable, but its render width
profile was accidentally governed by the late geometry multiplier and late
gradient freeze. The old direct route had allowed these coverage controls to
learn before 9k. R037 restores that role without restoring direct endpoints.

## Contract

`RenderGeometryResidualField.COVERAGE_SCALAR_NAMES` is the single source of
truth for schedule classification. Effective coverage uses:

`coverage_scale = configured_residual_scale * coverage_ramp(iteration)`

The late-geometry freeze skips exactly that declared set. This prevents a
future positive coverage field from being composed with one ramp but silently
frozen by another helper.

No body-region rule, physical threshold, percentile, or sample-specific value
is introduced.

## Local Verification

- complete repository suite: `59 passed`;
- Python compilation and `git diff --check`: passed;
- with geometry ramp zero and coverage ramp one, length and direction remain
  exactly at the guide field while all four coverage controls change: passed;
- the late-geometry freeze preserves gradients for exactly the declared
  coverage set and zeros the remaining residual gradients: passed;
- direct bounded render groom fields remain outside the formal optimizer:
  passed.

Formal acceptance requires a strict from-zero 0-30k run and matched
full-resolution metrics, width/child distribution audits, lifecycle and memory
checks, and fixed V11-protocol pure-fur QA. R037 does not replace the last
accepted measured baseline before those gates pass.
