# R034: Uncapped Absolute Segments And Width Profile

Status: formal 0-30k run complete. The segment correction is accepted; the
direct render-root width-profile change is rejected.

## Base And Scope

- measured reference: R032 and the completed R033 30k run;
- executable parent: the strict direct-3D R033 source;
- correction: restore the accepted absolute-length/complexity segment allocator
  and remove only its upper clamps;
- new decoder change: remove the measured artificial endpoints from tip-width
  ratio and width taper;
- intentionally unchanged: root width, losses, lifecycle evidence, schedules,
  clean-flow initialization, guide/render geometry, opacity, and RGB controls.

The segment correction is treated as an implementation repair. The width
profile is the R034 method variable.

## Uncapped Segment Allocation

For physical groom length `L`, geometric strand complexity `C`, and minimum
representation `N_min`, R034 uses:

`N = max(N_min, round(N_min + rho_L * max(L - L0, 0) + rho_C * C))`

The active calibration exactly preserves the accepted pre-R033 linear formula:

- `L0 = 0.010`;
- `rho_L = 84.19047619047619`;
- `rho_C = 23.771428571428572`.

These densities are the algebraic form of the old `10..36` calibration, not a
new fitted rule. There is no `MAX_SEGMENTS`, no final upper clamp, and no
initialization-derived segment reference. Equal physical lengths receive equal
length contribution; curvature can only add representation. A memory guard
remains the runtime failure boundary rather than silently changing the model.

On the fixed R033 100k-strand export, the corrected allocator gives mean/min/max
`11.041 / 10 / 17` and `1.104M` Gaussians. R033's rejected allocator gives mean
`20.102`; R032's accepted allocator gives mean `10.928`. A synthetic regression
also proves that a sufficiently long strand receives more than the former 36
segments.

Zero-length segments are excluded from curvature statistics. This fixes the
old degenerate-strand case where zero direction vectors looked like maximal
turns.

## Width Profile Domain

The R033 checkpoint provides direct saturation evidence:

- `68.75%` of render tip-width ratios occupy the highest 1% of the old
  `[0.012, 0.30]` interval;
- `82.61%` of width-taper values occupy the highest 1% of the old
  `[0.55, 3.20]` interval;
- root width is not globally saturated and remains unchanged.

R034 therefore uses:

- `tip_ratio = sigmoid(raw)`, the semantic `[0,1]` relation between tip and root
  width, without padded animal-specific endpoints;
- `width_taper = exp(asinh(raw))`, a positive unbounded exponent with neutral
  coordinate `raw=0 -> taper=1`.

Initialization retains the existing physical values (`tip_ratio=0.07`,
`taper=1.8`). Render-root densification interpolates physical values over the
surface support and then re-encodes them in the new coordinates. It does not
interpolate old bounded raw values.

## Checkpoint Contract

R034 removes `segment_reference_length` from model state and replaces
`max_segments` in the config with explicit absolute-resolution coefficients.
Checkpoint version is 3. R033 and older configs are deliberately rejected by
the strict schema; the formal run must start from zero.

## Verification Before Training

- Python compilation: passed;
- full semantic tip-ratio endpoints: passed;
- positive unbounded taper and inverse round trip: passed;
- physical width-profile interpolation through render-root lifecycle: passed;
- uncapped absolute length and curvature allocation: passed;
- degenerate-strand curvature handling: passed;
- full repository suite: 54 tests passed.

Formal acceptance requires a complete 0-30k run, matched full-resolution
metrics, lifecycle/memory inspection, parameter saturation diagnostics, and
fixed-protocol pure-fur renders.

## Formal Result

The H100 from-zero run completed with no restart or OOM:

- final train/test composite PSNR: `33.27798 / 32.47945`;
- best test composite PSNR: `32.66906` at 29k;
- final render roots: `325,619`;
- final Gaussians: `14,268,540`;
- mean/max segments: `10.955 / 17`;
- peak allocated CUDA memory: `24,033 MB`.

Against R033, test quality is unchanged to measurement noise while Gaussian
count falls from `25.92M` to `14.27M` and peak allocated memory falls from
`29.32GB` to `24.03GB`. The absolute uncapped segment allocator is therefore
retained.

The width-profile parameter audit rejects the direct-learning part of R034:

- `62.15%` of render roots have tip/root ratio at least `0.99`;
- median tip/root ratio is `0.9933`;
- median taper is `5.871`;
- the median normalized width profile remains above `0.9986` even at 75% of
  strand length.

The canonical pure-fur render does not catastrophically collapse, but the
parameter field has become nearly cylindrical and taper is non-identifiable.
Removing endpoints alone is therefore insufficient: RGB can move every
render-root width profile coherently, which local smoothness cannot detect.
R035 keeps the segment repair and moves width-profile ownership into the
guide/render hierarchy.
