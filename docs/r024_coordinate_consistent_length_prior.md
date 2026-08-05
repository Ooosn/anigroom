# R024 Coordinate-Consistent Length Prior

## Status

`rejected`; the formal held-H100 9k-to-30k run and late-checkpoint asset QA
are complete. R024 delays the failure seen in R023, but does not prevent it.

## Parent And Single Change

R024 inherits R023's positive, scale-relative, unbounded length decoder:

```text
effective_length = guide_length * exp(asinh(raw))
```

R023 correctly smoothed `raw` on the surface graph, but its guide prior still
regularized `tanh(raw)`. That coordinate saturates: at R023's 30k maximum
`raw=5.363`, the `tanh` derivative is nearly zero. A connected group can then
move together, evade edge smoothness, and receive almost no guide-prior
gradient. The canonical 30k render exposes this as a few long tail-tip hairs.

R024 changes only the coordinate used by the existing length prior:

```text
R023 prior: abs(tanh(raw))
R024 prior: abs(asinh(raw)) = abs(log(effective_length / guide_length))
```

The weight and schedule are unchanged. This is not an absolute length limit:
the effective length remains positive and unbounded, and coherent regional
length remains available through the guide field.

## Acceptance Gate

1. Match R023's ordinary distribution and reconstruction trajectory.
2. Eliminate the visible isolated tail-tip hairs at retained late checkpoints.
3. Keep effective length positive and free of absolute min/max clamps.
4. Preserve full-resolution test composite near the R018/R019 range.
5. Pass graph-tail and canonical 100k-strand Blender QA through 30k.

## Formal Result

Run:

```text
/home/wangyy/anigroom-r024-coordinate-prior-20260729/
  outputs/r024_coordinate_consistent_length_prior_9k_30k_20260729_h100
```

The ordinary distribution remained stable while a sparse tail escaped:

| Iteration | Test composite PSNR | Length P95 | Length max | Raw max |
|---:|---:|---:|---:|---:|
| 18k | 31.6464 | 0.03650 | 0.18590 | 4.0134 |
| 20k | 31.9012 | 0.03640 | 0.29812 | 4.1846 |
| 25k | 31.9364 | 0.03664 | 0.42849 | 5.3998 |
| 27k | 32.1109 | 0.03670 | 0.59984 | 5.8396 |
| 30k | 32.0572 | 0.03687 | 0.78007 | 5.1325 |

At 30k, the top 0.1% contains 218 roots. Their largest mesh-graph connected
component contains 81 roots, with a directed internal-edge fraction of 0.5665.
This is a coherent local escape, not a global shift in the learned length
distribution. One representative root changes from `raw=0.239` at 25k to
`raw=4.753` at 27k while its neighbors are already similarly large.

Canonical Blender QA uses 100k strands, 32 samples per strand, the same mesh,
camera, width, and material at every checkpoint. It shows isolated tail-tip
hairs appearing at 25k, increasing at 27k, and becoming clearly invalid at
30k. The visual result therefore fails gate 2 and gate 5 even though PSNR and
P95 remain competitive.

## Conclusion

`abs(asinh(raw))` uses the same semantic coordinate as the decoder, but its
gradient still decays as `1 / sqrt(1 + raw^2)`. R024 only postpones the local
escape. R025 tests the next isolated change: retain the same positive,
relative, unbounded decoder but regularize the raw zero-centered residual so
the prior gradient does not vanish in the tail.
