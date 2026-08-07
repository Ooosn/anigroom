# R045 Secondary-Guide Local-Stencil Diagnostic

## Status

R045 is an isolated causal diagnostic built from the completed R044 candidate.
It is not a baseline and changes no accepted R043 or completed R044 file.

## Question

R044 uses 20k secondary guides, but its G1 K32 graph has a much larger physical
support than R043's render-root K32 graph:

| Graph | Edge distance P50 | Edge distance P95 |
| --- | ---: | ---: |
| R043 render K32 | 0.00649 | 0.00922 |
| R044 G1 K32 | 0.02637 | 0.04267 |

The G1 graph therefore spans 4.06x the median and 4.63x the P95 physical edge
length. This can over-low-pass a 20k field even if the representation itself
has sufficient capacity.

Checkpoint inspection rules out four alternative implementation failures:

- all nine G1 tensors have exactly 20k rows;
- every trainable G1 row has nonzero value and Adam state;
- every G1 point receives render-root support and nonzero interpolation mass;
- render roots use a median effective support of 4.05 G1 points, so K8
  interpolation has not collapsed to one global or parent value.

## Single Variable

Resume the exact formal R044 `checkpoint_010000.pt`, including optimizer and RNG
state, and run to iteration 16000. The only model-setting change is:

```text
SECONDARY_GUIDE_SMOOTH_K: 32 -> 4
```

G1 remains 20k. Render-to-G1 interpolation remains K8. All scalar loss weights,
learning rates, residual ramps, views, renderer settings, and lifecycle state
remain unchanged. Since render densification ended at 9000, this continuation
does not change root topology.

K4 is used as the smallest stable local surface stencil, not as an accepted
sample-specific hyperparameter. At G1 density its exact edge-distance P50/P95
is 0.01100/0.01500, much closer to R043 than G1 K32 while retaining a local
multi-neighbor graph.

## Gate

At the same 12k, 14k, and 16k iterations compare against the saved R044 curve:

- train/test composite PSNR and RGB L1;
- G1 residual direction and length magnitude;
- effective length distribution;
- effective, clean-flow, and residual smoothness raw values;
- peak memory and elapsed continuation time.

If useful residual magnitude and PSNR recover without a new structural
artifact, the R044 failure is caused by operator scale rather than 20k capacity.
If they do not recover, the next audit must test interpolation/basis capacity;
the G1 population must not be increased merely to hide the result.
