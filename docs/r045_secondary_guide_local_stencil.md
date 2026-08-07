# R045 Secondary-Guide Local-Stencil Diagnostic

## Status

Completed and rejected as a promotion. R045 is an isolated causal diagnostic
built from the completed R044 candidate. It is not a baseline and changes no
accepted R043 or completed R044 file.

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

## Formal Result

Run location:

```text
/home/wangyy/anigroom-r045-secondary-guide-runtime-20260807/outputs/r045_k4_resume10k_16k_h100_20260807
```

The run used one H100, completed without fallback, OOM, lifecycle changes, or
root-topology changes, and peaked at 10.60 GB allocated memory.

| Iteration | R043 test composite | R044 K32 | R045 K4 | K4 - K32 | K4 - R043 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12000 | 30.7874 | 30.5793 | 30.5920 | +0.0127 | -0.1954 |
| 14000 | 31.3715 | 30.8477 | 30.8607 | +0.0130 | -0.5108 |
| 16000 | 31.7395 | 31.0366 | 31.0501 | +0.0135 | -0.6894 |

The smaller graph did release direction variation, but not reconstruction
quality:

| Iteration | R044 direction P95 | R045 direction P95 | R044 length P95 | R045 length P95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12000 | 0.0212 | 0.0246 | 0.0145 | 0.0144 |
| 14000 | 0.0334 | 0.0397 | 0.0216 | 0.0224 |
| 16000 | 0.0418 | 0.0567 | 0.0325 | 0.0334 |

At 16000, the raw effective/flow smoothness values fell from R044's
0.07112/0.01065 to 0.02399/0.00358. Despite that large operator-scale change,
test composite improved by only 0.0135 dB.

## Conclusion

The fixed K32 graph is physically too broad and should not be described as a
density-independent secondary-field operator. However, it is not the principal
cause of R044's PSNR gap. Twenty thousand G1 nodes are all active, all receive
render-root support, and all have optimizer state; reducing the explicit
smoothing span does not recover the missing fit.

The next causal test must separate two remaining mechanisms:

1. explicit G1 geometry regularizers suppressing the field even with a local
   graph;
2. the fixed K8 render-to-G1 interpolation basis averaging/cancelling local RGB
   gradients before they reach G1.

Do not infer from R045 that 20k is spatially sparse, and do not increase G1
count before measuring the interpolation basis's representational upper bound.
