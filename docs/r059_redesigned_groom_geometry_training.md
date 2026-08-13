# R059 Redesigned Groom Geometry Training

Status date: 2026-08-13.

Status: formal from-zero H100 run completed and accepted as the trained
advanced-geometry research checkpoint. R050 remains the strict zero-foldback
structural/appearance reference, and R043 remains the default
structural/lifecycle baseline.

## Question

R058 replaced the invalid legacy curl/frizz forward geometry, but deliberately
did not train it. R059 asks one question: under the complete accepted R057
training contract, does the redesigned physical strand geometry learn a
cleaner optional shape field without sacrificing the established appearance
handoff?

## Single Variable

R059 inherits R057 without changing any behavior assignment:

- frozen data, mesh alignment, clean-flow initialization, and full resolution;
- 4500 primary guides, 20k secondary guides, and 400k initial render roots;
- typed surface interpolation and both smoothing graphs;
- render-root lifecycle, density, evidence, placement, and optimizer migration;
- all RGB, flow, anchor, smoothness, and appearance losses;
- all unlock schedules, weights, learning rates, and gradient ownership;
- Gaussian RGB residual and the primary/secondary shape handoff.

The only method variable is the R058 source geometry:

- physical curl radius and turns around the transported backbone frame;
- persistent non-trainable per-root frizz seed;
- independent band-limited frizz;
- additive curl/frizz offsets evaluated around the accepted brush backbone.

R057 shape ownership is also unchanged: the primary and secondary guide fields
learn curl radius and frizz amplitude, while render-root curl turns and phase
remain fixed semantic coordinates. Learning those coordinates is a separate
experiment and is not silently mixed into R059.

The executable configuration is
`configs/r059_redesigned_groom_geometry_0_30k.env`. It only sources R057 and
contains no behavior override.

## Checkpoint Boundary

R057 checkpoints contain the retired curl-frequency schema and no persistent
frizz seed. R059 contains `curl_turns_raw` and `frizz_seed_phase`. Strict state
loading rejects the retired schema; there is no alias, migration, non-strict
load, or resume. The formal R059 run starts from iteration zero.

Render-root lifecycle transports the periodic frizz seed to inserted roots and
preserves retained rows. The seed is checkpoint state but never an optimizer
parameter. Focused tests and the full-resolution active-path preflight verify
those properties before the 30k run starts.

## Formal Protocol

- preflight: full-resolution two-step forward/backward/eval with every staged
  geometry and appearance path active;
- training: uninterrupted from-zero 0-30k with the R057 schedule;
- intermediate checkpoints: the inherited 9k, 10k, 12k, 14k, 16k, 18k, 20k,
  22k, 25k, 27k, and 30k boundaries;
- postprocess: eight fixed full-resolution RGB views, view-09 attribute maps,
  and one 100k-strand child-1 canonical export;
- no fallback, resolution reduction, checkpoint migration, or substitute
  renderer.

## Acceptance Criteria

R059 is not accepted merely because it finishes or raises PSNR. The result must
be judged against R057 and the strict R050 structural reference using:

1. full-resolution train/test composite PSNR and the fixed eight-view mean;
2. Gaussian-residual-on/off contribution to verify appearance still absorbs
   high-frequency evidence;
3. pure-fur local direction, relative length continuity, arc/chord ratio,
   local turning, and backward-segment statistics;
4. large single-image inspection of head, shoulder, belly, hip, legs, and tail;
5. lifecycle completion, root/Gaussian population, peak memory, strict reload,
   and optimizer-state integrity.

The intended success is a physically understandable curl/frizz field with less
foldback and stripe-correlated shape noise. Reconstruction is a secondary
constraint, not the sole objective.

## Runtime

The formal execution uses:

- clean project checkout:
  `/home/wangyy/anigroom-r059-groom-geometry-20260813`;
- isolated runtime:
  `/home/wangyy/anigroom-r059-groom-geometry-runtime-20260813`;
- formal output:
  `/home/wangyy/anigroom-r059-groom-geometry-runtime-20260813/outputs/r059_redesigned_groom_geometry_0_30k_h100_20260813`.

## Formal Result

The isolated H100 run completed from zero through 30k without resume,
fallback, resolution reduction, or schema migration.

| statistic | R057 | R059 | R059 - R057 |
| --- | ---: | ---: | ---: |
| final train composite PSNR | 33.39819 | 33.38828 | -0.00991 |
| final test composite PSNR | 32.27096 | 32.25537 | -0.01559 |
| best test composite PSNR | 32.35660 | 32.33647 | -0.02013 |
| fixed eight-view composite mean | 33.33974 | 33.32501 | -0.01473 |
| render roots | 473705 | 474054 | +349 |
| generated Gaussians | 5703472 | 5535197 | -168275 |

The final checkpoint was written at iteration 30000 after `13351.595 s`.
Peak live CUDA allocation was `21938.24 MB`; the corresponding live process
measurement was `25810 MB`. The small reconstruction delta is secondary to
the structural question and does not indicate an appearance-path collapse.

## Appearance Handoff

Across views 0, 5, 9, 14, 18, 21, 27, and 32:

| fixed eight-view statistic | R057 | R059 |
| --- | ---: | ---: |
| full composite mean | 33.33974 | 33.32501 |
| mean without Gaussian RGB residual | 31.69299 | 31.71564 |
| mean gain from Gaussian RGB residual | +1.64675 | +1.60937 |
| Gaussian residual parameter RMS | 0.08189 | 0.07893 |
| residual saturation fraction | 0.01897 | 0.01820 |
| mean shape-detail image magnitude | 0.002075 | 0.001694 |

The Gaussian RGB residual still contributes `+1.60937 dB`, while its RMS and
saturation both decrease. The redesigned optional geometry therefore does not
take the high-frequency appearance role away from the Gaussian residual. Its
rendered shape-detail magnitude is `18.4%` lower than R057, consistent with a
less aggressive deformation field.

## Matched Strand Audit

The canonical audit uses the same deterministic child-1 export of 100k roots,
32 samples per strand, seed 29, and local 4-NN protocol for R050, R057, and
R059. It is now executable through `tools/audit_strand_structure.py` rather
than a per-run temporary script.

| statistic | R050 | R057 | R059 |
| --- | ---: | ---: | ---: |
| local relative-length mean | 0.02047 | 0.02204 | 0.02125 |
| local relative-length P95 | 0.07741 | 0.08292 | 0.07919 |
| local chord-direction mean | 3.8282 deg | 3.8444 deg | 3.9465 deg |
| local chord-direction P95 | 11.2959 deg | 11.3640 deg | 11.4754 deg |
| strands with a backward segment | 0 | 177 | 34 |
| backward-segment fraction | 0 | 0.0003389 | 0.0000652 |
| arc/chord P95 | 1.00673 | 1.10047 | 1.06190 |
| arc/chord P99 | 1.02534 | 1.20297 | 1.14865 |
| maximum-turn P95 | 0.9546 deg | 50.3090 deg | 10.2654 deg |
| maximum-turn P99 | 2.4301 deg | 68.6494 deg | 17.6356 deg |
| maximum-turn max | 3.1876 deg | 146.4653 deg | 56.1627 deg |
| maximum arc length | 0.12852 | 0.12754 | 0.10805 |

Relative to R057, R059 reduces backward strands by `80.8%`, maximum-turn P95
by `79.6%`, maximum-turn P99 by `74.3%`, and local length discontinuity by
`3.6%` in the mean and `4.5%` at P95. Direction continuity remains effectively
matched but is slightly worse, not improved. R059 also lowers generated
Gaussian count by `2.95%` because the redesigned curves require less adaptive
complexity.

The 34 remaining backward strands are genuine compact hooks or loops rather
than numerical threshold noise. Their roots form one spatially coherent patch
on the head crown: all 34 connect under a 0.02 world-space radius, 29 connect
under 0.01, and no highlighted strand appears on the torso, belly, legs, hip,
or tail. This concentration suggests a shared low-frequency shape-field issue,
but the full-model audit alone does not assign it to the primary or secondary
field. No body-region rule or sample-specific suppression is added.

## Lifecycle And Checkpoint Integrity

- all 85 uncapped render-root lifecycle events ran every 100 iterations from
  600 through 9000;
- the final event produced exactly 474054 roots, matching the final checkpoint;
- lifecycle statistics are inactive at 30k, so no hidden post-9k accumulation
  remains;
- the strict 30k checkpoint is schema version 6, kind `stage1_full`, and
  contains all 85 lifecycle records and RNG state;
- all 26 optimizer parameters have state; all 78 optimizer tensors are finite;
- Adam first moments are nonzero for primary guide direction, stiffness,
  curl, frizz, secondary direction/curl/frizz, root/tip color, and generated
  Gaussian RGB residual;
- fixed curl turns and persistent frizz seed remain outside the optimizer;
- three independent postprocess programs strictly reloaded the final
  checkpoint and produced RGB views, attribute maps, and the 100k-strand
  export.

## Decision

R059 accepts the R058 geometry as the trained advanced-geometry research
checkpoint and replaces R057 as the parent for subsequent optional-shape work.
It does not replace R050's strict zero-foldback structural/appearance reference
or R043's default structural/lifecycle baseline. The next structural question
is the remaining compact head-crown patch, not a new schedule, a global
curl/frizz disable, or a sample-specific clamp.

## Frozen Evidence

- source commit: `78d589769db656edec60a47e7e72c0cda40430c9`;
- final checkpoint SHA-256:
  `4b7bd141069b86b08c0518a72cf00fa8e6fd3310263beefe690f6fd5ad4ab91b`;
- R059 config SHA-256:
  `2612c1df6c397f94f828eea33284b16948fac082805129a120417a19950f58a7`;
- fixed-view report SHA-256:
  `b577359732f6f65fe000b55565a40d825fb6ac764a7eab1a343ca53ab34879bc`;
- 100k-strand export SHA-256:
  `d46481914fb9695a5fcf7e7cd5350516a8bfc537d305a33a863d6530455f039c`;
- local canonical assets:
  `D:/RTS/_tmp/r059_h100_postprocess_20260813/postprocess/r059_redesigned_groom_geometry/assets`;
- local foldback highlights:
  `D:/RTS/_tmp/r059_h100_postprocess_20260813/postprocess/r059_redesigned_groom_geometry/foldback_highlights`;
- local rigid planar gallery:
  `D:/RTS/_tmp/r059_h100_postprocess_20260813/postprocess/r059_redesigned_groom_geometry/foldback_gallery/r059_foldback_all34_planar_4k.png`;
- matched R050/R057/R059 audit:
  `D:/RTS/_tmp/r059_h100_postprocess_20260813/postprocess/r050_r057_r059_strand_audit_canonical.json`.
