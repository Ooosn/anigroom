# R059 Redesigned Groom Geometry Training

Status date: 2026-08-13.

Status: formal from-zero H100 run prepared; results pending.

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

The formal runtime will be recorded after launch under a new immutable HGC
project checkout and runtime directory. Results and artifact hashes must be
added here before R059 can become a checkpoint.
