# AniGroom Working Rules

This project is not a smoke test. Do not optimize for quick command completion.
Every action must serve the current research goal: RGB-aware explicit animal
grooming reconstruction that can support the paper.

## Current Main Goal

AniGroom is not a generic 3DGS reconstruction project and not a PSNR-only
white-tiger fitting project. The paper story is RGB-aware explicit animal
grooming reconstruction: use RGB evidence, but prevent RGB noise, stripes,
shadows, and inter-fur appearance from corrupting fur geometry.

Before changing training logic or documentation, read:

1. `docs/research_narrative.md`
2. `docs/anigroom_module_map.md`
3. the relevant `docs/modules/*.md` file

The formal direction is explicit grooming parameters, clean 3D flow
initialization, guide/render-root separation, root lifecycle, and delayed
appearance residuals for non-grooming RGB effects. Any PSNR gain that creates
long color-dragging strands, curled/tangled fur, muddy blocks, or noisy grooming
fields is a failed route.

## Non-Negotiable Rules

0. Respect the user's specified method and scope. If the user asks for a
   specific source of truth or method, such as NeuralFur/SMAL body-part
   annotation for a head mask, do that method. Do not substitute PCA probes,
   image-space boxes, heuristic shortcuts, quick visual guesses, or unrelated
   diagnostics unless the user explicitly approves that detour first.
1. Do not use fallback implementations, silent degradation, reduced
   resolution, fake renderers, or one-step smoke tests for formal modules.
   If a formal module fails, let it fail and diagnose the real cause.
2. Before changing training logic, first produce visual evidence that the
   input signal is valid. For orientation/flow, this means readable direction
   line visualizations, not just point overlays or confidence heatmaps.
3. The accepted clean-flow `v4_surface_direction` target and its
   normal-compatible parallel-transport runtime interpolation are one atomic
   part of the current formal Stage 1 route. Keep `v3_height_smooth` only as a
   controlled rollback/ablation target; do not mix a v4 target with the old
   Euclidean runtime sampler.
4. Treat black-white tiger stripes as a known source of false orientation.
   Confidence alone is not a valid anchor criterion. Anchor selection must
   consider stripe rejection, local direction coherence, visibility, and
   eventually multi-view consistency.
5. Root visibility must not replace correct per-Gaussian depth clipping.
   Fur rendering should be clipped by mesh depth so back-side/body-hidden
   Gaussians do not contribute to forward or backward.
6. Densification/pruning must follow the confirmed root lifecycle in the formal
   config. Accumulate root evidence over 100-iteration windows, densify from
   root-level evidence, keep tensor/optimizer/root-id mappings synchronized,
   then prune only after enough visibility history.
7. Groom parameters must remain interpretable. Do not introduce carrier,
   hidden density/hairness, latent shortcuts, or unrelated appearance stories
   unless explicitly approved.
8. For white tiger Stage 1, mesh/random backing is only to prevent fur from
   exploiting transparent/white-background blending. It is not a new
   disentanglement story.
9. Do not preserve old experimental code in the formal path. Diagnostic code
   must be clearly separated from formal modules.
10. If the user points out a conceptual issue, stop and analyze that issue
    directly before writing more code.
11. The formal Stage 1 config must not default to 70k render roots, 8192 guide
    roots, 1000-iteration densification intervals, or old screen/dark/luma
    overlong heuristics.
12. The three required validation points are mandatory before claiming the new
    baseline: child strands/clump, RGB-to-flow or edge-style loss, and cleaned
    flow for guide initialization.
13. Visualization must be stable and comparable. Unless the user explicitly
    asks for a sheet, output single original-resolution images, using the same
    script, view, crop, root mode, line length, line width, and color convention
    when comparing versions. Do not change visualization code or style while
    claiming algorithmic progress.
    Every accepted training phase must be visualized through
    `anigroom.visualization.stage1`; visualization implementations belong only
    under `anigroom/visualization`, while `tools/` may contain thin CLI entry
    points. Phase comparisons must use the checked-in protocol file and may
    not override its views, resolution, sampling seed, strand count, camera,
    material, arrow style, or scalar display ranges.
14. Diagnostic code must not be presented as a formal result. If a script or
    output is exploratory, label it as diagnostic and keep it out of the formal
    path until it has passed the agreed visual and numeric checks.
15. Do not continue implementing while confused about the user's current
    request. Stop, restate the exact requested module, and inspect the relevant
    official code, old code, or documentation before writing more code.
16. Before any server/GPU action, read and follow the maintained
    `westlake-evoweave` skill. Do not use old handoff folders or remembered SSH
    state. Development environments are about 4 hours only: check remaining
    time and run a live resource audit first. Never run compute in a resource
    group whose name starts with `huangxiangru`; select an allowed A100 group.
17. Local white-tiger training must enable the CUDA memory guard unless the user
    explicitly disables it. The default local limit is 25GB process memory,
    checked with both PyTorch memory stats and current-process `nvidia-smi`
    usage. If it trips, stop and diagnose; do not raise the limit to hide a
    memory bug.

## Required Pre-Action Checklist

Before any substantive action, answer these internally:

- What is the current module being validated?
- Did the user specify the method to use, and am I following that method rather
  than replacing it with a shortcut?
- What visual evidence proves the input signal is correct?
- Is this formal code or diagnostic code?
- Am I using the confirmed config, not an old config?
- Am I relying on a fallback, lower resolution, or fake renderer?
- Are the visualizations single-image, original-resolution, and comparable to
  the previous baseline unless the user explicitly requested a sheet?
- Does the change preserve the paper story: RGB evidence is used through
  explicit grooming parameters and appearance residuals, without letting RGB
  noise corrupt fur geometry?
- If this is a local training run, is the 25GB GPU memory guard enabled?
- If this touches a server or GPU, did I first read the maintained Westlake
  skill, check dev-environment time, run the live resource audit, and select an
  allowed non-`huangxiangru` group with sufficient A100 capacity?

If any answer is unclear, do not continue with implementation. Inspect, visualize,
or ask.

## Current Module Boundaries

Use `docs/anigroom_module_map.md` as the current module map. The active split is:

- Flow / initialization: mesh alignment, SMAL head/body guide roots, multi-view
  flow projection, normal-shell lift, direction clean/flip, and consensus.
- Strand-to-Gaussian representation: explicit groom controls, strand generation,
  child strands, adaptive segments, and Gaussian conversion.
- Multi-level root / training: guide/render-root hierarchy, interpolation,
  residual unlock, losses, root movement, and densification/prune policy.
- Rendering / visualization / export: gsplat rendering, mesh depth clipping,
  random backing, stable canonical visualizations, and exports.

Densification/prune is part of the multi-level root training module. Do not
design or tune it as an isolated standalone module.

## Current Flow Target

The current accepted clean-flow target is:

`D:\petsgaussianhair\_downloads\tiger_hair_flow_36\shell_fused_smal_head500_body4000_candidate65536_headk24_bodyk12_v4_surface_direction`

`v3_height_smooth` and `v2_consensus` are retained only as controlled
rollback/ablation lines. Do not treat either as the default.

Do not replace it with older `v11`, `8192`, or ad hoc flow outputs unless the
replacement is explicitly accepted and documented in
`docs/modules/01_flow_initialization.md`.

## Metric Reporting

Composite PSNR is the primary Stage 1 PSNR. Raw RGB PSNR may be logged as a
diagnostic, but summaries, comparisons, and server status reports should lead
with composite PSNR.

## Local Environment Notes

- Canonical local project root: `D:\petsgaussianhair`. Do not use the legacy
  Documents checkout, `D:\RTS`, or any outer Codex workspace directory as the
  AniGroom project root.
- Local PowerShell entry scripts should source `scripts/local/env.ps1`, which
  sets the project root, activates `mygs`, and sets `PYTHONPATH`.
- Local training/debug environment: activate with `conda activate mygs` from
  PowerShell. The environment currently imports `torch`, `gsplat`, `trimesh`,
  and `numpy`.
- Local Blender executable for strand/asset visualization:
  `D:\Program Files\Blender Foundation\Blender 5.0\blender.exe`. It is not on
  `PATH`, so scripts should use this absolute path unless PATH is explicitly
  updated.
