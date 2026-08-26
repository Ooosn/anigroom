# Seed Flow Brush

This optional artist-guided input path stores sparse directed image-space seed
flow. It complements automatic flow extraction and does not encode strand
length.

## Launch

```powershell
powershell -ExecutionPolicy Bypass -File scripts/windows/run_groom_flow_annotator.ps1
```

An image and output folder can be supplied directly:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/windows/run_groom_flow_annotator.ps1 `
  -InputDir D:\path\to\images `
  -OutputDir D:\path\to\flow_guidance
```

## Interaction

- `Seed` scatters points. Radius and density determine local coverage.
- `Comb` turns nearby seed directions toward the brush stroke.
- `Relax` releases manual anchors so surrounding anchors can interpolate them.
- `Erase` removes seeds.
- Manual anchors are pink; inferred followers are cyan.
- Arrow size is one global display control and is never saved as hair length.
- Space or middle-drag pans; wheel zooms; `1` through `4` select tools.

After a comb stroke, a cached K8 seed graph updates only a bounded neighborhood
of unmodified followers. Saving performs a deterministic full follower smooth
while preserving every manual anchor exactly.

## Data Contract

Each image receives `<stem>.flow.json` with schema
`anigroom.seed_flow.v1`. Every seed stores only:

```text
id, position_px, position_uv, direction_px, manual
```

There is no endpoint and no per-seed length. Existing
`anigroom.sparse_flow.v1` arrow files remain readable; their starts and unit
directions load as manual seeds, and the next save writes the new schema.
