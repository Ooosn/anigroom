# Sparse Flow Annotator

The sparse-flow annotator is the optional artist-guided input path for
AniGroom. It does not replace automatic flow extraction. Both modes produce
confidence-weighted root-to-tip anchors that feed the same 3D surface-field
fusion and multilevel-root optimization.

Launch the Panda project on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/windows/run_groom_flow_annotator.ps1
```

Or select another input and output folder:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/windows/run_groom_flow_annotator.ps1 `
  -InputDir D:\path\to\images `
  -OutputDir D:\path\to\flow_guidance
```

Each left-drag creates one directed root-to-tip arrow. The canvas supports
wheel zoom, middle-button or Space+left-button pan, right-click selection,
Delete, Ctrl+Z/Ctrl+Y, and automatic save while navigating. A confidence value
is stored with each arrow.

Each image receives a separate `<stem>.flow.json` using schema
`anigroom.sparse_flow.v1`. Pixel coordinates remain in original image space;
normalized UV coordinates and image SHA-256 are stored alongside them. The
output folder also contains `project.json`, which indexes every source image
and annotation file.

Arrow length is retained for inspection but does not define physical fur
length. Downstream guide construction consumes the root pixel, normalized
root-to-tip direction, and confidence. Physical length remains an independent
groom parameter learned and interpolated by the Stage 1 representation.

The annotation reader is `anigroom.flow_annotations`. Manual arrows are sparse
high-confidence evidence, not a dense hand-painted flow map. Unmarked regions
remain the responsibility of automatic image evidence, mesh-surface
interpolation, and local consistency.
