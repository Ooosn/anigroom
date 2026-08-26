# V5 Surface-Direction Target

This directory contains the accepted clean-flow input after adding the
parameter-free directional-observability correction to the V4 parent.

- file: `guide_flow3d_shell_targets_exclude_004_024_025.npz`
- SHA-256: `fd13c000d5643387f5d364eea3a1a38a41b579ac6e7cbe32b65c7b4d79a9fdb6`
- excluded source views: `004`, `024`, `025`
- guide layout: SMAL head `500`, body `4000`, candidate pool `65536`
- clean neighborhoods: head `K=24`, body `K=12`
- parent: `v4_surface_direction`

V5 changes only initial tangent-axis evidence. A camera view contributes in
proportion to how stably its projected tangent basis can explain that specific
2D flow axis. Direction/lift fitting, sign cleaning, consensus, shell settings,
root positions, and the training interface are unchanged.

The complete white-tiger rerun preserves all `4407` observed roots and changes
the median final direction by `0.99` degrees. The Panda cross-sample rerun uses
the same code and preserves `4194` observed roots.

V4 remains an immutable rollback input. Replacing V5 requires another named
flow-module revision and matched cross-sample validation.
