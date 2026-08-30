# Flow / Initialization Code Space

This package owns clean-flow target loading, interpolation, 3D direction control
conversion, and clean-flow losses.

Current accepted line:

`baseline_inputs/v7_surface_direction/guide_flow3d_shell_targets_exclude_004_024_025.npz`

The opt-in V8 pre-training candidate adds canonical confidence-guided directed
propagation after V7's guarded ratio refit. Its formal implementation is
`confidence_guided_direction.py`; V7 remains accepted until formal Panda/white
target and visual gates pass.

Retained baselines:

`D:\petsgaussianhair\_downloads\tiger_hair_flow_36\shell_fused_smal_head500_body4000_candidate65536_headk24_bodyk12_v3_height_smooth`

`D:\petsgaussianhair\_downloads\tiger_hair_flow_36\shell_fused_smal_head500_body4000_candidate65536_headk24_bodyk12_v2_consensus`

See `docs/modules/01_flow_initialization.md` for the full module boundary.

Directed 3D flow is always root-to-tip. Formal interpolation requires source
and query surface normals, parallel-transports directions into the query tangent
frame, and must never reuse unsigned-axis sign alignment.
