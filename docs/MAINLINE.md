# AniGroom White Tiger Mainline

This repository is currently reduced to the white-tiger asset path only.

## Active Goal

Train a displayable and editable white-tiger fur asset:

- Stage 1 trains mesh-rooted parametric fur.
- Stage 1 may use random colored mesh backing only to avoid white-fur transparency.
- Stage 2 adds the real mesh surface texture.
- UV maps are storage and visualization; smoothness should be enforced on mesh/root geometry.

## Active Entry Points

- `scripts/run_white_tiger_stage1_reconstruction.sh`
- `scripts/run_white_tiger_stage2_surface_texture.sh`
- `scripts/run_white_tiger_formal_asset_pipeline.sh`
- `scripts/train_white_tiger_uv_groom_server.sh`

## Active Tools

- `tools/train_white_tiger_uv_groom.py`
- `tools/render_white_tiger_uv_groom_orbit.py`
- `tools/generate_orientation_maps.py`
- `tools/make_video_from_frames.py`

## Archive

Older experiments, reports, temporary leader demos, server pulls, and candidate
scripts are archived under:

`_archive/cleanup_20260619_mainline/`

They are not part of the active route.
