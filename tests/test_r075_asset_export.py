from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "scripts" / "server" / "export_panda_r075_assets.sh"
QSUB = ROOT / "scripts" / "server" / "export_panda_r075_assets_qsub.sh"


def test_r075_asset_export_has_explicit_identity_and_output_guards() -> None:
    source = MAIN.read_text(encoding="utf-8")

    for fragment in (
        ': "${PROJECT_ROOT:?R075 asset export requires PROJECT_ROOT}"',
        ': "${EXPECTED_SOURCE_COMMIT:?R075 asset export requires EXPECTED_SOURCE_COMMIT}"',
        ': "${CHECKPOINT:?R075 asset export requires CHECKPOINT}"',
        ': "${EXPECTED_CHECKPOINT_SHA256:?R075 asset export requires EXPECTED_CHECKPOINT_SHA256}"',
        ': "${ASSET_OUTPUT_ROOT:?R075 asset export requires ASSET_OUTPUT_ROOT}"',
        ': "${CUDA_VISIBLE_DEVICES:?R075 asset export requires CUDA_VISIBLE_DEVICES}"',
        'actual_commit="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"',
        'git -C "$PROJECT_ROOT" status --porcelain=v1 --untracked-files=all',
        'actual_checkpoint_sha256="$(sha256sum "$CHECKPOINT" | awk \'{print $1}\')"',
        '[[ "$actual_checkpoint_sha256" == "$EXPECTED_CHECKPOINT_SHA256" ]] ||',
        'iteration != 3000',
        '[[ ! -e "$ASSET_OUTPUT_ROOT" ]] ||',
        'checkpoint_iteration',
    ):
        assert fragment in source


def test_r075_asset_export_uses_the_exact_deterministic_asset_protocol() -> None:
    source = MAIN.read_text(encoding="utf-8")

    assert source.count("tools/export_white_tiger_checkpoint_strands.py") == 1
    assert source.count("--max-strands 100000") == 1
    assert source.count("--max-strands 0") == 1
    assert source.count("--samples 32") == 2
    assert source.count("--root-domain render") == 2
    assert source.count("--child-count 1") == 2
    assert source.count("--seed 29") == 3
    assert source.count("--uniform-color 0.82 0.80 0.72") == 2
    assert "tools/export_white_tiger_checkpoint_gaussians_ply.py" in source
    assert "--max-gaussians 0" in source
    assert "--sh-degree 3" in source
    assert "r075_003000_render_child1_100k_samples32.npz" in source
    assert "r075_003000_render_child1_all_samples32.npz" in source
    assert "r075_003000_full_3dgs.ply" in source


def test_r075_asset_export_validates_finite_data_and_exact_full_counts() -> None:
    source = MAIN.read_text(encoding="utf-8")

    for fragment in (
        "require_finite(value, str(path))",
        "np.isfinite(value).all()",
        "full_root_count",
        "full_gaussians",
        "gaussian_unique_root_count",
        "exported_gaussians == full_gaussians",
        "ply_vertex_count == full_gaussians",
        "allow_nan=False",
        "checkpoint_hashes.sha256",
        "strand_hashes.sha256",
        "gaussian_ply_hashes.sha256",
        "SHA256SUMS",
        'tee "$LOG_ROOT/export.log"',
    ):
        assert fragment in source

    forbidden = (
        "run_white_tiger",
        "train_white_tiger",
        "resume",
        "scheduler",
        "qsub",
        "qstat",
        "qdel",
        "qalter",
        "s_vmem",
    )
    lowered = source.lower()
    for term in forbidden:
        assert term not in lowered, term


def test_r075_asset_qsub_wrapper_maps_one_physical_device_to_one_local_ordinal() -> None:
    source = QSUB.read_text(encoding="utf-8")

    for fragment in (
        ': "${JOB_ID:?R075 asset qsub wrapper requires JOB_ID}"',
        ': "${PROJECT_ROOT:?R075 asset qsub wrapper requires PROJECT_ROOT}"',
        'job_detail="$(qstat -j "$JOB_ID")"',
        "granted_devices",
        "grep -oE '/dev/nvidia[0-9]+'",
        '[[ "${#granted_devices[@]}" -ne 1 ]]',
        'physical_granted_device="${granted_devices[0]}"',
        "unset CUDA_VISIBLE_DEVICES",
        "nvidia-smi --query-gpu=index --format=csv,noheader,nounits",
        '[[ "${#visible_devices[@]}" -ne 1 ]]',
        'export CUDA_VISIBLE_DEVICES="${visible_devices[0]}"',
        "PHYSICAL_GRANTED_DEVICE=$physical_granted_device",
        "LOCAL_VISIBLE_DEVICE=$local_visible_device",
        'nvidia-smi -i "$CUDA_VISIBLE_DEVICES"',
        'exec bash "$PROJECT_ROOT/scripts/server/export_panda_r075_assets.sh"',
    ):
        assert fragment in source

    forbidden = (
        "CUDA_VISIBLE_DEVICES=0",
        "CUDA_VISIBLE_DEVICES=1",
        "qdel",
        "qalter",
        "kill",
        "release",
        "s_vmem",
    )
    for fragment in forbidden:
        assert fragment not in source
