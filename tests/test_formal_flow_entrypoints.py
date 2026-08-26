from __future__ import annotations

import importlib


def test_formal_v4_flow_entrypoints_import_with_training_io() -> None:
    module_names = (
        "tools.build_white_tiger_smal_head_guides",
        "tools.fuse_gpt_flow_multiview",
        "tools.fuse_gpt_flow_shell_multiview",
        "tools.visualize_flow_targets_as_strands",
    )
    modules = [importlib.import_module(name) for name in module_names]
    assert all(callable(module.main) for module in modules)

    trainer = importlib.import_module("tools.train_white_tiger_stage1")
    assert callable(trainer.load_camera_tensors)
    assert callable(trainer.load_mask)
