"""Strict metadata-only migration from the R074 schema12 checkpoint format."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import torch


CURRENT_CHECKPOINT_VERSION = 14
SOURCE_CHECKPOINT_VERSION = 12
REQUIRED_CHECKPOINT_KIND = "stage1_full"
REQUIRED_ITERATION = 3000

MIGRATED_CONFIG_DEFAULTS: dict[str, object] = {
    "guide_length_freeze_until": -1,
    "clean_flow_guide_length_anchor_weight": 0.0,
    "clean_flow_guide_length_anchor_reduction": "mean_l1",
    "view_gate_geometry_support": False,
    "view_gate_length_confidence_support": False,
}


def _qualified_type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, tuple):
        return {"tuple": [_jsonable(item) for item in value]}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return repr(value)


def _mapping_sort_key(key: object) -> str:
    payload = {"type": _qualified_type_name(key), "value": _jsonable(key)}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _mapping_items(value: Mapping[object, object]) -> list[tuple[object, object]]:
    return sorted(value.items(), key=lambda item: _mapping_sort_key(item[0]))


def _path_key(key: object) -> str:
    return f"[{json.dumps(_jsonable(key), sort_keys=True, ensure_ascii=False)}]"


def _iter_tensors(value: object, path: str) -> Iterator[tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        yield path, value
        return
    if isinstance(value, Mapping):
        for key, child in _mapping_items(value):
            yield from _iter_tensors(child, f"{path}{_path_key(key)}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _iter_tensors(child, f"{path}[{index}]")
        return
    if isinstance(value, (set, frozenset)):
        for index, child in enumerate(sorted(value, key=repr)):
            yield from _iter_tensors(child, f"{path}[{index}]")


def _manifest_key(key: object) -> dict[str, object]:
    return {
        "type": _qualified_type_name(key),
        "value": _jsonable(key),
    }


def _object_key_shape_manifest(value: object, path: str) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []

    def visit(child: object, child_path: str) -> None:
        if torch.is_tensor(child):
            manifest.append(
                {
                    "path": child_path,
                    "kind": "tensor",
                    "dtype": str(child.dtype),
                    "shape": [int(size) for size in child.shape],
                }
            )
            return
        if isinstance(child, Mapping):
            items = _mapping_items(child)
            manifest.append(
                {
                    "path": child_path,
                    "kind": "mapping",
                    "keys": [_manifest_key(key) for key, _ in items],
                }
            )
            for key, nested in items:
                visit(nested, f"{child_path}{_path_key(key)}")
            return
        if isinstance(child, (list, tuple)):
            manifest.append(
                {
                    "path": child_path,
                    "kind": "tuple" if isinstance(child, tuple) else "list",
                    "length": len(child),
                }
            )
            for index, nested in enumerate(child):
                visit(nested, f"{child_path}[{index}]")
            return
        if isinstance(child, (set, frozenset)):
            ordered = sorted(child, key=repr)
            manifest.append(
                {
                    "path": child_path,
                    "kind": "set" if isinstance(child, set) else "frozenset",
                    "length": len(ordered),
                }
            )
            for index, nested in enumerate(ordered):
                visit(nested, f"{child_path}[{index}]")
            return
        if isinstance(child, np.ndarray):
            manifest.append(
                {
                    "path": child_path,
                    "kind": "ndarray",
                    "dtype": str(child.dtype),
                    "shape": [int(size) for size in child.shape],
                }
            )
            return
        manifest.append(
            {
                "path": child_path,
                "kind": "value",
                "type": _qualified_type_name(child),
            }
        )

    visit(value, path)
    return manifest


def _tensor_bytes(value: torch.Tensor) -> bytes:
    if value.layout != torch.strided:
        raise RuntimeError(
            "unsupported non-strided checkpoint tensor layout at migration time: "
            f"{value.layout}"
        )
    if value.device.type == "meta":
        raise RuntimeError("meta tensors are not valid checkpoint payloads")
    contiguous = value.detach().cpu().contiguous()
    try:
        return contiguous.numpy().tobytes(order="C")
    except (TypeError, RuntimeError):
        if contiguous.numel() == 0:
            return b""
        try:
            return contiguous.view(torch.uint8).numpy().tobytes(order="C")
        except (TypeError, RuntimeError) as error:
            raise RuntimeError(
                "checkpoint tensor dtype cannot be represented as raw bytes: "
                f"{contiguous.dtype}"
            ) from error


def _tensor_records(
    value: object,
    path: str,
) -> tuple[list[dict[str, object]], str]:
    records: list[dict[str, object]] = []
    aggregate = hashlib.sha256()
    for tensor_path, tensor in _iter_tensors(value, path):
        raw = _tensor_bytes(tensor)
        descriptor = {
            "path": tensor_path,
            "dtype": str(tensor.dtype),
            "shape": [int(size) for size in tensor.shape],
        }
        digest = hashlib.sha256(raw).hexdigest()
        records.append(
            {
                **descriptor,
                "byte_count": len(raw),
                "sha256": digest,
            }
        )
        header = json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        aggregate.update(len(header).to_bytes(8, "big"))
        aggregate.update(header)
        aggregate.update(len(raw).to_bytes(8, "big"))
        aggregate.update(raw)
    return records, aggregate.hexdigest()


def checkpoint_tensor_integrity(
    checkpoint: Mapping[str, object],
) -> dict[str, object]:
    """Return recursive model/optimizer tensor digests and object manifests."""

    section_reports: dict[str, object] = {}
    all_aggregate = hashlib.sha256()
    object_manifest: dict[str, list[dict[str, object]]] = {}
    for section in ("model", "optimizer"):
        records, aggregate = _tensor_records(checkpoint[section], section)
        section_manifest = _object_key_shape_manifest(checkpoint[section], section)
        section_report = {
            "tensor_count": len(records),
            "tensor_manifest": records,
            "aggregate_sha256": aggregate,
            "object_key_shape_manifest": section_manifest,
            "object_key_shape_manifest_sha256": _json_sha256(section_manifest),
        }
        section_reports[section] = section_report
        object_manifest[section] = section_manifest
        all_aggregate.update(section.encode("utf-8"))
        all_aggregate.update(bytes.fromhex(aggregate))
    return {
        **section_reports,
        "aggregate_sha256": all_aggregate.hexdigest(),
        "object_key_shape_manifest": object_manifest,
        "object_key_shape_manifest_sha256": _json_sha256(object_manifest),
    }


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_integrity_equal(
    expected: Mapping[str, object],
    observed: Mapping[str, object],
    label: str,
) -> None:
    if expected.get("aggregate_sha256") != observed.get("aggregate_sha256"):
        raise RuntimeError(f"{label}: aggregate tensor hash mismatch")
    if expected.get("object_key_shape_manifest") != observed.get(
        "object_key_shape_manifest"
    ):
        raise RuntimeError(f"{label}: object key/shape manifest mismatch")
    for section in ("model", "optimizer"):
        expected_section = expected[section]
        observed_section = observed[section]
        if not isinstance(expected_section, Mapping) or not isinstance(
            observed_section, Mapping
        ):
            raise RuntimeError(f"{label}: malformed {section} tensor report")
        if expected_section.get("tensor_manifest") != observed_section.get(
            "tensor_manifest"
        ):
            raise RuntimeError(f"{label}: {section} individual tensor hash mismatch")
        if expected_section.get("aggregate_sha256") != observed_section.get(
            "aggregate_sha256"
        ):
            raise RuntimeError(f"{label}: {section} aggregate tensor hash mismatch")
        if expected_section.get("object_key_shape_manifest") != observed_section.get(
            "object_key_shape_manifest"
        ):
            raise RuntimeError(f"{label}: {section} object key/shape manifest mismatch")


def _scalar_value(value: torch.Tensor) -> object:
    if value.numel() != 1:
        return None
    scalar = value.detach().cpu().reshape(()).item()
    return _jsonable(scalar)


def _tensor_summary(value: object, name: str) -> dict[str, object]:
    if not torch.is_tensor(value):
        raise RuntimeError(f"guide_length_raw optimizer state is missing tensor {name}")
    raw = _tensor_bytes(value)
    cpu = value.detach().cpu()
    return {
        "dtype": str(value.dtype),
        "shape": [int(size) for size in value.shape],
        "byte_count": len(raw),
        "nonzero_count": int(torch.count_nonzero(cpu).item()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        **({"value": _scalar_value(value)} if name == "step" else {}),
    }


def _guide_length_optimizer_snapshot(
    checkpoint: Mapping[str, object],
) -> dict[str, object]:
    optimizer = checkpoint["optimizer"]
    names = checkpoint["optimizer_param_names"]
    if not isinstance(optimizer, Mapping) or not isinstance(names, list):
        raise RuntimeError("checkpoint optimizer metadata is incomplete")
    groups = optimizer.get("param_groups")
    state = optimizer.get("state")
    if not isinstance(groups, list) or not isinstance(state, Mapping):
        raise RuntimeError("checkpoint optimizer metadata is incomplete")

    matches: list[tuple[int, int]] = []
    for group_index, group_names in enumerate(names):
        if not isinstance(group_names, list):
            raise RuntimeError("checkpoint optimizer_param_names has a non-list group")
        for parameter_index, name in enumerate(group_names):
            if name == "guide_length_raw":
                matches.append((group_index, parameter_index))
    if len(matches) != 1:
        raise RuntimeError(
            "checkpoint optimizer_param_names must contain exactly one guide_length_raw"
        )
    group_index, parameter_index = matches[0]
    group = groups[group_index]
    if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
        raise RuntimeError("checkpoint optimizer parameter group is incomplete")
    parameter_ids = group["params"]
    if parameter_index >= len(parameter_ids):
        raise RuntimeError("guide_length_raw optimizer parameter index is out of range")
    state_id = parameter_ids[parameter_index]
    if state_id not in state:
        raise RuntimeError(f"guide_length_raw optimizer state id is missing: {state_id!r}")
    state_entry = state[state_id]
    if not isinstance(state_entry, Mapping):
        raise RuntimeError("guide_length_raw optimizer state is not a mapping")
    step = _tensor_summary(state_entry.get("step"), "step")
    exp_avg = _tensor_summary(state_entry.get("exp_avg"), "exp_avg")
    exp_avg_sq = _tensor_summary(state_entry.get("exp_avg_sq"), "exp_avg_sq")
    moments = {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}
    return {
        "group_index": group_index,
        "parameter_index": parameter_index,
        "state_id": state_id,
        "step": step,
        "moments": moments,
        "step_nonzero_count": step["nonzero_count"],
        "step_sha256": step["sha256"],
        "moment_nonzero_counts": {
            name: summary["nonzero_count"] for name, summary in moments.items()
        },
        "moment_sha256": {
            name: summary["sha256"] for name, summary in moments.items()
        },
    }


def current_stage1_config_fields() -> frozenset[str]:
    """Read current Stage1Config declarations without importing the trainer."""

    source_path = Path(__file__).with_name("train_white_tiger_stage1.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "Stage1Config":
            continue
        names: list[str] = []
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target, ast.Name
            ):
                names.append(statement.target.id)
            elif isinstance(statement, ast.Assign):
                names.extend(
                    target.id
                    for target in statement.targets
                    if isinstance(target, ast.Name)
                )
        if not names:
            raise RuntimeError("current Stage1Config declares no fields")
        if len(names) != len(set(names)):
            raise RuntimeError("current Stage1Config declares duplicate fields")
        return frozenset(names)
    raise RuntimeError(f"could not find Stage1Config in {source_path}")


def _require_config_key_set(
    config: object,
    expected_fields: frozenset[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(config, dict):
        raise RuntimeError(f"{label} checkpoint config is not a dictionary")
    if not all(isinstance(key, str) for key in config):
        raise RuntimeError(f"{label} checkpoint config keys must be strings")
    observed_fields = set(config)
    if observed_fields != expected_fields:
        missing = sorted(expected_fields - observed_fields)
        extra = sorted(observed_fields - expected_fields)
        raise RuntimeError(
            f"{label} checkpoint config key set mismatch: "
            f"missing={missing}, extra={extra}"
        )
    return config


def _require_added_config_values(config: Mapping[str, object]) -> None:
    for key, expected in MIGRATED_CONFIG_DEFAULTS.items():
        if key not in config:
            raise RuntimeError(f"migrated checkpoint config is missing {key}")
        observed = config[key]
        if type(observed) is not type(expected) or observed != expected:
            raise RuntimeError(
                f"migrated checkpoint config {key} changed: "
                f"expected={expected!r}, got={observed!r}"
            )


def _require_optimizer_structure(
    optimizer: object,
    optimizer_param_names: object,
) -> None:
    if not isinstance(optimizer, dict):
        raise RuntimeError("checkpoint optimizer state is not a dictionary")
    if not isinstance(optimizer_param_names, list):
        raise RuntimeError("checkpoint optimizer_param_names is not a list")
    groups = optimizer.get("param_groups")
    state = optimizer.get("state")
    if not isinstance(groups, list) or not groups:
        raise RuntimeError("checkpoint optimizer has no complete param_groups")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint optimizer has no complete state mapping")
    if len(groups) != len(optimizer_param_names):
        raise RuntimeError(
            "checkpoint optimizer_param_names group count does not match optimizer"
        )

    parameter_ids: list[int] = []
    all_names: list[str] = []
    for group_index, (group, names) in enumerate(zip(groups, optimizer_param_names)):
        if not isinstance(group, dict):
            raise RuntimeError(f"checkpoint optimizer group {group_index} is not a mapping")
        parameters = group.get("params")
        if not isinstance(parameters, list):
            raise RuntimeError(f"checkpoint optimizer group {group_index} has no params")
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise RuntimeError(
                f"checkpoint optimizer_param_names group {group_index} is incomplete"
            )
        if len(parameters) != len(names):
            raise RuntimeError(
                f"checkpoint optimizer group {group_index} parameter/name count mismatch"
            )
        for parameter_id in parameters:
            if type(parameter_id) is not int:
                raise RuntimeError(
                    "checkpoint optimizer parameter IDs must be integers: "
                    f"{parameter_id!r}"
                )
            parameter_ids.append(parameter_id)
        all_names.extend(names)

    if len(parameter_ids) != len(set(parameter_ids)):
        raise RuntimeError("checkpoint optimizer contains duplicate parameter IDs")
    if len(all_names) != len(set(all_names)):
        raise RuntimeError("checkpoint optimizer_param_names contains duplicate names")
    state_ids = list(state)
    if any(type(state_id) is not int for state_id in state_ids):
        raise RuntimeError("checkpoint optimizer state IDs must be integers")
    if set(state_ids) != set(parameter_ids):
        missing = sorted(set(parameter_ids) - set(state_ids))
        extra = sorted(set(state_ids) - set(parameter_ids))
        raise RuntimeError(
            "checkpoint optimizer state is incomplete: "
            f"missing={missing}, extra={extra}"
        )
    if not all(isinstance(entry, dict) for entry in state.values()):
        raise RuntimeError("checkpoint optimizer state entries must be mappings")
    if all_names.count("guide_length_raw") != 1:
        raise RuntimeError(
            "checkpoint optimizer_param_names must contain exactly one guide_length_raw"
        )


def _require_checkpoint_structure(
    checkpoint: object,
    *,
    expected_version: int,
    expected_config_fields: frozenset[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"{label} checkpoint is not a dictionary")
    required = {
        "checkpoint_version",
        "checkpoint_kind",
        "iteration",
        "config",
        "model",
        "optimizer",
        "optimizer_param_names",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(f"{label} checkpoint is incomplete: missing={missing}")
    if type(checkpoint["checkpoint_version"]) is not int or checkpoint[
        "checkpoint_version"
    ] != expected_version:
        raise RuntimeError(
            f"{label} checkpoint_version mismatch: "
            f"expected {expected_version}, got {checkpoint.get('checkpoint_version')!r}"
        )
    if checkpoint["checkpoint_kind"] != REQUIRED_CHECKPOINT_KIND:
        raise RuntimeError(
            f"{label} checkpoint_kind mismatch: "
            f"expected {REQUIRED_CHECKPOINT_KIND!r}, got {checkpoint['checkpoint_kind']!r}"
        )
    if type(checkpoint["iteration"]) is not int or checkpoint["iteration"] != REQUIRED_ITERATION:
        raise RuntimeError(
            f"{label} checkpoint iteration mismatch: "
            f"expected {REQUIRED_ITERATION}, got {checkpoint['iteration']!r}"
        )
    config = _require_config_key_set(
        checkpoint["config"], expected_config_fields, label
    )
    model = checkpoint["model"]
    if not isinstance(model, dict) or not model:
        raise RuntimeError(f"{label} checkpoint model state is incomplete")
    if next(_iter_tensors(model, "model"), None) is None:
        raise RuntimeError(f"{label} checkpoint model contains no tensors")
    _require_optimizer_structure(
        checkpoint["optimizer"], checkpoint["optimizer_param_names"]
    )
    optimizer_tensors = next(_iter_tensors(checkpoint["optimizer"], "optimizer"), None)
    if optimizer_tensors is None:
        raise RuntimeError(f"{label} checkpoint optimizer contains no tensors")
    _guide_length_optimizer_snapshot(checkpoint)
    return checkpoint


def _values_exact_equal(left: object, right: object) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            return False
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            return False
        return _tensor_bytes(left) == _tensor_bytes(right)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return False
        if left.dtype != right.dtype or left.shape != right.shape:
            return False
        return np.ascontiguousarray(left).tobytes() == np.ascontiguousarray(right).tobytes()
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(_values_exact_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(_values_exact_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (set, frozenset)) or isinstance(right, (set, frozenset)):
        return type(left) is type(right) and left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, float) and math.isnan(left):
        return math.isnan(right)
    try:
        result = left == right
    except Exception:
        return repr(left) == repr(right)
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _require_checkpoint_values_unchanged(
    expected: Mapping[str, object],
    observed: Mapping[str, object],
    *,
    allowed: frozenset[str],
    label: str,
) -> None:
    if set(expected) != set(observed):
        raise RuntimeError(f"{label}: checkpoint object keys changed")
    for key in expected:
        if key in allowed:
            continue
        if not _values_exact_equal(expected[key], observed[key]):
            raise RuntimeError(f"{label}: checkpoint value changed at {key!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _numpy_pickle_compat_aliases() -> Iterator[None]:
    aliases = {
        "numpy._core": "numpy.core",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
    }
    installed: dict[str, object] = {}
    for alias, source in aliases.items():
        if alias not in sys.modules:
            module = importlib.import_module(source)
            sys.modules[alias] = module
            installed[alias] = module
    try:
        yield
    finally:
        for alias, module in installed.items():
            if sys.modules.get(alias) is module:
                del sys.modules[alias]


def _load_checkpoint_cpu(path: Path) -> dict[str, object]:
    def load() -> object:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    try:
        checkpoint = load()
    except ModuleNotFoundError as error:
        if error.name is None or not error.name.startswith("numpy._core"):
            raise
        with _numpy_pickle_compat_aliases():
            checkpoint = load()
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"checkpoint is not a dictionary: {path}")
    return checkpoint


def _normalise_expected_sha256(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("expected input SHA256 must be exactly 64 hexadecimal characters")
    return value.lower()


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _resolved_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _make_temporary_sibling(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(descriptor)
    return Path(name)


def _write_json_atomically(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _make_temporary_sibling(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        if _path_exists(path):
            raise FileExistsError(f"report output already exists: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def migrate_checkpoint(
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    expected_input_sha256: str,
) -> dict[str, object]:
    """Migrate one R074 schema12 checkpoint without changing training state."""

    source_path = _resolved_path(input_path)
    destination_path = _resolved_path(output_path)
    metadata_path = _resolved_path(report_path)
    expected_sha256 = _normalise_expected_sha256(expected_input_sha256)

    if not source_path.is_file():
        raise FileNotFoundError(f"input checkpoint does not exist: {source_path}")
    if _path_exists(destination_path):
        raise FileExistsError(f"checkpoint output already exists: {destination_path}")
    if _path_exists(metadata_path):
        raise FileExistsError(f"report output already exists: {metadata_path}")
    if os.path.normcase(str(destination_path)) == os.path.normcase(str(metadata_path)):
        raise ValueError("checkpoint output and report output must be different paths")

    source_sha256 = _sha256_file(source_path)
    if source_sha256 != expected_sha256:
        raise RuntimeError(
            "input SHA256 mismatch: "
            f"expected {expected_sha256}, got {source_sha256}"
        )

    current_fields = current_stage1_config_fields()
    missing_migration_fields = set(MIGRATED_CONFIG_DEFAULTS) - current_fields
    if missing_migration_fields:
        raise RuntimeError(
            "current Stage1Config is missing migration fields: "
            f"{sorted(missing_migration_fields)}"
        )
    schema12_fields = current_fields - set(MIGRATED_CONFIG_DEFAULTS)

    source = _load_checkpoint_cpu(source_path)
    _require_checkpoint_structure(
        source,
        expected_version=SOURCE_CHECKPOINT_VERSION,
        expected_config_fields=schema12_fields,
        label="schema12",
    )
    source_config = source["config"]
    if not isinstance(source_config, dict):
        raise RuntimeError("schema12 checkpoint config is not a dictionary")
    source_integrity = checkpoint_tensor_integrity(source)
    source_guide = _guide_length_optimizer_snapshot(source)

    migrated = source.copy()
    migrated_config = source_config.copy()
    migrated_config.update(MIGRATED_CONFIG_DEFAULTS)
    migrated["config"] = migrated_config
    migrated["checkpoint_version"] = CURRENT_CHECKPOINT_VERSION
    _require_checkpoint_structure(
        migrated,
        expected_version=CURRENT_CHECKPOINT_VERSION,
        expected_config_fields=current_fields,
        label="migrated",
    )
    _require_added_config_values(migrated_config)
    _require_checkpoint_values_unchanged(
        source,
        migrated,
        allowed=frozenset({"checkpoint_version", "config"}),
        label="in-memory migration",
    )
    for key, value in source_config.items():
        if not _values_exact_equal(value, migrated_config[key]):
            raise RuntimeError(f"in-memory migration changed config value at {key!r}")
    migrated_integrity = checkpoint_tensor_integrity(migrated)
    _require_integrity_equal(source_integrity, migrated_integrity, "in-memory migration")
    migrated_guide = _guide_length_optimizer_snapshot(migrated)
    if not _values_exact_equal(source_guide, migrated_guide):
        raise RuntimeError("in-memory migration changed guide_length_raw optimizer metadata")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint: Path | None = _make_temporary_sibling(destination_path)
    try:
        torch.save(migrated, temporary_checkpoint)
        reloaded = _load_checkpoint_cpu(temporary_checkpoint)
        _require_checkpoint_structure(
            reloaded,
            expected_version=CURRENT_CHECKPOINT_VERSION,
            expected_config_fields=current_fields,
            label="reloaded schema14",
        )
        _require_added_config_values(reloaded["config"])
        _require_checkpoint_values_unchanged(
            migrated,
            reloaded,
            allowed=frozenset(),
            label="serialized checkpoint reload",
        )
        reloaded_integrity = checkpoint_tensor_integrity(reloaded)
        _require_integrity_equal(
            migrated_integrity,
            reloaded_integrity,
            "serialized checkpoint reload",
        )
        reloaded_guide = _guide_length_optimizer_snapshot(reloaded)
        if not _values_exact_equal(source_guide, reloaded_guide):
            raise RuntimeError(
                "serialized checkpoint reload changed guide_length_raw optimizer metadata"
            )

        if _path_exists(destination_path):
            raise FileExistsError(f"checkpoint output already exists: {destination_path}")
        os.replace(temporary_checkpoint, destination_path)
        temporary_checkpoint = None
        output_sha256 = _sha256_file(destination_path)

        report: dict[str, object] = {
            "status": "pass",
            "migration": "R074 schema12 -> schema14",
            "source": {
                "path": str(source_path),
                "sha256": source_sha256,
            },
            "output": {
                "path": str(destination_path),
                "sha256": output_sha256,
            },
            "source_sha256": source_sha256,
            "output_sha256": output_sha256,
            "source_checkpoint_sha256": source_sha256,
            "migrated_checkpoint_sha256": output_sha256,
            "expected_input_sha256": expected_sha256,
            "source_checkpoint_version": SOURCE_CHECKPOINT_VERSION,
            "output_checkpoint_version": CURRENT_CHECKPOINT_VERSION,
            "checkpoint_kind": REQUIRED_CHECKPOINT_KIND,
            "iteration": REQUIRED_ITERATION,
            "source_iteration": REQUIRED_ITERATION,
            "migrated_iteration": REQUIRED_ITERATION,
            "config_delta": {
                "removed": [],
                "added": dict(MIGRATED_CONFIG_DEFAULTS),
                "changed": {},
            },
            "tensor_manifests": {
                "source": source_integrity,
                "migrated": migrated_integrity,
                "output": reloaded_integrity,
            },
            "tensor_integrity_checks": {
                "source_to_migrated_identical": True,
                "migrated_to_output_identical": True,
                "individual_and_aggregate_hashes_identical": True,
                "object_key_shape_manifests_identical": True,
            },
            "tensor_identity": {
                "model": True,
                "optimizer": True,
                "individual_hashes": True,
                "aggregate_hashes": True,
                "object_key_shape_manifests": True,
            },
            "schema14_defaults": dict(MIGRATED_CONFIG_DEFAULTS),
            "guide_length_raw_optimizer": {
                "group_index": source_guide["group_index"],
                "parameter_index": source_guide["parameter_index"],
                "state_id": source_guide["state_id"],
                "source": source_guide,
                "output": reloaded_guide,
            },
        }
        if _path_exists(metadata_path):
            raise FileExistsError(f"report output already exists: {metadata_path}")
        _write_json_atomically(metadata_path, report)
        return report
    finally:
        if temporary_checkpoint is not None:
            temporary_checkpoint.unlink(missing_ok=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly migrate an R074 schema12 Stage1 checkpoint to schema14."
    )
    parser.add_argument(
        "--input",
        "--input-checkpoint",
        "--checkpoint",
        dest="input_path",
        type=Path,
        required=True,
        help="schema12 checkpoint input path",
    )
    parser.add_argument(
        "--output",
        "--output-checkpoint",
        dest="output_path",
        type=Path,
        required=True,
        help="new schema14 checkpoint output path; must not exist",
    )
    parser.add_argument(
        "--report",
        "--report-path",
        dest="report_path",
        type=Path,
        required=True,
        help="migration report output path; must not exist",
    )
    parser.add_argument(
        "--expected-input-sha256",
        "--expected-sha256",
        dest="expected_input_sha256",
        required=True,
        help="expected SHA256 for the input checkpoint",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = migrate_checkpoint(
            args.input_path,
            args.output_path,
            args.report_path,
            args.expected_input_sha256,
        )
    except Exception as error:
        print(f"migration failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
