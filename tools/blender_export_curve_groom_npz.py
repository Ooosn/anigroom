"""Export selected Blender curve groom strands to AniGroom .npz.

Run inside Blender, for example:

blender scene.blend --background --python tools/blender_export_curve_groom_npz.py -- D:/tmp/groom.npz

This exporter targets Blender curve objects converted/evaluated as groom
strands. It writes arrays compatible with validate_strand_gaussian_correspondence.py:
strands [N, S, 3], widths [N, S, 1], colors [N, S, 3], opacities [N, S, 1].
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _args_after_double_dash() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def _resample_polyline(points: np.ndarray, samples: int) -> np.ndarray:
    if points.shape[0] < 2:
        raise ValueError("a groom strand needs at least two points")
    seg = points[1:] - points[:-1]
    lens = np.linalg.norm(seg, axis=-1)
    cumulative = np.concatenate([[0.0], np.cumsum(lens)])
    total = max(float(cumulative[-1]), 1e-8)
    target = np.linspace(0.0, total, samples)
    out = np.zeros((samples, 3), dtype=np.float32)
    for i, d in enumerate(target):
        upper = int(np.searchsorted(cumulative, d, side="right"))
        upper = min(max(upper, 1), points.shape[0] - 1)
        lower = upper - 1
        denom = max(float(cumulative[upper] - cumulative[lower]), 1e-8)
        w = (d - cumulative[lower]) / denom
        out[i] = points[lower] * (1.0 - w) + points[upper] * w
    return out


def main() -> None:
    args = _args_after_double_dash()
    if not args:
        raise SystemExit("usage: blender --background --python blender_export_curve_groom_npz.py -- output.npz [samples]")
    output = Path(args[0])
    samples = int(args[1]) if len(args) > 1 else 24

    import bpy

    selected = [obj for obj in bpy.context.selected_objects if obj.type == "CURVE"]
    if not selected:
        raise RuntimeError("select at least one Blender CURVE object before export")

    strands: list[np.ndarray] = []
    for obj in selected:
        matrix = np.array(obj.matrix_world, dtype=np.float32)
        for spline in obj.data.splines:
            pts = []
            if spline.type == "BEZIER":
                for p in spline.bezier_points:
                    co = np.array([p.co.x, p.co.y, p.co.z, 1.0], dtype=np.float32)
                    pts.append((matrix @ co)[:3])
            else:
                for p in spline.points:
                    co = np.array([p.co.x, p.co.y, p.co.z, 1.0], dtype=np.float32)
                    pts.append((matrix @ co)[:3])
            if len(pts) >= 2:
                strands.append(_resample_polyline(np.asarray(pts, dtype=np.float32), samples))

    if not strands:
        raise RuntimeError("selected curve objects did not contain usable strands")

    strands_arr = np.stack(strands, axis=0).astype(np.float32)
    t = np.linspace(0.0, 1.0, samples, dtype=np.float32).reshape(1, samples, 1)
    widths = (0.0035 * (1.0 - 0.75 * t)).repeat(strands_arr.shape[0], axis=0).astype(np.float32)
    colors = np.ones((strands_arr.shape[0], samples, 3), dtype=np.float32)
    opacities = np.full((strands_arr.shape[0], samples, 1), 0.75, dtype=np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, strands=strands_arr, widths=widths, colors=colors, opacities=opacities)
    print(f"exported {strands_arr.shape[0]} strands to {output}")


if __name__ == "__main__":
    main()
