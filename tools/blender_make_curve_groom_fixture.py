"""Create a small Blender curve-groom fixture, render it, and export strands.

Run with:
D:/Program Files/Blender Foundation/Blender 5.0/blender.exe --background --python tools/blender_make_curve_groom_fixture.py -- OUTPUT_DIR
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def make_groom_fixture(strand_count: int, samples: int, width: float, style: str):
    rows = int(math.sqrt(strand_count))
    cols = int(math.ceil(strand_count / max(rows, 1)))
    roots = []
    for r in range(rows):
        for c in range(cols):
            if len(roots) >= strand_count:
                break
            x = -0.45 + 0.90 * (c / max(cols - 1, 1))
            y = -0.28 + 0.56 * (r / max(rows - 1, 1))
            roots.append((x, y))
    roots = np.asarray(roots, dtype=np.float32)
    t = np.linspace(0.0, 1.0, samples, dtype=np.float32).reshape(1, samples, 1)
    phase = (roots[:, [0]] * 5.0 + roots[:, [1]] * 3.0).reshape(-1, 1, 1)
    root_xyz = np.concatenate([roots[:, [0]], roots[:, [1]], np.full((roots.shape[0], 1), 2.2, dtype=np.float32)], axis=-1).reshape(-1, 1, 3)
    flow = np.asarray([0.20, 0.62, 0.18], dtype=np.float32).reshape(1, 1, 3)
    side = np.asarray([1.0, 0.0, 0.0], dtype=np.float32).reshape(1, 1, 3)
    up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32).reshape(1, 1, 3)
    if style == "straight":
        curl = 0.010 * np.sin(1.0 * math.pi * t + phase)
        droop = -0.025 * (t * t)
        length = 0.36
    elif style == "wavy":
        curl = 0.055 * np.sin(2.5 * math.pi * t + phase)
        droop = -0.060 * (t * t)
        length = 0.42
    elif style == "curly":
        curl = 0.080 * np.sin(5.2 * math.pi * t + phase)
        droop = -0.040 * (t * t)
        length = 0.34
    else:
        raise ValueError(f"unknown style: {style}")
    strands = root_xyz + length * t * flow + curl * side + droop * up
    widths = (float(width) * (1.0 - 0.68 * t)).repeat(strands.shape[0], axis=0).astype(np.float32)
    colors = np.ones((*strands.shape[:2], 3), dtype=np.float32)
    stripe = ((np.arange(strands.shape[0]).reshape(-1, 1, 1) % 5) == 0).astype(np.float32)
    colors = colors * (0.92 - 0.48 * stripe)
    colors[..., 1] *= 0.96
    colors[..., 2] *= 0.86
    opacities = np.ones((*strands.shape[:2], 1), dtype=np.float32) * 0.82
    return strands.astype(np.float32), widths, colors.astype(np.float32), opacities


def main() -> None:
    args = args_after_dash()
    if not args:
        raise SystemExit("usage: blender --background --python blender_make_curve_groom_fixture.py -- OUTPUT_DIR [style]")
    output_dir = Path(args[0])
    style = args[1] if len(args) > 1 else "wavy"
    output_dir.mkdir(parents=True, exist_ok=True)
    width_px, height_px = 1280, 720
    focal_px = 1450.0
    strands, widths, colors, opacities = make_groom_fixture(120, 24, 0.0022, style)

    import bpy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    scene = bpy.context.scene
    scene.render.resolution_x = width_px
    scene.render.resolution_y = height_px
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.world = bpy.data.worlds.new("groom_world") if scene.world is None else scene.world
    scene.world.color = (0.70, 0.72, 0.74)
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items} else "BLENDER_EEVEE"

    cam_data = bpy.data.cameras.new("camera")
    cam = bpy.data.objects.new("camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = (0.0, 0.0, 0.0)
    cam.rotation_euler = (0.0, math.pi, 0.0)
    cam_data.angle = 2.0 * math.atan(width_px / (2.0 * focal_px))
    scene.camera = cam

    mat_cache = {}
    for sid, strand in enumerate(strands):
        color = tuple(float(v) for v in colors[sid, 0])
        key = tuple(round(v, 3) for v in color)
        if key not in mat_cache:
            mat = bpy.data.materials.new(f"mat_{sid}")
            mat.use_nodes = True
            mat.node_tree.nodes.clear()
            out = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
            emission = mat.node_tree.nodes.new("ShaderNodeEmission")
            emission.inputs["Color"].default_value = (*color, 1.0)
            emission.inputs["Strength"].default_value = 1.0
            mat.node_tree.links.new(emission.outputs["Emission"], out.inputs["Surface"])
            mat_cache[key] = mat
        curve = bpy.data.curves.new(f"strand_{sid:04d}", "CURVE")
        curve.dimensions = "3D"
        curve.resolution_u = 2
        curve.bevel_depth = float(widths[sid, 0, 0]) * 0.45
        curve.bevel_resolution = 2
        spl = curve.splines.new("POLY")
        spl.points.add(strand.shape[0] - 1)
        for p, co in zip(spl.points, strand):
            p.co = (float(co[0]), float(co[1]), float(co[2]), 1.0)
        obj = bpy.data.objects.new(curve.name, curve)
        obj.data.materials.append(mat_cache[key])
        bpy.context.collection.objects.link(obj)

    render_path = output_dir / "blender_groom_reference.png"
    scene.render.filepath = str(render_path)
    bpy.ops.render.render(write_still=True)
    npz_path = output_dir / "blender_groom_fixture.npz"
    np.savez_compressed(npz_path, strands=strands, widths=widths, colors=colors, opacities=opacities)
    report = {"style": style, "render": str(render_path), "npz": str(npz_path), "strand_count": int(strands.shape[0]), "samples": int(strands.shape[1])}
    (output_dir / "blender_groom_fixture.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
