from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def args_after_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render exported strand curves as pure fur in Blender.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mesh", default="")
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--mesh-translation", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--max-strands", type=int, default=0)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--resolution", nargs=2, type=int, default=[1920, 1080])
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--width-scale", type=float, default=1.65)
    parser.add_argument(
        "--curve-chunk-size",
        type=int,
        default=100000,
        help="Maximum strands per Blender Curve object. Chunking avoids Blender 2.82 curve array limits.",
    )
    parser.add_argument("--material-color", nargs=3, type=float, default=[0.82, 0.80, 0.72])
    parser.add_argument("--mesh-color", nargs=3, type=float, default=[0.52, 0.55, 0.60])
    parser.add_argument("--background-color", nargs=3, type=float, default=[0.18, 0.18, 0.18])
    parser.add_argument(
        "--camera",
        choices=("side_y", "side_y_pos", "side_x", "side_x_neg", "front_z", "view09_three_quarter"),
        default="side_y",
    )
    parser.add_argument("--camera-offset", nargs=3, type=float, default=None)
    parser.add_argument(
        "--target-offset",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        help="Camera target offset in the selected Blender coordinate system.",
    )
    parser.add_argument(
        "--coord-system",
        choices=("identity", "tiger_y_up"),
        default="tiger_y_up",
        help="Map exported project coordinates into Blender coordinates. tiger_y_up maps (x,y,z)->(z,x,y).",
    )
    parser.add_argument("--ortho-scale", type=float, default=0.0)
    parser.add_argument("--crop-report", action="store_true")
    return parser.parse_args(args_after_dash())


def look_at(obj, target: np.ndarray) -> None:
    import mathutils

    direction = mathutils.Vector((float(target[0] - obj.location.x), float(target[1] - obj.location.y), float(target[2] - obj.location.z)))
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def sample_strands(strands: np.ndarray, widths: np.ndarray, colors: np.ndarray, max_strands: int, seed: int):
    if max_strands <= 0 or strands.shape[0] <= max_strands:
        return strands, widths, colors
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(strands.shape[0], size=max_strands, replace=False))
    return strands[ids], widths[ids], colors[ids]


def map_coordinates(strands: np.ndarray, coord_system: str) -> np.ndarray:
    if coord_system == "identity":
        return strands
    if coord_system == "tiger_y_up":
        # Project coordinates use Y as the vertical axis and Z as the tiger's
        # body length. Blender is Z-up, so remap to length/depth/up.
        return strands[..., [2, 0, 1]]
    raise ValueError(f"Unsupported coord system: {coord_system}")


def read_obj_mesh(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                face = []
                for part in parts:
                    idx = int(part.split("/")[0])
                    face.append(idx - 1 if idx > 0 else len(vertices) + idx)
                if len(face) >= 3:
                    faces.append(face)
    if not vertices or not faces:
        raise RuntimeError(f"OBJ did not contain usable mesh data: {path}")
    return np.asarray(vertices, dtype=np.float32), faces


def add_mesh_object(args: argparse.Namespace, mat) -> object | None:
    if not args.mesh:
        return None
    import bpy

    vertices, faces = read_obj_mesh(Path(args.mesh))
    translation = np.asarray(args.mesh_translation, dtype=np.float32).reshape(1, 3)
    vertices = vertices * float(args.mesh_scale) + translation
    vertices = map_coordinates(vertices, args.coord_system)
    mesh = bpy.data.meshes.new("furless_body_mesh")
    mesh.from_pydata([tuple(v) for v in vertices], [], faces)
    mesh.update()
    obj = bpy.data.objects.new("furless_body_mesh", mesh)
    obj.data.materials.append(mat)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    try:
        bpy.ops.object.shade_smooth()
    finally:
        obj.select_set(False)
    return obj


def main() -> None:
    args = parse_args()
    data = np.load(args.input, allow_pickle=True)
    strands = np.asarray(data["strands"], dtype=np.float32)
    widths = np.asarray(data["widths"], dtype=np.float32)
    colors = np.asarray(data["colors"], dtype=np.float32)
    strands, widths, colors = sample_strands(strands, widths, colors, int(args.max_strands), int(args.seed))
    if strands.ndim != 3 or strands.shape[-1] != 3:
        raise RuntimeError(f"strands must be [N,S,3], got {strands.shape}")
    strands = map_coordinates(strands, args.coord_system)

    import bpy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    scene = bpy.context.scene
    scene.render.resolution_x = int(args.resolution[0])
    scene.render.resolution_y = int(args.resolution[1])
    scene.render.film_transparent = False
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(args.samples)
    scene.cycles.use_denoising = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.world = bpy.data.worlds.new("pure_fur_world") if scene.world is None else scene.world
    scene.world.color = tuple(float(v) for v in args.background_color)
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = tuple(float(v) for v in args.background_color) + (1.0,)
        bg.inputs["Strength"].default_value = 0.85

    mat = bpy.data.materials.new("pure_fur_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is not None:
        color = tuple(float(v) for v in args.material_color) + (1.0,)
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.72
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = 1.0

    mesh_mat = bpy.data.materials.new("furless_body_material")
    mesh_mat.use_nodes = True
    mesh_nodes = mesh_mat.node_tree.nodes
    mesh_bsdf = mesh_nodes.get("Principled BSDF")
    if mesh_bsdf is not None:
        mesh_bsdf.inputs["Base Color"].default_value = tuple(float(v) for v in args.mesh_color) + (1.0,)
        mesh_bsdf.inputs["Roughness"].default_value = 0.86
    add_mesh_object(args, mesh_mat)

    base_width = max(float(np.percentile(widths, 60)) * float(args.width_scale), 1.0e-5)
    chunk_size = max(int(args.curve_chunk_size), 1)
    for chunk_start in range(0, int(strands.shape[0]), chunk_size):
        chunk_end = min(chunk_start + chunk_size, int(strands.shape[0]))
        curve = bpy.data.curves.new(f"white_tiger_pure_fur_curves_{chunk_start:07d}", "CURVE")
        curve.dimensions = "3D"
        curve.resolution_u = 2
        curve.bevel_resolution = 2
        curve.bevel_depth = base_width
        curve.use_path = False
        obj = bpy.data.objects.new(f"white_tiger_pure_fur_{chunk_start:07d}", curve)
        obj.data.materials.append(mat)
        bpy.context.collection.objects.link(obj)

        for strand, width in zip(strands[chunk_start:chunk_end], widths[chunk_start:chunk_end]):
            spl = curve.splines.new("POLY")
            spl.points.add(strand.shape[0] - 1)
            radius = np.clip(width.reshape(-1) * float(args.width_scale) / base_width, 0.05, 3.0)
            for point, co, rad in zip(spl.points, strand, radius):
                point.co = (float(co[0]), float(co[1]), float(co[2]), 1.0)
                point.radius = float(rad)

    bbox_min = strands.reshape(-1, 3).min(axis=0)
    bbox_max = strands.reshape(-1, 3).max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    target = center + np.asarray(args.target_offset, dtype=np.float32)
    extent = bbox_max - bbox_min
    max_extent = float(np.max(extent))
    distance = max_extent * 2.6 + 0.5

    cam_data = bpy.data.cameras.new("camera")
    cam = bpy.data.objects.new("camera", cam_data)
    bpy.context.collection.objects.link(cam)
    if args.camera_offset is not None:
        offset = np.asarray(args.camera_offset, dtype=np.float32)
        offset = offset / max(float(np.linalg.norm(offset)), 1.0e-8)
        loc = target + offset * distance
        cam.location = (float(loc[0]), float(loc[1]), float(loc[2]))
    elif args.camera == "side_y":
        cam.location = (float(target[0]), float(target[1] - distance), float(target[2] + 0.05 * max_extent))
    elif args.camera == "side_y_pos":
        cam.location = (float(target[0]), float(target[1] + distance), float(target[2] + 0.05 * max_extent))
    elif args.camera == "side_x":
        cam.location = (float(target[0] + distance), float(target[1]), float(target[2] + 0.05 * max_extent))
    elif args.camera == "side_x_neg":
        cam.location = (float(target[0] - distance), float(target[1]), float(target[2] + 0.05 * max_extent))
    elif args.camera == "view09_three_quarter":
        loc = target + np.asarray([-0.35, 1.0, 0.16], dtype=np.float32) / np.linalg.norm([-0.35, 1.0, 0.16]) * distance
        cam.location = (float(loc[0]), float(loc[1]), float(loc[2]))
    else:
        cam.location = (float(target[0]), float(target[1]), float(target[2] + distance))
    look_at(cam, target)
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = float(args.ortho_scale) if args.ortho_scale > 0 else max(float(extent[0]), float(extent[2])) * 1.10
    scene.camera = cam

    light_data = bpy.data.lights.new("large_softbox", "AREA")
    light = bpy.data.objects.new("large_softbox", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (float(center[0] - 0.6 * distance), float(center[1] - 0.8 * distance), float(center[2] + 0.9 * distance))
    light_data.energy = 900.0
    light_data.size = max_extent * 1.6

    fill_data = bpy.data.lights.new("soft_fill", "AREA")
    fill = bpy.data.objects.new("soft_fill", fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = (float(center[0] + 0.7 * distance), float(center[1] + 0.4 * distance), float(center[2] + 0.35 * distance))
    fill_data.energy = 260.0
    fill_data.size = max_extent * 2.0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)

    report = {
        "input": str(args.input),
        "output": str(output_path),
        "strand_count": int(strands.shape[0]),
        "samples_per_strand": int(strands.shape[1]),
        "base_width": base_width,
        "curve_chunk_size": chunk_size,
        "curve_object_count": int(math.ceil(int(strands.shape[0]) / chunk_size)),
        "camera": args.camera,
        "target_offset": [float(v) for v in args.target_offset],
        "coord_system": args.coord_system,
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
    }
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
