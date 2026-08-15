from __future__ import annotations

import argparse
import json
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
    parser.add_argument(
        "--highlight-backward-strands",
        action="store_true",
        help="Highlight strands containing a segment that points against the strand chord.",
    )
    parser.add_argument(
        "--highlight-mask-key",
        default="",
        help="Boolean NPZ array selecting strands to highlight.",
    )
    parser.add_argument("--highlight-color", nargs=3, type=float, default=[1.0, 0.0, 0.08])
    parser.add_argument("--highlight-width-scale", type=float, default=4.0)
    parser.add_argument(
        "--secondary-highlight-mask-key",
        default="",
        help="Optional second boolean NPZ strand mask for pair-level diagnostics.",
    )
    parser.add_argument("--secondary-highlight-color", nargs=3, type=float, default=[0.02, 0.45, 1.0])
    parser.add_argument("--secondary-highlight-width-scale", type=float, default=4.0)
    parser.add_argument(
        "--contact-points-key",
        default="",
        help="Optional NPZ [N,3] array of exact contact points to mark.",
    )
    parser.add_argument("--contact-point-color", nargs=3, type=float, default=[1.0, 0.72, 0.02])
    parser.add_argument(
        "--contact-point-radius",
        type=float,
        default=0.0,
        help="World-space marker radius; 0 derives a readable radius from the asset extent.",
    )
    parser.add_argument("--mesh-color", nargs=3, type=float, default=[0.52, 0.55, 0.60])
    parser.add_argument("--background-color", nargs=3, type=float, default=[0.18, 0.18, 0.18])
    parser.add_argument("--key-light-energy", type=float, default=900.0)
    parser.add_argument("--fill-light-energy", type=float, default=260.0)
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
        ids = np.arange(strands.shape[0], dtype=np.int64)
        return strands, widths, colors, ids
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(strands.shape[0], size=max_strands, replace=False))
    return strands[ids], widths[ids], colors[ids], ids


def map_coordinates(strands: np.ndarray, coord_system: str) -> np.ndarray:
    if coord_system == "identity":
        return strands
    if coord_system == "tiger_y_up":
        # Project coordinates use Y as the vertical axis and Z as the tiger's
        # body length. Blender is Z-up, so remap to length/depth/up.
        return strands[..., [2, 0, 1]]
    raise ValueError(f"Unsupported coord system: {coord_system}")


def backward_strand_mask(strands: np.ndarray) -> np.ndarray:
    segments = np.diff(strands, axis=1)
    chord = strands[:, -1] - strands[:, 0]
    chord_unit = chord / np.maximum(
        np.linalg.norm(chord, axis=-1, keepdims=True), 1.0e-12
    )
    projection = np.einsum("nsd,nd->ns", segments, chord_unit)
    return np.any(projection < -1.0e-10, axis=1)


def add_strand_curve_objects(
    *,
    strands: np.ndarray,
    widths: np.ndarray,
    material,
    base_width: float,
    width_scale: float,
    radius_multiplier: float,
    chunk_size: int,
    name_prefix: str,
) -> int:
    import bpy

    object_count = 0
    for chunk_start in range(0, int(strands.shape[0]), chunk_size):
        chunk_end = min(chunk_start + chunk_size, int(strands.shape[0]))
        curve = bpy.data.curves.new(
            f"{name_prefix}_curves_{chunk_start:07d}", "CURVE"
        )
        curve.dimensions = "3D"
        curve.resolution_u = 2
        curve.bevel_resolution = 2
        curve.bevel_depth = base_width
        curve.use_path = False
        obj = bpy.data.objects.new(f"{name_prefix}_{chunk_start:07d}", curve)
        obj.data.materials.append(material)
        bpy.context.collection.objects.link(obj)

        for strand, width in zip(
            strands[chunk_start:chunk_end], widths[chunk_start:chunk_end]
        ):
            spl = curve.splines.new("POLY")
            spl.points.add(strand.shape[0] - 1)
            radius = np.clip(
                width.reshape(-1)
                * width_scale
                * radius_multiplier
                / base_width,
                0.05,
                3.0 * radius_multiplier,
            )
            for point, co, rad in zip(spl.points, strand, radius):
                point.co = (
                    float(co[0]),
                    float(co[1]),
                    float(co[2]),
                    1.0,
                )
                point.radius = float(rad)
        object_count += 1
    return object_count


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


def add_contact_point_object(points: np.ndarray, radius: float, material) -> object | None:
    if points.size == 0:
        return None
    import bmesh
    import bpy
    from mathutils import Matrix

    mesh = bpy.data.meshes.new("crossing_contact_markers")
    bm = bmesh.new()
    try:
        for point in points:
            bmesh.ops.create_icosphere(
                bm,
                subdivisions=2,
                radius=float(radius),
                matrix=Matrix.Translation(tuple(float(value) for value in point)),
            )
        bm.to_mesh(mesh)
    finally:
        bm.free()
    obj = bpy.data.objects.new("crossing_contact_markers", mesh)
    obj.data.materials.append(material)
    bpy.context.collection.objects.link(obj)
    return obj


def main() -> None:
    args = parse_args()
    data = np.load(args.input, allow_pickle=True)
    strands = np.asarray(data["strands"], dtype=np.float32)
    widths = np.asarray(data["widths"], dtype=np.float32)
    colors = np.asarray(data["colors"], dtype=np.float32)
    source_strand_count = int(strands.shape[0])
    strands, widths, colors, selected_ids = sample_strands(
        strands,
        widths,
        colors,
        int(args.max_strands),
        int(args.seed),
    )
    if strands.ndim != 3 or strands.shape[-1] != 3:
        raise RuntimeError(f"strands must be [N,S,3], got {strands.shape}")
    strands = map_coordinates(strands, args.coord_system)
    if args.highlight_mask_key and args.highlight_backward_strands:
        raise RuntimeError(
            "choose either --highlight-mask-key or --highlight-backward-strands"
        )
    if args.highlight_mask_key:
        if args.highlight_mask_key not in data.files:
            raise RuntimeError(
                f"NPZ has no highlight mask array: {args.highlight_mask_key}"
            )
        source_mask = np.asarray(data[args.highlight_mask_key]).reshape(-1)
        if source_mask.shape[0] != source_strand_count:
            raise RuntimeError(
                "highlight mask length does not match the source strand count"
            )
        highlight_mask = source_mask[selected_ids].astype(bool, copy=False)
    elif args.highlight_backward_strands:
        highlight_mask = backward_strand_mask(strands)
    else:
        highlight_mask = np.zeros(strands.shape[0], dtype=bool)
    if args.secondary_highlight_mask_key:
        if args.secondary_highlight_mask_key not in data.files:
            raise RuntimeError(
                f"NPZ has no secondary highlight mask array: {args.secondary_highlight_mask_key}"
            )
        source_secondary_mask = np.asarray(
            data[args.secondary_highlight_mask_key]
        ).reshape(-1)
        if source_secondary_mask.shape[0] != source_strand_count:
            raise RuntimeError(
                "secondary highlight mask length does not match the source strand count"
            )
        secondary_highlight_mask = source_secondary_mask[selected_ids].astype(
            bool, copy=False
        )
    else:
        secondary_highlight_mask = np.zeros(strands.shape[0], dtype=bool)
    highlight_overlap_count = int(
        np.logical_and(highlight_mask, secondary_highlight_mask).sum()
    )
    secondary_highlight_mask = np.logical_and(
        secondary_highlight_mask, ~highlight_mask
    )
    if args.contact_points_key:
        if args.max_strands > 0 and source_strand_count > int(args.max_strands):
            raise RuntimeError(
                "contact-point rendering requires the complete source strand set"
            )
        if args.contact_points_key not in data.files:
            raise RuntimeError(
                f"NPZ has no contact point array: {args.contact_points_key}"
            )
        contact_points = np.asarray(
            data[args.contact_points_key], dtype=np.float32
        ).reshape(-1, 3)
        contact_points = map_coordinates(contact_points, args.coord_system)
    else:
        contact_points = np.empty((0, 3), dtype=np.float32)

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

    highlight_mat = bpy.data.materials.new("highlighted_fur_material")
    highlight_mat.use_nodes = True
    highlight_nodes = highlight_mat.node_tree.nodes
    highlight_bsdf = highlight_nodes.get("Principled BSDF")
    if highlight_bsdf is not None:
        highlight_color = tuple(float(v) for v in args.highlight_color) + (1.0,)
        highlight_bsdf.inputs["Base Color"].default_value = highlight_color
        highlight_bsdf.inputs["Roughness"].default_value = 0.48

    secondary_highlight_mat = bpy.data.materials.new(
        "secondary_highlighted_fur_material"
    )
    secondary_highlight_mat.use_nodes = True
    secondary_highlight_nodes = secondary_highlight_mat.node_tree.nodes
    secondary_highlight_bsdf = secondary_highlight_nodes.get("Principled BSDF")
    if secondary_highlight_bsdf is not None:
        secondary_highlight_color = tuple(
            float(v) for v in args.secondary_highlight_color
        ) + (1.0,)
        secondary_highlight_bsdf.inputs["Base Color"].default_value = (
            secondary_highlight_color
        )
        secondary_highlight_bsdf.inputs["Roughness"].default_value = 0.48

    contact_mat = bpy.data.materials.new("crossing_contact_material")
    contact_mat.use_nodes = True
    contact_nodes = contact_mat.node_tree.nodes
    contact_bsdf = contact_nodes.get("Principled BSDF")
    if contact_bsdf is not None:
        contact_color = tuple(float(v) for v in args.contact_point_color) + (1.0,)
        contact_bsdf.inputs["Base Color"].default_value = contact_color
        contact_bsdf.inputs["Roughness"].default_value = 0.32
        if "Emission Color" in contact_bsdf.inputs:
            contact_bsdf.inputs["Emission Color"].default_value = contact_color
            contact_bsdf.inputs["Emission Strength"].default_value = 0.35

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
    normal_mask = ~(highlight_mask | secondary_highlight_mask)
    curve_object_count = add_strand_curve_objects(
        strands=strands[normal_mask],
        widths=widths[normal_mask],
        material=mat,
        base_width=base_width,
        width_scale=float(args.width_scale),
        radius_multiplier=1.0,
        chunk_size=chunk_size,
        name_prefix="white_tiger_pure_fur",
    )
    if np.any(highlight_mask):
        curve_object_count += add_strand_curve_objects(
            strands=strands[highlight_mask],
            widths=widths[highlight_mask],
            material=highlight_mat,
            base_width=base_width,
            width_scale=float(args.width_scale),
            radius_multiplier=float(args.highlight_width_scale),
            chunk_size=chunk_size,
            name_prefix="highlighted_backward_fur",
        )
    if np.any(secondary_highlight_mask):
        curve_object_count += add_strand_curve_objects(
            strands=strands[secondary_highlight_mask],
            widths=widths[secondary_highlight_mask],
            material=secondary_highlight_mat,
            base_width=base_width,
            width_scale=float(args.width_scale),
            radius_multiplier=float(args.secondary_highlight_width_scale),
            chunk_size=chunk_size,
            name_prefix="secondary_highlighted_fur",
        )

    bbox_min = strands.reshape(-1, 3).min(axis=0)
    bbox_max = strands.reshape(-1, 3).max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    target = center + np.asarray(args.target_offset, dtype=np.float32)
    extent = bbox_max - bbox_min
    max_extent = float(np.max(extent))
    contact_point_radius = (
        float(args.contact_point_radius)
        if args.contact_point_radius > 0.0
        else max(max_extent * 0.0016, base_width * 4.0)
    )
    add_contact_point_object(contact_points, contact_point_radius, contact_mat)
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
    light_data.energy = float(args.key_light_energy)
    light_data.size = max_extent * 1.6

    fill_data = bpy.data.lights.new("soft_fill", "AREA")
    fill = bpy.data.objects.new("soft_fill", fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = (float(center[0] + 0.7 * distance), float(center[1] + 0.4 * distance), float(center[2] + 0.35 * distance))
    fill_data.energy = float(args.fill_light_energy)
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
        "highlight_backward_strands": bool(args.highlight_backward_strands),
        "highlighted_strand_count": int(highlight_mask.sum()),
        "highlight_width_scale": float(args.highlight_width_scale),
        "secondary_highlighted_strand_count": int(
            secondary_highlight_mask.sum()
        ),
        "secondary_highlight_width_scale": float(
            args.secondary_highlight_width_scale
        ),
        "highlight_mask_overlap_count": highlight_overlap_count,
        "contact_point_count": int(contact_points.shape[0]),
        "contact_point_radius": float(contact_point_radius),
        "key_light_energy": float(args.key_light_energy),
        "fill_light_energy": float(args.fill_light_energy),
        "base_width": base_width,
        "curve_chunk_size": chunk_size,
        "curve_object_count": int(curve_object_count),
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
