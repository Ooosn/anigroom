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
    parser.add_argument("--material-roughness", type=float, default=0.72)
    parser.add_argument("--material-specular", type=float, default=0.22)
    parser.add_argument(
        "--use-input-colors",
        action="store_true",
        help="Shade each displayed strand with its NPZ root-to-tip color profile.",
    )
    parser.add_argument(
        "--ground-plane",
        action="store_true",
        help="Add a shallow convex receiver below the strand for soft contact shadows.",
    )
    parser.add_argument("--ground-color", nargs=3, type=float, default=[0.14, 0.15, 0.16])
    parser.add_argument(
        "--ground-relief",
        type=float,
        default=0.025,
        help="Receiver relief as a fraction of the strand's largest extent.",
    )
    parser.add_argument(
        "--ground-width-scale",
        type=float,
        default=2.4,
        help="Half-width of the receiver relative to the strand's largest extent.",
    )
    parser.add_argument(
        "--ground-depth-scale",
        type=float,
        default=0.45,
        help="Half-depth of the receiver relative to the strand's largest extent.",
    )
    parser.add_argument(
        "--ground-screen-height",
        type=float,
        default=0.0,
        help="Target receiver height as a fraction of the orthographic frame; zero uses ground-depth-scale.",
    )
    parser.add_argument(
        "--ground-wave-amplitude",
        type=float,
        default=0.0,
        help="Optional cosine-wave amplitude across the receiver's horizontal axis.",
    )
    parser.add_argument(
        "--ground-wave-frequency",
        type=float,
        default=0.0,
        help="Angular frequency of the optional horizontal receiver wave.",
    )
    parser.add_argument(
        "--ground-wave-phase",
        type=float,
        default=0.0,
        help="Phase of the optional horizontal receiver wave.",
    )
    parser.add_argument(
        "--ground-base-z",
        type=float,
        default=None,
        help="Optional undeformed receiver height; defaults to the mean root height.",
    )
    parser.add_argument(
        "--gaussian-outline-only",
        action="store_true",
        help="Render only wire outlines of Gaussian ellipsoids stored in the input NPZ.",
    )
    parser.add_argument("--gaussian-outline-color", nargs=3, type=float, default=[0.86, 0.66, 0.25])
    parser.add_argument("--gaussian-accent-color", nargs=3, type=float, default=[0.95, 0.68, 0.24])
    parser.add_argument("--gaussian-outline-width", type=float, default=0.0022)
    parser.add_argument("--gaussian-outline-scale", type=float, default=1.0)
    parser.add_argument("--sample-color", nargs=3, type=float, default=[0.92, 0.55, 0.12])
    parser.add_argument(
        "--sample-radius",
        type=float,
        default=0.0,
        help="Radius for optional sample_points stored in the input NPZ; zero chooses an automatic radius.",
    )
    parser.add_argument(
        "--highlight-backward-strands",
        action="store_true",
        help="Highlight strands containing a segment that points against the strand chord.",
    )
    parser.add_argument("--highlight-color", nargs=3, type=float, default=[1.0, 0.0, 0.08])
    parser.add_argument("--highlight-width-scale", type=float, default=4.0)
    parser.add_argument("--mesh-color", nargs=3, type=float, default=[0.52, 0.55, 0.60])
    parser.add_argument("--background-color", nargs=3, type=float, default=[0.18, 0.18, 0.18])
    parser.add_argument("--world-strength", type=float, default=0.85)
    parser.add_argument(
        "--camera-background-strength",
        type=float,
        default=0.0,
        help="Background strength visible to the camera; zero shares world-strength.",
    )
    parser.add_argument("--key-light-energy", type=float, default=900.0)
    parser.add_argument("--key-light-size", type=float, default=1.6)
    parser.add_argument(
        "--key-light-offset",
        nargs=3,
        type=float,
        default=[-0.12, -0.18, 1.8],
        help="Key-light position relative to scene center, in camera-distance units.",
    )
    parser.add_argument(
        "--key-light-type",
        choices=("area", "sun"),
        default="area",
    )
    parser.add_argument("--sun-angle-deg", type=float, default=8.0)
    parser.add_argument("--fill-light-energy", type=float, default=260.0)
    parser.add_argument(
        "--fill-light-size",
        type=float,
        default=2.0,
        help="Fill-light size relative to the fixed presentation extent.",
    )
    parser.add_argument(
        "--shadow-sun-energy",
        type=float,
        default=0.0,
        help="Optional low-energy sun used only to strengthen directional contact shadows.",
    )
    parser.add_argument(
        "--shadow-sun-offset",
        nargs=3,
        type=float,
        default=[0.35, -0.10, 1.80],
        help="Shadow-sun position relative to scene center, in camera-distance units.",
    )
    parser.add_argument("--shadow-sun-angle-deg", type=float, default=6.0)
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
        "--target-root-offset",
        nargs=3,
        type=float,
        default=None,
        help="Camera target relative to the mean strand root; overrides target-offset.",
    )
    parser.add_argument(
        "--coord-system",
        choices=("identity", "tiger_y_up"),
        default="tiger_y_up",
        help="Map exported project coordinates into Blender coordinates. tiger_y_up maps (x,y,z)->(z,x,y).",
    )
    parser.add_argument("--ortho-scale", type=float, default=0.0)
    parser.add_argument(
        "--reference-extent",
        type=float,
        default=0.0,
        help="Fixed presentation extent for receiver and lighting; zero uses geometry bounds.",
    )
    parser.add_argument(
        "--frame-margin",
        type=float,
        default=1.16,
        help="Margin used by automatic camera framing.",
    )
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
    strand_materials: list | None = None,
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
        if strand_materials is None:
            obj.data.materials.append(material)
        else:
            for strand_material in strand_materials[chunk_start:chunk_end]:
                obj.data.materials.append(strand_material)
        bpy.context.collection.objects.link(obj)

        for local_index, (strand, width) in enumerate(
            zip(strands[chunk_start:chunk_end], widths[chunk_start:chunk_end])
        ):
            spl = curve.splines.new("POLY")
            if strand_materials is not None:
                spl.material_index = local_index
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


def make_root_tip_material(
    *,
    name: str,
    root_color: np.ndarray,
    tip_color: np.ndarray,
    roughness: float,
    specular: float,
):
    import bpy

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    hair_info = nodes.new("ShaderNodeHairInfo")
    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MIX"
    mix.inputs[1].default_value = tuple(float(v) for v in root_color) + (1.0,)
    mix.inputs[2].default_value = tuple(float(v) for v in tip_color) + (1.0,)
    links.new(hair_info.outputs["Intercept"], mix.inputs[0])
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return material
    links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = float(roughness)
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = float(specular)
    return material


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


def add_ground_plane(
    *,
    center: np.ndarray,
    root_center: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    max_extent: float,
    material,
    relief_fraction: float,
    width_scale: float,
    depth_scale: float,
    wave_amplitude: float,
    wave_frequency: float,
    wave_phase: float,
    base_z: float | None,
) -> object:
    import bpy

    # Keep the receiver wider than the camera frame so only its shallow crown
    # is visible, never an ellipse or an outer mesh edge.
    margin = max(max_extent * 0.004, 1.0e-4)
    relief = max(max_extent * float(relief_fraction), 0.0)
    dome_center_x = float(root_center[0])
    dome_center_y = float(root_center[1])
    top_z = float(root_center[2] if base_z is None else base_z) - margin
    half_width = max(float(width_scale) * max_extent, 1.0e-5)
    half_depth = max(float(depth_scale) * max_extent, 1.0e-5)
    x_count = 65
    y_count = 25
    vertices: list[tuple[float, float, float]] = []
    for y_index in range(y_count):
        y_unit = -1.0 + 2.0 * y_index / (y_count - 1)
        y = dome_center_y + half_depth * y_unit
        for x_index in range(x_count):
            x_unit = -1.0 + 2.0 * x_index / (x_count - 1)
            x = dome_center_x + half_width * x_unit
            x_relative = (x - dome_center_x) / max(max_extent, 1.0e-8)
            x_profile = float(np.clip(x_relative, -1.25, 1.25))
            if abs(float(wave_amplitude)) > 0.0:
                wave = float(wave_amplitude) * math.cos(
                    float(wave_frequency) * (x - dome_center_x) + float(wave_phase)
                )
                z = top_z + wave - relief * y_unit * y_unit
            else:
                z = top_z - relief * (
                    0.78 * x_profile * x_profile + 0.22 * y_unit * y_unit
                )
            vertices.append((float(x), float(y), float(z)))
    faces: list[tuple[int, ...]] = []
    for y_index in range(y_count - 1):
        for x_index in range(x_count - 1):
            lower_left = y_index * x_count + x_index
            lower_right = lower_left + 1
            upper_left = lower_left + x_count
            upper_right = upper_left + 1
            faces.append((lower_left, lower_right, upper_right, upper_left))
    mesh = bpy.data.meshes.new("strand_shadow_receiver_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    receiver = bpy.data.objects.new("strand_shadow_receiver", mesh)
    receiver.data.materials.append(material)
    bpy.context.collection.objects.link(receiver)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    return receiver


def _orthogonal_frame(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    major = direction / max(float(np.linalg.norm(direction)), 1.0e-8)
    reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(major, reference))) > 0.92:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    transverse_a = np.cross(major, reference)
    transverse_a /= max(float(np.linalg.norm(transverse_a)), 1.0e-8)
    transverse_b = np.cross(major, transverse_a)
    transverse_b /= max(float(np.linalg.norm(transverse_b)), 1.0e-8)
    return major, transverse_a, transverse_b


def add_gaussian_outline_curves(
    *,
    means: np.ndarray,
    directions: np.ndarray,
    scales: np.ndarray,
    material,
    accent_material,
    line_radius: float,
    outline_scale: float,
) -> int:
    import bpy

    curve = bpy.data.curves.new("strand_gaussian_outline_curves", "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_resolution = 2
    curve.bevel_depth = float(line_radius)
    curve.use_path = False
    obj = bpy.data.objects.new("strand_gaussian_outlines", curve)
    obj.data.materials.append(material)
    obj.data.materials.append(accent_material)
    bpy.context.collection.objects.link(obj)

    angles = np.linspace(0.0, 2.0 * np.pi, 49, dtype=np.float32)
    ring_count = 0
    for mean, direction, scale in zip(means, directions, scales):
        major, transverse_a, transverse_b = _orthogonal_frame(direction)
        radii = np.maximum(np.asarray(scale, dtype=np.float32) * float(outline_scale), 1.0e-5)
        rings = (
            (major, transverse_a, float(radii[0]), float(radii[1])),
            (major, transverse_b, float(radii[0]), float(radii[2])),
            (transverse_a, transverse_b, float(radii[1]), float(radii[2])),
        )
        for ring_index, (axis_a, axis_b, radius_a, radius_b) in enumerate(rings):
            points = (
                mean[None, :]
                + np.cos(angles)[:, None] * radius_a * axis_a[None, :]
                + np.sin(angles)[:, None] * radius_b * axis_b[None, :]
            )
            spline = curve.splines.new("POLY")
            spline.material_index = 1 if ring_index == 2 else 0
            spline.points.add(points.shape[0] - 1)
            for spline_point, point in zip(spline.points, points):
                spline_point.co = (float(point[0]), float(point[1]), float(point[2]), 1.0)
            ring_count += 1
    return ring_count


def gaussian_outline_framing_points(
    *,
    means: np.ndarray,
    directions: np.ndarray,
    scales: np.ndarray,
    outline_scale: float,
) -> np.ndarray:
    points: list[np.ndarray] = []
    for mean, direction, scale in zip(means, directions, scales):
        major, transverse_a, transverse_b = _orthogonal_frame(direction)
        radii = np.maximum(np.asarray(scale, dtype=np.float32) * float(outline_scale), 1.0e-5)
        points.extend(
            (
                mean + major * radii[0],
                mean - major * radii[0],
                mean + transverse_a * radii[1],
                mean - transverse_a * radii[1],
                mean + transverse_b * radii[2],
                mean - transverse_b * radii[2],
            )
        )
    return np.asarray(points, dtype=np.float32)


def automatic_ortho_scale(
    *,
    camera,
    target: np.ndarray,
    points: np.ndarray,
    aspect: float,
    margin: float,
) -> float:
    import mathutils

    rotation = camera.rotation_euler.to_matrix()
    right = np.asarray(rotation @ mathutils.Vector((1.0, 0.0, 0.0)), dtype=np.float32)
    up = np.asarray(rotation @ mathutils.Vector((0.0, 1.0, 0.0)), dtype=np.float32)
    relative = points.reshape(-1, 3) - target.reshape(1, 3)
    horizontal = relative @ right
    vertical = relative @ up
    horizontal_span = float(horizontal.max() - horizontal.min())
    vertical_span = float(vertical.max() - vertical.min())
    return max(vertical_span, horizontal_span / max(aspect, 1.0e-6)) * float(margin)


def add_sample_markers(
    *,
    points: np.ndarray,
    material,
    radius: float,
) -> int:
    import bpy

    for index, point in enumerate(points):
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=16,
            ring_count=8,
            radius=radius,
            location=(float(point[0]), float(point[1]), float(point[2])),
        )
        marker = bpy.context.object
        marker.name = f"adaptive_sample_{index:03d}"
        marker.data.materials.append(material)
    return int(points.shape[0])


def main() -> None:
    args = parse_args()
    data = np.load(args.input, allow_pickle=True)
    strands = np.asarray(data["strands"], dtype=np.float32)
    widths = np.asarray(data["widths"], dtype=np.float32)
    colors = np.asarray(data["colors"], dtype=np.float32)
    sample_points = (
        np.asarray(data["sample_points"], dtype=np.float32)
        if "sample_points" in data.files
        else np.empty((0, 3), dtype=np.float32)
    )
    gaussian_means = (
        np.asarray(data["gaussian_means"], dtype=np.float32)
        if "gaussian_means" in data.files
        else np.empty((0, 3), dtype=np.float32)
    )
    gaussian_directions = (
        np.asarray(data["gaussian_directions"], dtype=np.float32)
        if "gaussian_directions" in data.files
        else np.empty((0, 3), dtype=np.float32)
    )
    gaussian_scales = (
        np.asarray(data["gaussian_scales"], dtype=np.float32)
        if "gaussian_scales" in data.files
        else np.empty((0, 3), dtype=np.float32)
    )
    strands, widths, colors = sample_strands(strands, widths, colors, int(args.max_strands), int(args.seed))
    if strands.ndim != 3 or strands.shape[-1] != 3:
        raise RuntimeError(f"strands must be [N,S,3], got {strands.shape}")
    strands = map_coordinates(strands, args.coord_system)
    if sample_points.size:
        if sample_points.ndim != 2 or sample_points.shape[-1] != 3:
            raise RuntimeError(f"sample_points must be [N,3], got {sample_points.shape}")
        sample_points = map_coordinates(sample_points, args.coord_system)
    if gaussian_means.size:
        if gaussian_means.ndim != 2 or gaussian_means.shape[-1] != 3:
            raise RuntimeError(f"gaussian_means must be [N,3], got {gaussian_means.shape}")
        if gaussian_directions.shape != gaussian_means.shape or gaussian_scales.shape != gaussian_means.shape:
            raise RuntimeError("gaussian directions/scales must match gaussian means")
        gaussian_means = map_coordinates(gaussian_means, args.coord_system)
        gaussian_directions = map_coordinates(gaussian_directions, args.coord_system)
        if args.coord_system == "tiger_y_up":
            gaussian_scales = gaussian_scales[..., [0, 2, 1]]
    highlight_mask = (
        backward_strand_mask(strands)
        if bool(args.highlight_backward_strands)
        else np.zeros(strands.shape[0], dtype=bool)
    )

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
    world_nodes = scene.world.node_tree.nodes
    world_links = scene.world.node_tree.links
    if float(args.camera_background_strength) > 0.0:
        world_nodes.clear()
        world_output = world_nodes.new("ShaderNodeOutputWorld")
        environment_bg = world_nodes.new("ShaderNodeBackground")
        camera_bg = world_nodes.new("ShaderNodeBackground")
        light_path = world_nodes.new("ShaderNodeLightPath")
        mix_shader = world_nodes.new("ShaderNodeMixShader")
        background_color = tuple(float(v) for v in args.background_color) + (1.0,)
        environment_bg.inputs["Color"].default_value = background_color
        environment_bg.inputs["Strength"].default_value = float(args.world_strength)
        camera_bg.inputs["Color"].default_value = background_color
        camera_bg.inputs["Strength"].default_value = float(args.camera_background_strength)
        world_links.new(light_path.outputs["Is Camera Ray"], mix_shader.inputs[0])
        world_links.new(environment_bg.outputs["Background"], mix_shader.inputs[1])
        world_links.new(camera_bg.outputs["Background"], mix_shader.inputs[2])
        world_links.new(mix_shader.outputs["Shader"], world_output.inputs["Surface"])
    else:
        bg = world_nodes.get("Background")
        if bg is not None:
            bg.inputs["Color"].default_value = tuple(float(v) for v in args.background_color) + (1.0,)
            bg.inputs["Strength"].default_value = float(args.world_strength)

    mat = bpy.data.materials.new("pure_fur_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is not None:
        color = tuple(float(v) for v in args.material_color) + (1.0,)
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = float(args.material_roughness)
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = 1.0
    strand_materials = None
    if bool(args.use_input_colors):
        if strands.shape[0] > 256:
            raise RuntimeError("--use-input-colors is limited to 256 displayed strands")
        strand_materials = [
            make_root_tip_material(
                name=f"strand_profile_{index:03d}",
                root_color=strand_colors[0],
                tip_color=strand_colors[-1],
                roughness=float(args.material_roughness),
                specular=float(args.material_specular),
            )
            for index, strand_colors in enumerate(colors)
        ]

    highlight_mat = bpy.data.materials.new("highlighted_fur_material")
    highlight_mat.use_nodes = True
    highlight_nodes = highlight_mat.node_tree.nodes
    highlight_bsdf = highlight_nodes.get("Principled BSDF")
    if highlight_bsdf is not None:
        highlight_color = tuple(float(v) for v in args.highlight_color) + (1.0,)
        highlight_bsdf.inputs["Base Color"].default_value = highlight_color
        highlight_bsdf.inputs["Roughness"].default_value = 0.48

    mesh_mat = bpy.data.materials.new("furless_body_material")
    mesh_mat.use_nodes = True
    mesh_nodes = mesh_mat.node_tree.nodes
    mesh_bsdf = mesh_nodes.get("Principled BSDF")
    if mesh_bsdf is not None:
        mesh_bsdf.inputs["Base Color"].default_value = tuple(float(v) for v in args.mesh_color) + (1.0,)
        mesh_bsdf.inputs["Roughness"].default_value = 0.86
    add_mesh_object(args, mesh_mat)

    ground_mat = bpy.data.materials.new("strand_shadow_receiver_material")
    ground_mat.use_nodes = True
    ground_bsdf = ground_mat.node_tree.nodes.get("Principled BSDF")
    if ground_bsdf is not None:
        ground_bsdf.inputs["Base Color"].default_value = tuple(float(v) for v in args.ground_color) + (1.0,)
        ground_bsdf.inputs["Roughness"].default_value = 1.0
        if "Specular IOR Level" in ground_bsdf.inputs:
            ground_bsdf.inputs["Specular IOR Level"].default_value = 0.08

    sample_mat = bpy.data.materials.new("adaptive_sample_material")
    sample_mat.use_nodes = True
    sample_bsdf = sample_mat.node_tree.nodes.get("Principled BSDF")
    if sample_bsdf is not None:
        sample_bsdf.inputs["Base Color"].default_value = tuple(float(v) for v in args.sample_color) + (1.0,)
        sample_bsdf.inputs["Roughness"].default_value = 0.38
        if "Metallic" in sample_bsdf.inputs:
            sample_bsdf.inputs["Metallic"].default_value = 0.18

    gaussian_outline_mat = bpy.data.materials.new("strand_gaussian_outline_material")
    gaussian_outline_mat.use_nodes = True
    gaussian_bsdf = gaussian_outline_mat.node_tree.nodes.get("Principled BSDF")
    if gaussian_bsdf is not None:
        gaussian_bsdf.inputs["Base Color"].default_value = tuple(
            float(v) for v in args.gaussian_outline_color
        ) + (1.0,)
        gaussian_bsdf.inputs["Roughness"].default_value = 0.30
        gaussian_bsdf.inputs["Metallic"].default_value = 0.32
        if "Specular IOR Level" in gaussian_bsdf.inputs:
            gaussian_bsdf.inputs["Specular IOR Level"].default_value = 0.42

    gaussian_accent_mat = bpy.data.materials.new("strand_gaussian_accent_material")
    gaussian_accent_mat.use_nodes = True
    accent_bsdf = gaussian_accent_mat.node_tree.nodes.get("Principled BSDF")
    if accent_bsdf is not None:
        accent_bsdf.inputs["Base Color"].default_value = tuple(
            float(v) for v in args.gaussian_accent_color
        ) + (1.0,)
        accent_bsdf.inputs["Roughness"].default_value = 0.24
        accent_bsdf.inputs["Metallic"].default_value = 0.48
        if "Specular IOR Level" in accent_bsdf.inputs:
            accent_bsdf.inputs["Specular IOR Level"].default_value = 0.48

    base_width = max(float(np.percentile(widths, 60)) * float(args.width_scale), 1.0e-5)
    chunk_size = max(int(args.curve_chunk_size), 1)
    normal_mask = ~highlight_mask
    curve_object_count = 0
    gaussian_outline_ring_count = 0
    if bool(args.gaussian_outline_only):
        if not gaussian_means.size:
            raise RuntimeError("--gaussian-outline-only requires Gaussian arrays in the NPZ")
        gaussian_outline_ring_count = add_gaussian_outline_curves(
            means=gaussian_means,
            directions=gaussian_directions,
            scales=gaussian_scales,
            material=gaussian_outline_mat,
            accent_material=gaussian_accent_mat,
            line_radius=float(args.gaussian_outline_width),
            outline_scale=float(args.gaussian_outline_scale),
        )
    else:
        curve_object_count = add_strand_curve_objects(
            strands=strands[normal_mask],
            widths=widths[normal_mask],
            material=mat,
            base_width=base_width,
            width_scale=float(args.width_scale),
            radius_multiplier=1.0,
            chunk_size=chunk_size,
            name_prefix="white_tiger_pure_fur",
            strand_materials=(
                [strand_materials[index] for index in np.flatnonzero(normal_mask)]
                if strand_materials is not None
                else None
            ),
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
                strand_materials=None,
            )

    bbox_min = strands.reshape(-1, 3).min(axis=0)
    bbox_max = strands.reshape(-1, 3).max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    root_center = strands[:, 0, :].mean(axis=0)
    extent = bbox_max - bbox_min
    geometry_extent = float(np.max(extent))
    max_extent = (
        float(args.reference_extent)
        if float(args.reference_extent) > 0.0
        else geometry_extent
    )
    target = (
        root_center + np.asarray(args.target_root_offset, dtype=np.float32)
        if args.target_root_offset is not None
        else center + np.asarray(args.target_offset, dtype=np.float32)
    )
    presentation_center = target if float(args.reference_extent) > 0.0 else center
    distance = max_extent * 2.6 + 0.5
    sample_marker_count = 0
    if sample_points.size and not bool(args.gaussian_outline_only):
        sample_radius = (
            float(args.sample_radius)
            if float(args.sample_radius) > 0.0
            else max(max_extent * 0.009, 2.0e-4)
        )
        sample_marker_count = add_sample_markers(
            points=sample_points,
            material=sample_mat,
            radius=sample_radius,
        )

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
    if args.ortho_scale > 0:
        cam_data.ortho_scale = float(args.ortho_scale)
    else:
        framing_points = (
            gaussian_outline_framing_points(
                means=gaussian_means,
                directions=gaussian_directions,
                scales=gaussian_scales,
                outline_scale=float(args.gaussian_outline_scale),
            )
            if bool(args.gaussian_outline_only)
            else strands.reshape(-1, 3)
        )
        cam_data.ortho_scale = automatic_ortho_scale(
            camera=cam,
            target=target,
            points=framing_points,
            aspect=float(scene.render.resolution_x) / max(float(scene.render.resolution_y), 1.0),
            margin=float(args.frame_margin),
        )
    scene.camera = cam

    resolved_ground_depth_scale = float(args.ground_depth_scale)
    if bool(args.ground_plane):
        if float(args.ground_screen_height) > 0.0:
            import mathutils

            rotation = cam.rotation_euler.to_matrix()
            camera_up = np.asarray(
                rotation @ mathutils.Vector((0.0, 1.0, 0.0)),
                dtype=np.float32,
            )
            projected_depth = abs(float(camera_up[1]))
            if projected_depth < 1.0e-5:
                raise RuntimeError("ground-screen-height requires a camera with a nonzero top-down component")
            aspect = float(scene.render.resolution_x) / max(
                float(scene.render.resolution_y), 1.0
            )
            camera_frame_height = float(cam_data.ortho_scale) / max(aspect, 1.0)
            target_height = float(args.ground_screen_height) * camera_frame_height
            relief_height = (
                0.22
                * float(args.ground_relief)
                * max_extent
                * abs(float(camera_up[2]))
            )
            half_depth = max(
                (target_height - relief_height) / (2.0 * projected_depth),
                max_extent * 0.05,
            )
            resolved_ground_depth_scale = half_depth / max(max_extent, 1.0e-8)
        add_ground_plane(
            center=center,
            root_center=root_center,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            max_extent=max_extent,
            material=ground_mat,
            relief_fraction=float(args.ground_relief),
            width_scale=float(args.ground_width_scale),
            depth_scale=resolved_ground_depth_scale,
            wave_amplitude=float(args.ground_wave_amplitude),
            wave_frequency=float(args.ground_wave_frequency),
            wave_phase=float(args.ground_wave_phase),
            base_z=(float(args.ground_base_z) if args.ground_base_z is not None else None),
        )

    light_type = "SUN" if args.key_light_type == "sun" else "AREA"
    light_data = bpy.data.lights.new("key_light", light_type)
    light = bpy.data.objects.new("key_light", light_data)
    bpy.context.collection.objects.link(light)
    light_data.energy = float(args.key_light_energy)
    if args.key_light_type == "sun":
        light.location = (
            float(presentation_center[0] + args.key_light_offset[0] * distance),
            float(presentation_center[1] + args.key_light_offset[1] * distance),
            float(presentation_center[2] + args.key_light_offset[2] * distance),
        )
        look_at(light, presentation_center)
        light_data.angle = math.radians(float(args.sun_angle_deg))
    else:
        light.location = (
            float(presentation_center[0] - 0.6 * distance),
            float(presentation_center[1] - 0.8 * distance),
            float(presentation_center[2] + 0.9 * distance),
        )
        light_data.size = max_extent * float(args.key_light_size)

    if float(args.fill_light_energy) > 0.0:
        fill_data = bpy.data.lights.new("soft_fill", "AREA")
        fill = bpy.data.objects.new("soft_fill", fill_data)
        bpy.context.collection.objects.link(fill)
        fill.location = (
            float(presentation_center[0] + 0.7 * distance),
            float(presentation_center[1] + 0.4 * distance),
            float(presentation_center[2] + 0.35 * distance),
        )
        fill_data.energy = float(args.fill_light_energy)
        fill_data.size = max_extent * float(args.fill_light_size)

    if float(args.shadow_sun_energy) > 0.0:
        shadow_sun_data = bpy.data.lights.new("shadow_sun", "SUN")
        shadow_sun = bpy.data.objects.new("shadow_sun", shadow_sun_data)
        bpy.context.collection.objects.link(shadow_sun)
        shadow_sun.location = (
            float(presentation_center[0] + args.shadow_sun_offset[0] * distance),
            float(presentation_center[1] + args.shadow_sun_offset[1] * distance),
            float(presentation_center[2] + args.shadow_sun_offset[2] * distance),
        )
        look_at(shadow_sun, presentation_center)
        shadow_sun_data.energy = float(args.shadow_sun_energy)
        shadow_sun_data.angle = math.radians(float(args.shadow_sun_angle_deg))

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
        "base_width": base_width,
        "curve_chunk_size": chunk_size,
        "curve_object_count": int(curve_object_count),
        "gaussian_outline_only": bool(args.gaussian_outline_only),
        "gaussian_outline_ring_count": int(gaussian_outline_ring_count),
        "ground_plane": bool(args.ground_plane),
        "ground_depth_scale": float(resolved_ground_depth_scale),
        "ground_screen_height": float(args.ground_screen_height),
        "ground_wave_amplitude": float(args.ground_wave_amplitude),
        "ground_wave_frequency": float(args.ground_wave_frequency),
        "ground_wave_phase": float(args.ground_wave_phase),
        "ground_base_z": (
            float(args.ground_base_z) if args.ground_base_z is not None else None
        ),
        "material_roughness": float(args.material_roughness),
        "material_specular": float(args.material_specular),
        "world_strength": float(args.world_strength),
        "camera_background_strength": float(args.camera_background_strength),
        "key_light_type": str(args.key_light_type),
        "key_light_energy": float(args.key_light_energy),
        "fill_light_energy": float(args.fill_light_energy),
        "shadow_sun_energy": float(args.shadow_sun_energy),
        "use_input_colors": bool(args.use_input_colors),
        "sample_marker_count": int(sample_marker_count),
        "camera": args.camera,
        "target_offset": [float(v) for v in args.target_offset],
        "target_root_offset": (
            [float(v) for v in args.target_root_offset]
            if args.target_root_offset is not None
            else None
        ),
        "reference_extent": float(max_extent),
        "geometry_extent": float(geometry_extent),
        "coord_system": args.coord_system,
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
    }
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
