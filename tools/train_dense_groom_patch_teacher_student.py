from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from anigroom.grooming import GroomParameterField, GroomRanges  # noqa: E402
from train_groom_layer_teacher_student import (  # noqa: E402
    build_render_inputs,
    image_loss,
    loss_components,
    orientation_detail_loss,
    make_sheet,
    orientation_flow_loss,
    orientation_to_pil,
    psnr,
    render,
    set_range,
    to_pil,
)
from visualize_dense_groom_patch import make_dense_roots, make_field  # noqa: E402


def dense_ranges() -> GroomRanges:
    return GroomRanges(
        length=(0.040, 0.220),
        root_width=(0.000035, 0.00075),
        tip_width_ratio=(0.012, 0.30),
        width_taper=(0.55, 3.20),
        lift=(0.000, 0.080),
        sag=(0.0, 0.75),
        stiffness=(0.05, 0.98),
        curl_radius=(0.0, 0.026),
        curl_frequency=(0.0, 5.5),
        frizz=(0.0, 0.010),
        child_radius=(0.0, 0.012),
        clump_strength=(0.0, 1.0),
        opacity=(0.05, 0.98),
        tip_opacity_ratio=(0.08, 0.90),
    )


def make_student_like_dense(
    root_count: int,
    roots: torch.Tensor,
    ranges: GroomRanges,
    device: torch.device,
    curl_bootstrap: bool,
) -> GroomParameterField:
    student = GroomParameterField(root_count, ranges=ranges, device=device)
    with torch.no_grad():
        set_range(student.length_raw, 0.090, ranges.length)
        set_range(student.root_width_raw, 0.00016, ranges.root_width)
        set_range(student.tip_width_ratio_raw, 0.070, ranges.tip_width_ratio)
        set_range(student.width_taper_raw, 1.80, ranges.width_taper)
        student.flow_xy[:, 0:1].fill_(0.92)
        student.flow_xy[:, 1:2].fill_(-0.12)
        set_range(student.flow_strength_raw, 0.86, ranges.flow_strength)
        set_range(student.lift_raw, 0.026, ranges.lift)
        set_range(student.sag_raw, 0.24, ranges.sag)
        set_range(student.stiffness_raw, 0.72, ranges.stiffness)
        if curl_bootstrap:
            x = roots[:, [0]]
            y = roots[:, [1]]
            set_range(student.curl_radius_raw, 0.0075, ranges.curl_radius)
            set_range(student.curl_frequency_raw, 1.65 + 0.20 * torch.sin(5.0 * y), ranges.curl_frequency)
            student.curl_phase.copy_(5.0 * x + 2.0 * y)
        else:
            set_range(student.curl_radius_raw, 0.0025, ranges.curl_radius)
            set_range(student.curl_frequency_raw, 0.55, ranges.curl_frequency)
        set_range(student.frizz_raw, 0.00035, ranges.frizz)
        set_range(student.child_radius_raw, 0.0022, ranges.child_radius)
        set_range(student.clump_strength_raw, 0.34, ranges.clump_strength)
        set_range(student.opacity_raw, 0.76, ranges.opacity)
        set_range(student.tip_opacity_ratio_raw, 0.35, ranges.tip_opacity_ratio)
    return student


def make_optimizer(
    student: GroomParameterField,
    lr: float,
    curl_lr_scale: float,
    flow_lr_scale: float,
) -> torch.optim.Optimizer:
    curl_names = {"curl_radius_raw", "curl_frequency_raw", "curl_phase", "frizz_raw"}
    flow_names = {"flow_xy", "flow_strength_raw"}
    curl_params = []
    flow_params = []
    base_params = []
    for name, param in student.named_parameters():
        if any(key in name for key in curl_names):
            curl_params.append(param)
        elif any(key in name for key in flow_names):
            flow_params.append(param)
        else:
            base_params.append(param)
    return torch.optim.Adam(
        [
            {"params": base_params, "lr": lr},
            {"params": flow_params, "lr": lr * float(flow_lr_scale)},
            {"params": curl_params, "lr": lr * float(curl_lr_scale)},
        ]
    )


def flow_coherence_loss(field: GroomParameterField, rows: int, cols: int) -> torch.Tensor:
    flow = F.normalize(field.decode().flow_xy, dim=-1, eps=1e-8).view(rows, cols, 2)
    dx = 1.0 - (flow[:, 1:] * flow[:, :-1]).sum(dim=-1).clamp(-1.0, 1.0)
    dy = 1.0 - (flow[1:, :] * flow[:-1, :]).sum(dim=-1).clamp(-1.0, 1.0)
    return 0.5 * (dx.mean() + dy.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\petsgaussianhair\_downloads\dense_groom_patch_teacher_student"))
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=48)
    parser.add_argument("--child-count", type=int, default=4)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=810)
    parser.add_argument("--focal", type=float, default=1725.0)
    parser.add_argument("--samples", type=int, default=72)
    parser.add_argument("--min-segments", type=int, default=18)
    parser.add_argument("--max-segments", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--segment-warmup", type=int, default=60)
    parser.add_argument("--segment-refresh", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.018)
    parser.add_argument("--orientation-weight", type=float, default=0.25)
    parser.add_argument("--orientation-detail-weight", type=float, default=0.20)
    parser.add_argument("--flow-coherence-weight", type=float, default=0.25)
    parser.add_argument("--curl-lr-scale", type=float, default=4.0)
    parser.add_argument("--flow-lr-scale", type=float, default=1.0)
    parser.add_argument("--curl-bootstrap", action="store_true")
    parser.add_argument("--target-style", default="base", choices=["base", "longer", "taper", "curled", "brushed_color"])
    parser.add_argument("--seed", type=int, default=321)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; dense groom training validation must use gsplat")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ranges = dense_ranges()
    roots, normals = make_dense_roots(device, args.rows, args.cols)
    teacher = make_field(args.target_style, roots, ranges).eval()
    student = make_student_like_dense(int(roots.shape[0]), roots, ranges, device, args.curl_bootstrap)

    with torch.no_grad():
        target = render(
            teacher,
            roots,
            normals,
            args.width,
            args.height,
            args.focal,
            args.samples,
            args.child_count,
            None,
            args.min_segments,
            args.max_segments,
        )
        initial = render(
            student,
            roots,
            normals,
            args.width,
            args.height,
            args.focal,
            args.samples,
            args.child_count,
            None,
            args.min_segments,
            args.max_segments,
        )

    optimizer = make_optimizer(student, args.lr, args.curl_lr_scale, args.flow_lr_scale)
    segment_counts: torch.Tensor | None = None
    history: list[dict[str, float | int]] = []

    for iteration in range(1, args.iterations + 1):
        if iteration == 1:
            segment_counts = None
        elif iteration >= args.segment_warmup and (iteration - args.segment_warmup) % args.segment_refresh == 0:
            with torch.no_grad():
                _, segment_counts, _ = build_render_inputs(
                    student,
                    roots,
                    normals,
                    args.samples,
                    args.child_count,
                    None,
                    args.min_segments,
                    args.max_segments,
                )

        optimizer.zero_grad(set_to_none=True)
        pred = render(
            student,
            roots,
            normals,
            args.width,
            args.height,
            args.focal,
            args.samples,
            args.child_count,
            segment_counts,
            args.min_segments,
            args.max_segments,
        )
        components = loss_components(pred, target, args.orientation_weight, args.orientation_detail_weight)
        flow_smooth = flow_coherence_loss(student, args.rows, args.cols)
        loss = components["total"] + float(args.flow_coherence_weight) * flow_smooth
        loss.backward()
        optimizer.step()

        if iteration == 1 or iteration % 20 == 0 or iteration == args.iterations:
            record = {
                "iter": iteration,
                "loss": float(loss.detach().cpu()),
                "rgb_l1": float(components["rgb_l1"].detach().cpu()),
                "rgb_mse": float(components["rgb_mse"].detach().cpu()),
                "alpha_l1": float(components["alpha_l1"].detach().cpu()),
                "orientation_loss": float(components["orientation"].detach().cpu()),
                "orientation_detail_loss": float(components["orientation_detail"].detach().cpu()),
                "flow_coherence_loss": float(flow_smooth.detach().cpu()),
                "psnr": psnr(pred.image, target.image),
                "guide_count": int(pred.stats["guide_count"]),
                "strand_count": int(pred.stats["strand_count"]),
                "gaussian_count": int(pred.stats["gaussian_count"]),
                "segment_min": int(pred.stats["adaptive_min_segments"]),
                "segment_max": int(pred.stats["adaptive_max_segments"]),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    with torch.no_grad():
        final = render(
            student,
            roots,
            normals,
            args.width,
            args.height,
            args.focal,
            args.samples,
            args.child_count,
            segment_counts,
            args.min_segments,
            args.max_segments,
        )
        diff = (final.image - target.image).abs().mul(3.0).clamp(0.0, 1.0)

    grad_report: dict[str, float] = {}
    for name, param in student.named_parameters():
        if param.grad is not None:
            grad_report[name] = float(param.grad.detach().abs().mean().cpu())

    image_paths = [
        args.output_dir / "target.png",
        args.output_dir / "initial.png",
        args.output_dir / "final.png",
        args.output_dir / "diff_x3.png",
    ]
    for path, image in zip(image_paths, [target.image, initial.image, final.image, diff]):
        to_pil(image).save(path)
    make_sheet(image_paths, ["dense target", "student initial", "student final", "final error x3"], args.output_dir / "training_sheet.png")

    orient_paths = [
        args.output_dir / "target_orientation.png",
        args.output_dir / "initial_orientation.png",
        args.output_dir / "final_orientation.png",
    ]
    orientation_to_pil(target.orientation, target.orientation_conf).save(orient_paths[0])
    orientation_to_pil(initial.orientation, initial.orientation_conf).save(orient_paths[1])
    orientation_to_pil(final.orientation, final.orientation_conf).save(orient_paths[2])
    make_sheet(orient_paths, ["dense target orientation", "initial orientation", "final orientation"], args.output_dir / "orientation_sheet.png")

    report = {
        "mode": "dense_groom_patch_teacher_student",
        "target_stats": target.stats,
        "initial_stats": initial.stats,
        "final_stats": final.stats,
        "history": history,
        "initial_psnr": psnr(initial.image, target.image),
        "final_psnr": psnr(final.image, target.image),
        "initial_loss": float(image_loss(initial, target, args.orientation_weight, args.orientation_detail_weight).detach().cpu()),
        "final_loss": float(image_loss(final, target, args.orientation_weight, args.orientation_detail_weight).detach().cpu()),
        "initial_orientation_loss": float(orientation_flow_loss(initial, target).detach().cpu()),
        "final_orientation_loss": float(orientation_flow_loss(final, target).detach().cpu()),
        "initial_orientation_detail_loss": float(orientation_detail_loss(initial, target).detach().cpu()),
        "final_orientation_detail_loss": float(orientation_detail_loss(final, target).detach().cpu()),
        "gradient_abs_mean_last_iter": grad_report,
        "orientation_weight": float(args.orientation_weight),
        "orientation_detail_weight": float(args.orientation_detail_weight),
        "final_flow_coherence_loss": float(flow_coherence_loss(student, args.rows, args.cols).detach().cpu()),
        "flow_coherence_weight": float(args.flow_coherence_weight),
        "curl_lr_scale": float(args.curl_lr_scale),
        "flow_lr_scale": float(args.flow_lr_scale),
        "curl_bootstrap": bool(args.curl_bootstrap),
        "root_grid": {"rows": args.rows, "cols": args.cols},
        "child_count": args.child_count,
        "target_style": args.target_style,
        "image_size": [args.width, args.height],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "initial_psnr": report["initial_psnr"], "final_psnr": report["final_psnr"]}, indent=2))


if __name__ == "__main__":
    main()
