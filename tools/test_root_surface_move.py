"""Verify differentiable face-local root movement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.roots.lifecycle import barycentric_to_points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="D:/petsgaussianhair/_downloads/root_surface_move_20260623")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.12)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vertices = torch.tensor(
        [
            [-1.0, -0.7, 0.0],
            [1.0, -0.7, 0.0],
            [-0.2, 0.9, 0.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    face_ids = torch.zeros(1, dtype=torch.long)
    target_bary = torch.tensor([[0.18, 0.23, 0.59]], dtype=torch.float32)
    target = barycentric_to_points(vertices, faces, face_ids, target_bary).detach()
    logits = torch.nn.Parameter(torch.log(torch.tensor([[0.72, 0.18, 0.10]], dtype=torch.float32)))
    optimizer = torch.optim.Adam([logits], lr=args.lr)
    history = []
    for iteration in range(1, args.iterations + 1):
        bary = torch.softmax(logits, dim=-1)
        point = barycentric_to_points(vertices, faces, face_ids, bary)
        loss = torch.mean((point - target).square())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iteration == 1 or iteration % 20 == 0 or iteration == args.iterations:
            history.append(
                {
                    "iteration": iteration,
                    "loss": float(loss.detach()),
                    "bary": [float(x) for x in bary.detach().reshape(-1)],
                    "point": [float(x) for x in point.detach().reshape(-1)],
                    "bary_sum": float(bary.detach().sum()),
                    "bary_min": float(bary.detach().min()),
                }
            )
    result = {
        "target_bary": [float(x) for x in target_bary.reshape(-1)],
        "target_point": [float(x) for x in target.reshape(-1)],
        "history": history,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["history"][-1]))


if __name__ == "__main__":
    main()
