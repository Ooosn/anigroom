import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="Directory containing frame_*.png")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=float, default=18.0)
    parser.add_argument("--glob", default="frame_*.png")
    parser.add_argument("--loop", action="store_true", help="Append the first frame at the end for a closed orbit")
    args = parser.parse_args()

    frame_dir = Path(args.frames)
    paths = sorted(frame_dir.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"no frames matched {args.glob} under {frame_dir}")
    if args.loop:
        paths = paths + [paths[0]]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    first = Image.open(paths[0]).convert("RGB")
    width, height = first.size
    writer = cv2.VideoWriter(
        str(out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {out}")
    try:
        for path in paths:
            img = Image.open(path).convert("RGB").resize((width, height))
            arr = np.asarray(img)
            writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    print(str(out.resolve()))


if __name__ == "__main__":
    main()
