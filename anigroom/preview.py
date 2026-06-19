from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _fit(points: np.ndarray, axes: tuple[int, int], size: tuple[int, int], pad: int) -> tuple[np.ndarray, float]:
    xy = points[:, list(axes)]
    center = xy.mean(axis=0)
    span = np.maximum(xy.max(axis=0) - xy.min(axis=0), 1e-5)
    scale = min((size[0] - 2 * pad) / span[0], (size[1] - 2 * pad) / span[1])
    return center.astype(np.float32), float(scale)


def _project(points: np.ndarray, fit: tuple[np.ndarray, float], axes: tuple[int, int], size: tuple[int, int], pad: int) -> np.ndarray:
    center, scale = fit
    xy = points[:, list(axes)]
    xy = (xy - center[None, :]) * scale
    xy[:, 0] += size[0] * 0.5
    xy[:, 1] = size[1] * 0.5 - xy[:, 1]
    return np.clip(xy, pad, [size[0] - pad, size[1] - pad])


def draw_curve_projection(
    curves: np.ndarray,
    width: np.ndarray,
    color: np.ndarray,
    alpha: np.ndarray,
    title: str,
    out: str | Path,
    axes: tuple[int, int] = (2, 1),
    size: tuple[int, int] = (1280, 900),
    max_draw_roots: int = 6000,
) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bg = np.array([248, 248, 246], dtype=np.float32)
    img = Image.new("RGB", size, tuple(int(x) for x in bg))
    draw = ImageDraw.Draw(img)
    fit = _fit(curves.reshape(-1, 3), axes, size, 28)
    stride = max(1, curves.shape[0] // max_draw_roots)
    for i in range(0, curves.shape[0], stride):
        pts = _project(curves[i], fit, axes, size, 28)
        a = float(np.clip(alpha[i, 0, 0], 0.02, 1.0))
        for j in range(len(pts) - 1):
            rgb = np.clip(color[i, j] * 255, 0, 255).astype(np.float32)
            shown = bg * (1.0 - a) + rgb * a
            line_w = max(1, int(np.clip(width[i, j, 0] * 1100, 1, 3)))
            draw.line([tuple(pts[j]), tuple(pts[j + 1])], fill=tuple(int(x) for x in shown), width=line_w)
    draw.text((16, 14), title, fill=(20, 20, 20))
    img.save(out)


def draw_proxy_projection(
    proxy: dict[str, np.ndarray],
    title: str,
    out: str | Path,
    axes: tuple[int, int] = (2, 1),
    size: tuple[int, int] = (1280, 900),
    max_draw_segments: int = 30000,
) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mean = proxy["mean"]
    chord = proxy["chord"]
    color = proxy["color"]
    width = proxy["width"]
    alpha = proxy["alpha"]
    starts = mean - 0.5 * chord
    ends = mean + 0.5 * chord
    all_pts = np.concatenate([starts.reshape(-1, 3), ends.reshape(-1, 3)], axis=0)
    fit = _fit(all_pts, axes, size, 28)
    bg = np.array([248, 248, 246], dtype=np.float32)
    img = Image.new("RGB", size, tuple(int(x) for x in bg))
    draw = ImageDraw.Draw(img)
    flat_n = mean.shape[0] * mean.shape[1]
    stride = max(1, flat_n // max_draw_segments)
    idx = 0
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            if idx % stride == 0:
                pts = _project(np.stack([starts[i, j], ends[i, j]], axis=0), fit, axes, size, 28)
                a = float(np.clip(alpha[i, j, 0], 0.02, 1.0))
                rgb = np.clip(color[i, j] * 255, 0, 255).astype(np.float32)
                shown = bg * (1.0 - a) + rgb * a
                line_w = max(1, int(np.clip(width[i, j, 0] * 1250, 1, 3)))
                draw.line([tuple(pts[0]), tuple(pts[1])], fill=tuple(int(x) for x in shown), width=line_w)
            idx += 1
    draw.text((16, 14), title, fill=(20, 20, 20))
    img.save(out)
