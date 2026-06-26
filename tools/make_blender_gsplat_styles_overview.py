from __future__ import annotations

import json
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


STYLES = ["straight", "wavy", "curly"]
COLUMNS = [
    ("blender_reference_composited.png", "Blender curve groom"),
    ("anigroom_gsplat.png", "AniGroom 3DGS from same strands"),
    ("difference_3x.png", "Pixel difference x3"),
]


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def crop_focus(img: Image.Image) -> Image.Image:
    # Keep the full groom while removing empty margins from the 1920x1080 render.
    w, h = img.size
    left = int(w * 0.17)
    right = int(w * 0.83)
    top = int(h * 0.18)
    bottom = int(h * 0.82)
    return img.crop((left, top, right, bottom))


def fit(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), (240, 240, 240))
    x = (target_w - img.width) // 2
    y = (target_h - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(r"D:\petsgaussianhair\_downloads\blender_vs_anigroom_styles_v2"))
    parser.add_argument("--output-name", default="overview_blender_vs_gsplat_styles.png")
    args = parser.parse_args()
    base = args.base

    font_title = load_font(34)
    font_label = load_font(26)
    font_small = load_font(21)

    cell_w = 760
    cell_h = 430
    left_margin = 170
    top_margin = 105
    label_h = 58
    row_gap = 46
    col_gap = 24

    width = left_margin + len(COLUMNS) * cell_w + (len(COLUMNS) - 1) * col_gap + 60
    height = top_margin + len(STYLES) * (label_h + cell_h) + (len(STYLES) - 1) * row_gap + 60
    sheet = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(sheet)

    draw.text((left_margin, 28), "Blender curve groom -> continuous 3D Gaussian strands", fill=(20, 20, 20), font=font_title)
    for col, (_, title) in enumerate(COLUMNS):
        x = left_margin + col * (cell_w + col_gap)
        draw.text((x + 12, 76), title, fill=(40, 40, 40), font=font_label)

    overview = {}
    for row, style in enumerate(STYLES):
        report_path = base / style / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        overview[style] = report

        y = top_margin + row * (label_h + cell_h + row_gap)
        row_title = (
            f"{style}: {report['strand_count']} strands, "
            f"{report['source_samples']} samples/strand, "
            f"{report.get('segment_mode', 'fixed')} seg {report.get('segment_min', report['segment_count'])}"
            f"-{report.get('segment_max', report['segment_count'])}, "
            f"{report['gaussian_count']} Gaussians, "
            f"mean abs diff {report['mean_abs_pixel_diff']:.4f}"
        )
        draw.text((28, y + 14), style, fill=(25, 75, 145), font=font_label)
        draw.text((left_margin, y + 16), row_title, fill=(70, 70, 70), font=font_small)

        for col, (filename, _) in enumerate(COLUMNS):
            x = left_margin + col * (cell_w + col_gap)
            img = Image.open(base / style / filename)
            img = fit(crop_focus(img), cell_w, cell_h)
            sheet.paste(img, (x, y + label_h))
            draw.rectangle((x, y + label_h, x + cell_w, y + label_h + cell_h), outline=(205, 205, 205), width=2)

    out = base / args.output_name
    sheet.save(out)
    (base / "overview_report.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
