from pathlib import Path

from PIL import Image, ImageDraw

base = Path(r"D:\petsgaussianhair\_downloads\blender_vs_anigroom_gsplat_2k")
out = base / "strand_gsplat_crop_compare.png"
labels = [
    ("Blender reference", "reference_polyline.png"),
    ("AniGroom 3DGS", "anigroom_gsplat.png"),
    ("3x diff", "difference_3x.png"),
]
crop = (620, 260, 1940, 1180)
label_h = 52
border = 18
w = crop[2] - crop[0]
h = crop[3] - crop[1]
sheet = Image.new("RGB", (w * 3 + border * 4, h + label_h + border * 2), (255, 255, 255))
draw = ImageDraw.Draw(sheet)
x = border
for label, name in labels:
    img = Image.open(base / name).convert("RGB").crop(crop)
    sheet.paste(img, (x, border + label_h))
    draw.text((x, border + 14), label, fill=(0, 0, 0))
    x += w + border
sheet.save(out)
print(out)
