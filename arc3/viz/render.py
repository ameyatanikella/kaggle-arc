"""Render 64x64 ARC frames to PNG (ARC-AGI-3 palette)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

PALETTE = np.array([
    (255, 255, 255), (204, 204, 204), (153, 153, 153), (102, 102, 102), (51, 51, 51), (0, 0, 0),
    (229, 58, 163), (255, 123, 204), (249, 60, 49), (30, 147, 255), (136, 216, 241), (255, 220, 0),
    (255, 133, 27), (146, 18, 49), (79, 204, 48), (163, 86, 214)], dtype=np.uint8)


def to_image(grid: np.ndarray, scale: int = 6) -> Image.Image:
    rgb = PALETTE[np.clip(grid, 0, 15)]
    img = Image.fromarray(rgb, "RGB")
    return img.resize((grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST)


def contact_sheet(grids, titles=None, scale=4, cols=5) -> Image.Image:
    ims = [to_image(g, scale) for g in grids]
    w, h = ims[0].size
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (w + 6), rows * (h + 18)), (40, 40, 40))
    d = ImageDraw.Draw(sheet)
    for i, im in enumerate(ims):
        r, c = divmod(i, cols)
        x, y = c * (w + 6), r * (h + 18)
        sheet.paste(im, (x, y + 16))
        if titles:
            d.text((x + 2, y + 2), str(titles[i]), fill=(255, 255, 255))
    return sheet
