from pathlib import Path
import random
import sys
import cv2
from cv2.typing import MatLike
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from lib.constants import HANGUL_SYLLABLES
from lib.generate_bubbles import Bubble, generate_bubbles
from lib.generate_panels import Panel, generate_panels
from lib.generate_text import Text, generate_texts

FONT_DIR = Path(sys.argv[1])

# random.seed(99)


def main():
    panels, wh = generate_panels()
    panel_map = {p.id: p for p in panels}

    canvas = np.zeros((wh[1], wh[0], 3), np.uint8)
    canvas.fill(255)
    canvas = draw_panels(canvas, panel_map)

    bubble_map = {b.id: b for p in panels for b in generate_bubbles(p)}
    canvas = draw_bubbles(canvas, bubble_map)

    fp_fonts = list(FONT_DIR.glob("**/*.ttf"))
    text_map = {
        t.id: t
        for b in bubble_map.values()
        for t in generate_texts(
            b,
            fp_fonts,
            HANGUL_SYLLABLES,
        )
    }
    canvas = draw_texts(canvas, bubble_map, text_map)

    cv2.imwrite("./tmp.png", canvas)


def draw_panels(canvas: MatLike, panel_map: dict[str, Panel]):
    for p in panel_map.values():
        cv2.rectangle(
            canvas,
            (p.bbox[1], p.bbox[0]),
            (p.bbox[3], p.bbox[2]),
            (0, 255, 255),
            3,
        )

        pts = np.array(p.poly)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], True, (0, 0, 0), 5)

    return canvas


def draw_bubbles(canvas: MatLike, bubbles: dict[str, Bubble]):
    for b in bubbles.values():
        pts = np.array(b.poly)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], True, (0, 255, 0), 3)

    return canvas


def draw_texts(
    canvas: MatLike,
    bubble_map: dict[str, Bubble],
    text_map: dict[str, Text],
):
    pil_canvas = Image.fromarray(canvas).convert("RGBA")

    for t in text_map.values():
        bubble = bubble_map[t.id_bubble]
        bubble_center = (
            bubble.bbox[1] + bubble.width // 2,
            bubble.bbox[0] + bubble.height // 2,
        )

        render = Image.new("RGBA", (canvas.shape[1], canvas.shape[0]), (0, 0, 0, 0))
        font = ImageFont.truetype(t.fp_font, t.font_size)

        draw = ImageDraw.Draw(render)
        draw.text(t.xy, t.letter, font=font, fill=(0, 0, 255))

        render = render.rotate(
            t.angle,
            resample=Image.Resampling.BICUBIC,
            center=bubble_center,
        )

        y1, x1, y2, x2 = t.bbox
        draw = ImageDraw.Draw(render)
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0))

        pil_canvas = Image.alpha_composite(pil_canvas, render)

    return np.array(pil_canvas.convert("RGB"))


main()
