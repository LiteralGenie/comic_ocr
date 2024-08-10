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

# random.seed(1)


def main():
    panels, wh = generate_panels()
    panel_map = {p.id: p for p in panels}

    canvas = np.zeros((wh[1], wh[0], 3), np.uint8)
    canvas.fill(255)
    canvas = draw_panels(canvas, panel_map)

    bubble_map = {p.id: generate_bubbles(p) for p in panels}
    canvas = draw_bubbles(canvas, bubble_map)

    fp_fonts = list(FONT_DIR.glob("**/*.ttf"))
    text_map = {
        b.id: generate_texts(
            b,
            fp_fonts,
            HANGUL_SYLLABLES,
        )
        for grp in bubble_map.values()
        for b in grp
    }
    canvas = draw_texts(canvas, text_map)

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


def draw_bubbles(canvas: MatLike, bubbles: dict[str, list[Bubble]]):
    for grp in bubbles.values():
        for b in grp:
            pts = np.array(b.poly)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], True, (0, 255, 0), 3)

    return canvas


def draw_texts(canvas: MatLike, texts: dict[str, list[Text]]):
    pil_canvas = Image.fromarray(canvas)

    for grp in texts.values():
        for t in grp:
            font = ImageFont.truetype(t.fp_font, t.font_size)

            draw = ImageDraw.Draw(pil_canvas)
            draw.text(t.xy, t.letter, font=font, fill=(0, 0, 255))

    return np.array(pil_canvas)


main()
