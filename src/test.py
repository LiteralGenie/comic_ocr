import json
from pathlib import Path
from random import randint
import sys

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from lib.constants import HANGUL_SYLLABLES
from lib.generate_bubbles import generate_bubbles
from lib.generate_panels import generate_panels
from lib.generate_text import generate_texts
from lib.label_utils import make_context
from lib.render_page import (
    RenderContext,
    dump_dataclass,
    build_render_info,
    render_page,
)

FONT_DIR = Path(sys.argv[1])
IMAGE_DIR = Path(sys.argv[2])

# random.seed(99)


def main():
    ctx = make_context(FONT_DIR, IMAGE_DIR)
    Path("./ctx.json").write_text(
        json.dumps(
            dump_dataclass(ctx),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    info = build_render_info(ctx)
    Path("./info.json").write_text(
        json.dumps(
            dump_dataclass(info),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    im = render_page(ctx, info)
    im.save("preview.png")
    im.convert("L").save("preview_bw.png")

    preview_layout(ctx).save("layout.png")


def preview_layout(ctx: RenderContext):
    canvas = Image.new("RGBA", ctx.wh, (255, 255, 255, 255))

    canvas = draw_panels(ctx, canvas)
    canvas = draw_bubbles(ctx, canvas)
    canvas = draw_texts(ctx, canvas)

    return canvas


def draw_panels(ctx: RenderContext, canvas: Image.Image):
    for p in ctx.panel_map.values():
        draw = ImageDraw.Draw(canvas)

        draw.rectangle(
            (
                (p.bbox[1], p.bbox[0]),
                (p.bbox[3], p.bbox[2]),
            ),
            (randint(0, 255), randint(0, 255), randint(0, 255), 127),
        )

        canvas_data = np.array(canvas)

        pts = np.array(p.poly)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(canvas_data, [pts], True, (255, 255, 0, 127), 5)
        canvas = Image.fromarray(canvas_data)

    return canvas


def draw_bubbles(ctx: RenderContext, canvas: Image.Image):
    for b in ctx.bubble_map.values():
        canvas_data = np.array(canvas)
        pts = np.array(b.poly)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(canvas_data, [pts], True, (0, 0, 0, 255), 3)
        canvas = Image.fromarray(canvas_data)

    return canvas


def draw_texts(ctx: RenderContext, canvas: Image.Image):
    for t in ctx.text_map.values():
        bubble = ctx.bubble_map[t.id_bubble]
        bubble_center = (bubble.width // 2, bubble.height // 2)

        render = Image.new("RGBA", (bubble.width, bubble.height), (0, 0, 0, 0))
        font = ImageFont.truetype(ctx.font_map[t.font_file], t.font_size)

        x, y = t.xy
        x -= bubble.bbox[1]
        y -= bubble.bbox[0]

        draw = ImageDraw.Draw(render)
        draw.text((x, y), t.letter, font=font, fill=(255, 255, 255, 255))

        render = render.rotate(
            t.angle,
            resample=Image.Resampling.BICUBIC,
            center=bubble_center,
        )

        y1, x1, y2, x2 = t.bbox
        x1 -= bubble.bbox[1]
        x2 -= bubble.bbox[1]
        y1 -= bubble.bbox[0]
        y2 -= bubble.bbox[0]

        draw = ImageDraw.Draw(render)
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0, 255))

        canvas.paste(render, (bubble.bbox[1], bubble.bbox[0]), render)

    return canvas


main()
