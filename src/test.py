import json
from pathlib import Path
import pickle
from random import randint
import sys

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from lib.constants import HANGUL_SYLLABLES
from lib.generate_bubbles import generate_bubbles
from lib.generate_panels import generate_panels
from lib.generate_text import generate_texts
from lib.render_page import (
    PageRenderInfo,
    RenderContext,
    build_render_info,
    render_page,
)

FONT_DIR = Path(sys.argv[1])
IMAGE_DIR = Path(sys.argv[2])

# random.seed(99)


def main():
    ctx = make_context()
    # with open(Path("./ctx.pkl"), "rb") as file:
    #     ctx: RenderContext = pickle.load(file)
    #     Path("./ctx.json").write_text(json.dumps(ctx.dump(), indent=2))
    preview_layout(ctx).save("layout.png")

    info = make_render_info(ctx)
    # with open(Path("./info.pkl"), "rb") as file:
    #     info: PageRenderInfo = pickle.load(file)

    im = render_page(ctx, info)
    im.save("preview.png")
    im.convert("L").save("preview_bw.png")


def make_context():
    font_map = {fp.name: fp for fp in FONT_DIR.glob("**/*.ttf")}

    panels, wh = generate_panels()
    panel_map = {p.id: p for p in panels}
    print(f"{len(panel_map)} panels")

    bubble_map = {b.id: b for p in panels for b in generate_bubbles(p)}
    print(f"{len(bubble_map)} bubbles")

    text_map = {
        t.id: t
        for b in bubble_map.values()
        for t in generate_texts(
            b,
            font_map,
            HANGUL_SYLLABLES,
        )
    }
    print(f"{len(text_map)} characters")

    ctx = RenderContext(
        font_map,
        IMAGE_DIR,
        wh,
        panel_map,
        bubble_map,
        text_map,
    )
    with open(Path("./ctx.pkl"), "wb") as file:
        pickle.dump(ctx, file)

    return ctx


def make_render_info(ctx: RenderContext):
    info = build_render_info(ctx)
    Path("./info.pkl").write_text(json.dumps(info.dump(), indent=2))
    with open(Path("./info.pkl"), "wb") as file:
        pickle.dump(info, file)

    return info


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
            (randint(0, 255), randint(0, 255), randint(0, 255), 255),
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
        bubble_center = (
            bubble.bbox[1] + bubble.width // 2,
            bubble.bbox[0] + bubble.height // 2,
        )

        render = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        font = ImageFont.truetype(ctx.font_map[t.font_file], t.font_size)

        draw = ImageDraw.Draw(render)
        draw.text(t.xy, t.letter, font=font, fill=(255, 255, 255, 255))

        render = render.rotate(
            t.angle,
            resample=Image.Resampling.BICUBIC,
            center=bubble_center,
        )

        y1, x1, y2, x2 = t.bbox
        draw = ImageDraw.Draw(render)
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0))

        canvas = Image.alpha_composite(canvas, render)

    return canvas


main()
