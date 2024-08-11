import hashlib
import json
import multiprocessing
import sys
from pathlib import Path

from PIL import Image
import numpy as np

from lib.constants import HANGUL_SYLLABLES
from lib.generate_bubbles import generate_bubbles
from lib.generate_panels import generate_panels
from lib.generate_text import generate_texts
from lib.render_page import (
    RenderContext,
    build_render_info,
    render_page,
)

FONT_DIR = Path(sys.argv[1])
IMAGE_DIR = Path(sys.argv[2])
OUT_DIR = Path(sys.argv[3])

OUT_DIR.mkdir(exist_ok=True)
[fp.unlink() for fp in OUT_DIR.glob("**/*")]

NUM_SAMPLES = 3  # 0_000
NUM_WORKERS = 8


def main():
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        labels = dict()

        for d in pool.imap_unordered(make_sample, range(NUM_SAMPLES)):
            img_hash = d["det_label"]["img_hash"]
            fp_out = OUT_DIR / f"{img_hash}.png"

            labels[img_hash] = d["det_label"]

            d["sample"].save(fp_out)

    fp_labels = OUT_DIR / "_labels.json"
    fp_labels.write_text(json.dumps(labels, indent=2))


def make_sample(_) -> dict:
    ctx = make_context()
    info = build_render_info(ctx)

    sample = render_page(ctx, info)

    det_label = export_detection_label(ctx, sample)

    return dict(
        ctx=ctx,
        info=info,
        sample=sample,
        det_label=det_label,
    )


def make_context():
    font_map = {fp.name: fp for fp in FONT_DIR.glob("**/*.ttf")}

    panels, wh = generate_panels()
    panel_map = {p.id: p for p in panels}

    bubble_map = {b.id: b for p in panels for b in generate_bubbles(p)}

    text_map = {
        t.id: t
        for b in bubble_map.values()
        for t in generate_texts(
            b,
            font_map,
            HANGUL_SYLLABLES,
        )
    }

    ctx = RenderContext(
        font_map,
        IMAGE_DIR,
        wh,
        panel_map,
        bubble_map,
        text_map,
    )

    return ctx


def export_detection_label(ctx: RenderContext, im: Image.Image):
    img_dimensions = ctx.wh

    arr = np.array(im).astype(np.uint16)
    img_hash = hashlib.sha256(arr.tobytes()).hexdigest()

    polygons = []
    for t in ctx.text_map.values():
        y1, x1, y2, x2 = t.bbox

        pts = [
            (x1, y1),
            (x1, y2),
            (x2, y2),
            (x2, y1),
        ]
        pts = [(int(x), int(y)) for x, y in pts]

        polygons.append(pts)

    return dict(
        img_dimensions=img_dimensions,
        img_hash=img_hash,
        polygons=polygons,
    )


if __name__ == "__main__":
    main()
