import hashlib
import json
import multiprocessing
import sqlite3
import sys
from pathlib import Path
import traceback

from PIL import Image
import numpy as np
from tqdm import tqdm

from lib.label_utils import build_kr_vocab, make_context
from lib.render_page import (
    RenderContext,
    build_render_info,
    render_page,
)

FONT_DIR = Path(sys.argv[1])
IMAGE_DIR = Path(sys.argv[2])
OUT_DIR = Path(sys.argv[3])
NUM_SAMPLES = int(sys.argv[4])

OUT_DIR.mkdir(exist_ok=True)

NUM_WORKERS = 4

WORKER_CTX = dict()


def main():
    db = init_db()

    WORKER_CTX["vocab"] = build_kr_vocab()

    count = 0
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        pbar = tqdm(total=NUM_SAMPLES)
        for d in pool.imap_unordered(make_detection_sample, range(NUM_SAMPLES)):
            if not d:
                continue

            pbar.update()

            img_hash = d["detection"]["label"]["img_hash"]
            fp_out = OUT_DIR / f"{img_hash}.png"

            d["detection"]["sample"].save(fp_out)
            insert_detection_label(db, d["detection"]["label"])

            if count >= NUM_SAMPLES:
                return


def init_db():
    db = sqlite3.connect(OUT_DIR / "det_labels.sqlite")
    db.row_factory = sqlite3.Row

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS labels (
            id      TEXT    PRIMARY KEY,
            data    TEXT    NOT NULL
        )
        """
    )

    db.commit()

    return db


def make_detection_sample(_) -> dict | None:
    try:
        ctx = make_context(FONT_DIR, IMAGE_DIR, WORKER_CTX["vocab"])
        info = build_render_info(ctx)

        sample = render_page(ctx, info)
        label = export_detection_label(ctx, sample)

        return dict(
            ctx=ctx,
            info=info,
            detection=dict(
                sample=sample,
                label=label,
            ),
        )
    except:
        traceback.print_exc()

        return None


def export_detection_label(ctx: RenderContext, im: Image.Image) -> dict:
    img_dimensions = ctx.wh

    arr = np.array(im).astype(np.uint8)
    img_hash = hashlib.sha256(arr.tobytes()).hexdigest()

    polygons = []
    for t in ctx.text_map.values():
        y1, x1, y2, x2 = t.bbox

        x1 = max(x1 - 1, 0)
        x2 = min(x2 + 1, im.size[0] - 1)
        y1 = max(y1 - 1, 0)
        y2 = min(y2 + 1, im.size[1] - 1)

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


def insert_detection_label(db: sqlite3.Connection, label: dict):
    db.execute(
        """
            INSERT INTO labels (
                id, data
            ) VALUES (
                ?, ?
            )
            """,
        [
            label["img_hash"],
            json.dumps(label, indent=2),
        ],
    )

    db.commit()


if __name__ == "__main__":
    main()
