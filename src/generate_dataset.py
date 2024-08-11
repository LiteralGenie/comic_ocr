import hashlib
import json
import multiprocessing
import sqlite3
import sys
from pathlib import Path

from PIL import Image
import numpy as np
from tqdm import tqdm

from lib.constants import HANGUL_SYLLABLES, KOREAN_ALPHABET
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
NUM_SAMPLES = int(sys.argv[4])

DETECTION_OUT_DIR = OUT_DIR / "detection"
DETECTION_OUT_DIR.mkdir(exist_ok=True)

RECOGNITION_OUT_DIR = OUT_DIR / "recognition"
RECOGNITION_OUT_DIR.mkdir(exist_ok=True)

NUM_WORKERS = 8


def main():
    db = init_db()

    with multiprocessing.Pool(NUM_WORKERS) as pool:
        pbar = tqdm(total=NUM_SAMPLES)
        for d in pool.imap_unordered(make_sample, range(NUM_SAMPLES)):
            pbar.update()

            img_hash = d["detection"]["label"]["img_hash"]
            fp_out = DETECTION_OUT_DIR / f"{img_hash}.png"

            d["detection"]["sample"].save(fp_out)
            insert_detection_label(db, d["detection"]["label"])

            for data in d["recognition"]:
                fp_out = RECOGNITION_OUT_DIR / f"{data['id']}.png"
                data["sample"].save(fp_out)
                insert_recognition_label(db, data)


def init_db():
    db = sqlite3.connect(OUT_DIR / "labels.sqlite")
    db.row_factory = sqlite3.Row

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS detection_labels (
            id      TEXT    PRIMARY KEY,
            data    TEXT    NOT NULL
        )
        """
    )

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS recognition_labels (
            id      TEXT    PRIMARY KEY,
            label   TEXT    NOT NULL
        )
        """
    )

    db.commit()

    return db


def make_sample(_) -> dict:
    ctx = make_context()
    info = build_render_info(ctx)

    det_sample = render_page(ctx, info)
    det_label = export_detection_label(ctx, det_sample)

    reco_data = export_recognition_labels(ctx, det_sample)

    return dict(
        ctx=ctx,
        info=info,
        detection=dict(
            sample=det_sample,
            label=det_label,
        ),
        recognition=reco_data,
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
            KOREAN_ALPHABET,
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


def export_detection_label(ctx: RenderContext, im: Image.Image) -> dict:
    img_dimensions = ctx.wh

    arr = np.array(im).astype(np.uint8)
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


def insert_detection_label(db: sqlite3.Connection, label: dict):
    db.execute(
        """
            INSERT INTO detection_labels (
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


def export_recognition_labels(ctx: RenderContext, im: Image.Image) -> list[dict]:
    labels = []

    for id, t in ctx.text_map.items():
        y1, x1, y2, x2 = t.bbox

        labels.append(
            dict(
                id=id,
                sample=im.crop((x1, y1, x2, y2)),
                label=t.letter,
            )
        )

    return labels


def insert_recognition_label(db: sqlite3.Connection, data: dict):
    db.execute(
        """
            INSERT INTO recognition_labels (
                id, label
            ) VALUES (
                ?, ?
            )
            """,
        [
            data["id"],
            data["label"],
        ],
    )

    db.commit()


if __name__ == "__main__":
    main()
