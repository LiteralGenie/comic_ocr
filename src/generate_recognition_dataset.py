import multiprocessing
import sqlite3
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from lib.label_utils import make_context
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

NUM_WORKERS = 8


def main():
    db = init_db()

    with multiprocessing.Pool(NUM_WORKERS) as pool:
        pbar = tqdm(total=NUM_SAMPLES)
        for d in pool.imap_unordered(make_recognition_sample, range(NUM_SAMPLES)):
            pbar.update()

            for data in d["recognition"]:
                fp_out = OUT_DIR / f"{data['id']}.png"
                data["sample"].save(fp_out)
                insert_recognition_label(db, data)


def init_db():
    db = sqlite3.connect(OUT_DIR / "reco_labels.sqlite")
    db.row_factory = sqlite3.Row

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS labels (
            id      TEXT    PRIMARY KEY,
            label   TEXT    NOT NULL
        )
        """
    )

    db.commit()

    return db


def make_recognition_sample(_) -> dict:
    ctx = make_context(FONT_DIR, IMAGE_DIR)
    info = build_render_info(ctx)

    det_sample = render_page(ctx, info)

    reco_data = export_recognition_labels(ctx, det_sample)

    return dict(
        ctx=ctx,
        info=info,
        recognition=reco_data,
    )


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
            INSERT INTO labels (
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
