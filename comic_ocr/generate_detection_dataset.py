import argparse
import hashlib
import json
import multiprocessing
import sqlite3
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from lib.config import Config
from lib.label_utils import load_vocab, make_context
from lib.render_page import RenderContext, build_render_info, render_page

WORKER_CTX = dict()


def run(args):
    cfg = Config.load_toml(args.config_file)
    cfg.det_dataset_dir.mkdir(parents=True, exist_ok=True)

    db = init_db(cfg.det_dataset_dir)

    WORKER_CTX["vocab"] = load_vocab(cfg.vocab_file)
    WORKER_CTX["font_dir"] = cfg.font_dir
    WORKER_CTX["image_dir"] = cfg.image_dir

    count = 0
    with multiprocessing.Pool(args.workers) as pool:
        pbar = tqdm(total=args.samples)
        for d in pool.imap_unordered(make_detection_sample, range(args.samples)):
            if not d:
                continue

            pbar.update()

            img_hash = d["detection"]["label"]["img_hash"]
            fp_out = cfg.det_dataset_dir / f"{img_hash}.png"

            d["detection"]["sample"].save(fp_out)
            insert_detection_label(db, d["detection"]["label"])

            count += 1
            if count >= args.samples:
                return


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config_file",
        type=Path,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    return parser.parse_args()


def init_db(fp_dir: Path):
    db = sqlite3.connect(fp_dir / "_det_labels.sqlite")
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
        ctx = make_context(
            WORKER_CTX["font_dir"],
            WORKER_CTX["image_dir"],
            WORKER_CTX["vocab"],
        )
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
    args = parse_args()
    run(args)
