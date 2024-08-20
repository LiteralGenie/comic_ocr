import argparse
import multiprocessing
import sqlite3
import traceback
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from lib.config import Config
from lib.label_utils import load_vocab, make_context
from lib.render_page import RenderContext, build_render_info, render_page

WORKER_CTX = dict()


def run(args):
    cfg = Config.load_toml(args.config_file)
    cfg.training.reco_dataset_dir.mkdir(parents=True, exist_ok=True)

    db = init_db(cfg.training.reco_dataset_dir)

    WORKER_CTX["vocab"] = load_vocab(cfg.training.vocab_file)
    WORKER_CTX["font_dir"] = cfg.training.font_dir
    WORKER_CTX["image_dir"] = cfg.training.image_dir

    count = 0
    with multiprocessing.Pool(args.workers) as pool:
        pbar = tqdm(total=args.samples)
        for grp in pool.imap_unordered(make_recognition_sample, range(args.samples)):
            if not grp:
                continue

            for d in grp["recognition"]:
                pbar.update()
                count += 1

                fp_out = cfg.training.reco_dataset_dir / f"{d['id']}.png"
                d["sample"].save(fp_out)
                insert_recognition_label(db, d)

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
    db = sqlite3.connect(fp_dir / "_reco_labels.sqlite")
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


def make_recognition_sample(_) -> dict | None:
    try:
        ctx = make_context(
            WORKER_CTX["font_dir"],
            WORKER_CTX["image_dir"],
            WORKER_CTX["vocab"],
        )
        info = build_render_info(ctx)

        sample = render_page(ctx, info)

        reco_data = export_recognition_labels(ctx, sample)

        return dict(
            ctx=ctx,
            info=info,
            recognition=reco_data,
        )
    except:
        traceback.print_exc()
        return None


def export_recognition_labels(ctx: RenderContext, im: Image.Image) -> list[dict]:
    labels = []

    for id, t in ctx.text_map.items():
        y1, x1, y2, x2 = t.bbox

        x1 = max(x1 - 1, 0)
        x2 = min(x2 + 1, im.size[0] - 1)
        y1 = max(y1 - 1, 0)
        y2 = min(y2 + 1, im.size[1] - 1)

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
    args = parse_args()
    run(args)
