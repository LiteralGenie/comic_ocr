from argparse import Namespace
import argparse
import json
from pathlib import Path
import random
import sqlite3
import sys

from PIL import Image

from lib.detection import train_detection

"""
If this error happens
    UnboundLocalError: local variable 'offset' referenced before assignment
Patch the overflow in
    /venv/lib/python3.10/site-packages/doctr/transforms/modules/pytorch.py
as mentioned here
    https://github.com/mindee/doctr/discussions/1667#discussioncomment-9997544
"""


def run(args):
    args.model_dir.mkdir(exist_ok=True)

    db = sqlite3.connect(args.dataset_dir / "det_labels.sqlite")
    db.row_factory = sqlite3.Row

    labels = {
        f'{r["id"]}.png': json.loads(r["data"])
        for r in db.execute("SELECT id, data FROM labels")
    }

    num_train = int((1 - args.split) * len(labels))

    keys = list(labels.keys())
    random.shuffle(keys)

    train_labels = {k: labels[k] for k in keys[:num_train]}
    val_labels = {k: labels[k] for k in keys[num_train:]}
    print(
        f"Found {len(train_labels)} training samples and {len(val_labels)} validation samples"
    )

    (args.dataset_dir / "train_labels.json").write_text(
        json.dumps(train_labels),
    )
    (args.dataset_dir / "val_labels.json").write_text(
        json.dumps(val_labels),
    )

    train_detection(
        Namespace(
            dataset_path=str(args.dataset_dir),
            save_path=str(args.model_dir),
            #
            arch=args.arch,
            pretrained=True,
            freeze_backbone=False,
            #
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            #
            name=None,
            device=None,
            save_interval_epoch=False,
            input_size=1024,
            weight_decay=0,
            workers=None,
            resume=None,
            test_only=False,
            show_samples=False,
            wb=False,
            push_to_hub=False,
            rotation=False,
            eval_straight=True,
            sched="poly",
            amp=False,
            find_lr=False,
            early_stop=False,
            early_stop_epochs=5,
            early_stop_delta=0.01,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="path to files created by generate_detection_dataset.py",
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="model weights will be saved to this folder",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="db_resnet50",
        help="https://mindee.github.io/doctr/modules/models.html",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.1,
        help="percentage of dataset to reserve for validation",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
