from argparse import Namespace
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

DATASET_DIR = Path(sys.argv[1])
MODEL_DIR = Path(sys.argv[2])

VAL_SPLIT = 0.1


def main():
    db = sqlite3.connect(DATASET_DIR / "det_labels.sqlite")
    db.row_factory = sqlite3.Row

    labels = {
        f'{r["id"]}.png': json.loads(r["data"])
        for r in db.execute("SELECT id, data FROM labels")
    }
    for k in labels:
        # assert (DATASET_DIR / 'detection' / k).exists(), k
        # assert Image.open(DATASET_DIR / 'detection' / k).load()
        pass

    num_train = int((1 - VAL_SPLIT) * len(labels))

    keys = list(labels.keys())
    random.shuffle(keys)

    train_labels = {k: labels[k] for k in keys[:num_train]}
    val_labels = {k: labels[k] for k in keys[num_train:]}
    print(
        f"Found {len(train_labels)} training samples and {len(val_labels)} validation samples"
    )

    (DATASET_DIR / "train_labels.json").write_text(
        json.dumps(train_labels),
    )
    (DATASET_DIR / "val_labels.json").write_text(
        json.dumps(val_labels),
    )

    train_detection(
        Namespace(
            dataset_path=str(DATASET_DIR),
            save_path=str(MODEL_DIR),
            #
            arch="db_resnet50",
            pretrained=True,
            freeze_backbone=False,
            #
            batch_size=4,
            #
            name=None,
            epochs=10,
            device=None,
            save_interval_epoch=False,
            input_size=1024,
            lr=0.001,
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


main()
