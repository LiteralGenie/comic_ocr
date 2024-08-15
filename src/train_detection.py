import argparse
import json
import random
import sqlite3
from argparse import Namespace
from pathlib import Path

from lib.config import Config
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
    cfg = Config.load_toml(args.config_file)
    cfg.det_dataset_dir.mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect(cfg.det_dataset_dir / "_det_labels.sqlite")
    db.row_factory = sqlite3.Row

    labels = {
        f'{r["id"]}.png': json.loads(r["data"])
        for r in db.execute("SELECT id, data FROM labels")
    }

    fp_train = cfg.det_dataset_dir / "_train_labels.json"
    fp_val = cfg.det_dataset_dir / "_val_labels.json"

    if args.resume_path:
        print("Resuming from", args.resume_path)

        old_train_labels = (
            json.loads(fp_train.read_text()) if fp_train.exists() else dict()
        )
        old_val_labels = json.loads(fp_val.read_text()) if fp_val.exists() else dict()

        print(
            f"Found {len(old_train_labels)} existing training samples and {len(old_val_labels)} existing validation samples"
        )

        train_labels, val_labels = update_labels(
            args,
            labels,
            old_train_labels,
            old_val_labels,
        )
    else:
        train_labels, val_labels = generate_labels(args, labels)

    print(
        f"Training with {len(train_labels)} training samples and {len(val_labels)} validation samples"
    )

    fp_train.write_text(json.dumps(train_labels))
    fp_val.write_text(json.dumps(val_labels))

    train_detection(
        Namespace(
            dataset_path=str(cfg.det_dataset_dir),
            save_path=str(cfg.det_model_dir),
            train_labels_path=str(fp_train),
            val_labels_path=str(fp_val),
            #
            arch=cfg.det_arch,
            pretrained=True,
            freeze_backbone=False,
            #
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            resume=args.resume_path,
            #
            name=None,
            device=None,
            save_interval_epoch=False,
            input_size=1024,
            weight_decay=0,
            workers=None,
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


def generate_labels(
    args,
    labels: dict[str, dict],
) -> tuple[dict[str, dict], dict[str, dict]]:
    num_train = int((1 - args.split) * len(labels))

    keys = list(labels.keys())
    random.shuffle(keys)

    train_labels = {k: labels[k] for k in keys[:num_train]}
    val_labels = {k: labels[k] for k in keys[num_train:]}

    return train_labels, val_labels


def update_labels(
    args,
    labels: dict[str, dict],
    train_labels: dict[str, dict],
    val_labels: dict[str, dict],
) -> tuple[dict[str, dict], dict[str, dict]]:
    new_keys = [k for k in labels if k not in train_labels and k not in val_labels]
    random.shuffle(new_keys)

    idx_split = int((1 - args.split) * len(new_keys))

    new_train = {k: labels[k] for k in new_keys[:idx_split]}
    new_train.update(
        {k: labels[k] for k in labels if k in train_labels}
    )  # move still-existing keys to new version (ignore deleted ones)

    new_val = {k: labels[k] for k in new_keys[idx_split:]}
    new_val.update({k: labels[k] for k in labels if k in val_labels})

    return new_train, new_val


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config_file",
        type=Path,
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
        "--resume-path",
        dest="resume_path",
        type=Path,
        default=None,
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
