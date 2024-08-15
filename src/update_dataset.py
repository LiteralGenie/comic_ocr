import argparse
import sqlite3
from pathlib import Path

from lib.config import Config


def det_deletion_scan(cfg: Config):
    db = sqlite3.connect(cfg.det_dataset_dir / "_det_labels.sqlite")
    db.row_factory = sqlite3.Row

    ids = [r["id"] for r in db.execute("SELECT id FROM labels")]

    missing = [
        f"{id}" for id in ids if not (cfg.det_dataset_dir / f"{id}.png").exists()
    ]

    if not missing:
        return

    tmp = input(f"Delete {len(missing)} / {len(ids)} label rows (y)? ")
    if tmp != "y":
        print("Quitter.")
        return

    for k in missing:
        db.execute("""DELETE FROM labels WHERE id=?""", [k])

    db.commit()


def det_insertion_scan(cfg: Config, fp_import: Path):
    db = sqlite3.connect(cfg.det_dataset_dir / "_det_labels.sqlite")
    db.row_factory = sqlite3.Row

    existing = {r["id"] for r in db.execute("SELECT id FROM labels")}

    import_db = sqlite3.connect(fp_import)
    import_db.row_factory = sqlite3.Row

    labels = {
        r["id"]: r["data"] for r in import_db.execute("SELECT id, data FROM labels")
    }
    to_add = []
    for k in labels:
        if k not in existing and (cfg.det_dataset_dir / f"{k}.png").exists():
            to_add.append(k)

    if not to_add:
        return

    tmp = input(f"Import {len(to_add)} label rows (y)? ")
    if tmp != "y":
        print("Quitter.")
        return

    for k in to_add:
        db.execute(
            """INSERT INTO labels (id, data) VALUES (?, ?)""",
            [k, labels[k]],
        )

    db.commit()


def reco_deletion_scan(cfg: Config):
    db = sqlite3.connect(cfg.reco_dataset_dir / "_reco_labels.sqlite")
    db.row_factory = sqlite3.Row

    ids = [r["id"] for r in db.execute("SELECT id FROM labels")]

    missing = [
        f"{id}" for id in ids if not (cfg.reco_dataset_dir / f"{id}.png").exists()
    ]

    if not missing:
        return

    tmp = input(f"Delete {len(missing)} / {len(ids)} label rows (y)? ")
    if tmp != "y":
        print("Quitter.")
        return

    for k in missing:
        db.execute("""DELETE FROM labels WHERE id=?""", [k])

    db.commit()


def reco_insertion_scan(cfg: Config, fp_import: Path):
    db = sqlite3.connect(cfg.reco_dataset_dir / "_reco_labels.sqlite")
    db.row_factory = sqlite3.Row

    existing = {r["id"] for r in db.execute("SELECT id FROM labels")}

    import_db = sqlite3.connect(fp_import)
    import_db.row_factory = sqlite3.Row

    labels = {
        r["id"]: r["label"] for r in import_db.execute("SELECT id, label FROM labels")
    }
    to_add = []
    for k in labels:
        if k not in existing and (cfg.reco_dataset_dir / f"{k}.png").exists():
            to_add.append(k)

    if not to_add:
        return

    tmp = input(f"Import {len(to_add)} label rows (y)? ")
    if tmp != "y":
        print("Quitter.")
        return

    for k in to_add:
        db.execute(
            """INSERT INTO labels (id, label) VALUES (?, ?)""",
            [k, labels[k]],
        )

    db.commit()


def run(args):
    config = Config.load_toml(args.config_file)

    if (fp_import := getattr(args, "import")) and not fp_import.exists():
        raise Exception(f"Database to import from does not exist: {fp_import}")

    if args.type in ["detection", "det"]:
        det_deletion_scan(config)

        if fp := getattr(args, "import"):
            det_insertion_scan(config, fp)
    else:
        reco_deletion_scan(config)

        if fp := getattr(args, "import"):
            reco_insertion_scan(config, fp)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config_file",
        type=Path,
    )
    parser.add_argument(
        "type",
        choices=[
            "detection",
            "det",
            "recognition",
            "reco",
        ],
    )
    parser.add_argument(
        "--import",
        type=Path,
        help="SQLite database to import labels from. Images for these labels should already be located in the dataset directory.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
