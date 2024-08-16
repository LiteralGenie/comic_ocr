import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from doctr import models
from doctr.models import ocr_predictor
from doctr.models.predictor import OCRPredictor
from PIL import Image, ImageFont
from tqdm import tqdm

from inspect_model import _draw_blocks, _draw_matches
from lib.config import Config
from lib.constants import KOREAN_ALPHABET
from lib.label_utils import (
    OcrMatch,
    calc_windows,
    eval_window,
    stitch_blocks,
    stitch_lines,
)
from lib.render_page import dump_dataclass


@dataclass
class Page:
    im: Image.Image
    fp_im: Path
    matches: list[OcrMatch]


def run(args):
    if not args.image_dir.exists():
        raise Exception(f"image_dir does not exist: {args.image_dir.absolute()}")

    args.out_file = args.out_file or args.image_dir / "ocr_data.json"
    _confirm_overwrite_if_exists(args.out_file)

    cfg = Config.load_toml(args.config_file)
    cfg.debug_dir.mkdir(parents=True, exist_ok=True)

    if args.preview:
        cfg.debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving previews to {cfg.debug_dir.absolute()}")

    det_model = models.detection.__dict__[cfg.det_arch](
        pretrained=False,
        pretrained_backbone=False,
    )
    if cfg.det_weights:
        print(f"Loading detector model weights from {cfg.det_weights.absolute()}")
        det_params = torch.load(
            cfg.det_weights,
            map_location="cpu",
            weights_only=True,
        )
        det_model.load_state_dict(det_params)
    else:
        print(
            f"Loading default weights (pre-trained) for detector network ({cfg.det_arch})"
        )

    reco_model = models.recognition.__dict__[cfg.reco_arch](
        vocab=KOREAN_ALPHABET,
        pretrained=False,
        pretrained_backbone=False,
    )
    if cfg.reco_weights:
        print(f"Loading detector model weights from {cfg.reco_weights.absolute()}")
        reco_params = torch.load(
            cfg.reco_weights,
            map_location="cpu",
            weights_only=True,
        )
        reco_model.load_state_dict(reco_params)
    else:
        print(
            f"Loading default weights (pre-trained) for recognizer network ({cfg.reco_arch})"
        )

    predictor = ocr_predictor(
        det_arch=det_model,
        reco_arch=reco_model,
        pretrained=True,
    ).cuda()

    fp_targets = [
        *args.image_dir.glob("**/*.png"),
        *args.image_dir.glob("**/*.jpg"),
    ]
    fp_targets.sort(key=lambda fp: fp.name)
    print(f"Found {len(fp_targets)} target images.")
    print([fp.name for fp in fp_targets])

    pages: list[Page] = []

    for idx, fp in enumerate(fp_targets):
        im, matches = _eval(
            predictor,
            fp,
            cfg.det_input_size,
            args.margin,
            0,  # args.min_confidence,
            f"({idx + 1} / {len(fp_targets)}) {fp.name}",
            args.resize,
        )

        if args.preview:
            prefix = f"{fp.parent.stem}_{fp.stem}"

            match_preview = _draw_matches(
                matches,
                im.copy(),
                args.preview_font,
                20,
            )
            match_preview.save(cfg.debug_dir / f"{prefix}_match_preview.png")

            lines = stitch_lines(matches)
            blocks = stitch_blocks(lines)
            block_preview = _draw_blocks(
                blocks,
                im.copy(),
                args.preview_font,
                20,
            )
            block_preview.save(cfg.debug_dir / f"{prefix}_bubble_preview.png")

        pages.append(
            Page(
                im=im,
                fp_im=fp,
                matches=matches,
            )
        )

    print(f"Found {len(matches)} words.")

    _dump_data(pages, args.out_file)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config_file",
        type=Path,
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Folder containing images (.jpg, .png) to scan. Output data is ordered alphabetically by file name.",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        help="Path to save the JSON data to. Defaults to reader_ocr.json in the specified image_dir (recommended).",
    )
    # Korean models always return low scores despite being usually correct
    # parser.add_argument(
    #     "--min-confidence",
    #     type=float,
    #     default=0.0,
    #     help="A value between [0, 1]. Useful for filtering matches that are unlikely to be correct (low confidence)."
    # )
    parser.add_argument(
        "--resize",
        type=float,
        help="Multiply image dimensions by this amount (eg 0.25 to shrink image to 1/4 original size). Recommended for images with large fonts (>60pt).",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=100,
        help="Input images are sliced into overlapping windows according to margin size before being fed to the predictor.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Generate preview of extracted text and save to the debug_dir specified in config file.",
    )
    parser.add_argument(
        "--preview-font",
        type=ImageFont.FreeTypeFont,
        default="assets/fonts/unifont/unifont-15.1.05.otf",
        help="Path to font to use for previews.",
    )

    return parser.parse_args()


def _eval(
    model: OCRPredictor,
    fp: Path,
    crop_size: int,
    margin_size: int,
    min_confidence: float,
    desc: str,
    resize_percentage: float | None,
) -> tuple[Image.Image, list[OcrMatch]]:
    im = Image.open(fp).convert("RGBA")
    if resize_percentage:
        w, h = im.size
        w2 = int(w * resize_percentage)
        h2 = int(h * resize_percentage)

        print(f"Resizing {fp.name} from {w}x{h} to {w2}x{h2} before scan")
        im = im.resize((w2, h2), Image.Resampling.BICUBIC)

    windows = calc_windows(im.size, crop_size, margin_size)

    pbar = tqdm(desc=desc, total=len(windows))

    matches: list[OcrMatch] = []
    for w in windows:
        r = eval_window(model, im, w, min_confidence)

        pbar.update()

        matches.extend(r["matches"])

    matches.sort(key=lambda m: m.confidence)

    return im, matches


def _dump_data(
    pages: list[Page],
    fp: Path,
):
    print(f"Saving to {fp.absolute()}")

    data = []

    for pg in pages:
        im_data = np.array(pg.im).astype(np.uint8)
        im_hash = hashlib.sha256(im_data.tobytes()).hexdigest()

        with_ids = [
            dict(
                id=uuid4().hex,
                **dump_dataclass(m),
            )
            for m in pg.matches
        ]

        data.append(
            dict(
                filename=fp.name,
                sha256=im_hash,
                matches=with_ids,
            )
        )

    fp.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )
    )


def _confirm_overwrite_if_exists(fp: Path):
    if fp.exists():
        while True:
            resp = input(
                f"{fp.absolute()} already exists. Overwrite this file? (y / n) "
            ).strip()
            if resp == "y":
                return
            elif resp == "n":
                sys.exit()
            else:
                continue


if __name__ == "__main__":
    args = parse_args()
    run(args)
