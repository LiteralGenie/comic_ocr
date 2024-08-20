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
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from inspect_model import _draw_blocks, _draw_matches
from lib.config import Config
from lib.constants import KOREAN_ALPHABET
from lib.label_utils import (
    OcrMatch,
    StitchedBlock,
    calc_windows,
    eval_window,
    stitch_blocks,
    stitch_lines,
)
from lib.render_page import dump_dataclass


@dataclass
class Page:
    im: Image.Image
    filename: str
    matches: list[OcrMatch]


def ocr(
    image_dir: Path,
    det_arch="db_resnet50",
    det_weights: Path | None = None,
    det_input_size=1024,
    reco_arch="parseq",
    reco_weights: Path | None = None,
    out_file: Path | None = None,
    margin=100,
    cpu=False,
    resize: float | None = None,
    preview_dir: Path | None = None,
    preview_font: Path | None = None,
    preview_font_size: float = 20,
):
    if not image_dir.exists():
        raise Exception(f"image_dir does not exist: {image_dir.absolute()}")

    out_file = out_file or image_dir / "ocr_data.json"
    _confirm_overwrite_if_exists(out_file)

    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving previews to {preview_dir.absolute()}")

    det_model = models.detection.__dict__[det_arch](
        pretrained=False,
        pretrained_backbone=False,
    )
    if det_weights:
        print(f"Loading detector model weights from {det_weights.absolute()}")
        det_params = torch.load(
            det_weights,
            map_location="cpu",
            weights_only=True,
        )
        det_model.load_state_dict(det_params)
    else:
        print(
            f"Loading default weights (pre-trained) for detector network ({det_arch})"
        )

    reco_model = models.recognition.__dict__[reco_arch](
        vocab=KOREAN_ALPHABET,
        pretrained=False,
        pretrained_backbone=False,
    )
    if reco_weights:
        print(f"Loading detector model weights from {reco_weights.absolute()}")
        reco_params = torch.load(
            reco_weights,
            map_location="cpu",
            weights_only=True,
        )
        reco_model.load_state_dict(reco_params)
    else:
        print(
            f"Loading default weights (pre-trained) for recognizer network ({reco_arch})"
        )

    predictor = ocr_predictor(
        det_arch=det_model,
        reco_arch=reco_model,
        pretrained=True,
    )
    if not cpu:
        if torch.cuda.is_available():
            predictor = predictor.cuda()
        else:
            print("No GPU detected, running models on CPU")

    fp_targets = [
        *image_dir.glob("**/*.png"),
        *image_dir.glob("**/*.jpg"),
    ]
    fp_targets.sort(key=lambda fp: fp.name)
    print(f"Found {len(fp_targets)} target images.")
    print([fp.name for fp in fp_targets])

    pages: list[Page] = []

    for idx, fp in enumerate(fp_targets):
        im = _load_im(fp, resize)

        matches = _eval(
            predictor,
            im,
            det_input_size,
            margin,
            0,  # args.min_confidence,
            f"({idx + 1} / {len(fp_targets)}) {fp.name}",
        )

        if preview_dir:
            font = ImageFont.truetype(
                preview_font,
                preview_font_size,
            )

            prefix = f"{fp.parent.stem}_{fp.stem}"

            match_preview = _draw_matches(
                matches,
                im.copy(),
                font,
                20,
            )
            match_preview.save(preview_dir / f"{prefix}_match_preview.png")

            lines = stitch_lines(matches)
            blocks = stitch_blocks(lines)
            block_preview = _draw_blocks(
                blocks,
                im.copy(),
                font,
                20,
            )
            block_preview.save(preview_dir / f"{prefix}_bubble_preview.png")

        pages.append(
            Page(
                im=im,
                filename=fp.name,
                matches=matches,
            )
        )

    print(f"Found {len(matches)} words.")

    _dump_data(pages, out_file)


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
        "--cpu",
        action="store_true",
        default=False,
        help="Run models on CPU instead of GPU.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Generate preview of extracted text and save to the debug_dir specified in config file.",
    )
    parser.add_argument(
        "--preview-font",
        type=Path,
        default="assets/fonts/unifont/unifont-15.1.05.otf",
        help="Path to font to use for previews.",
    )
    parser.add_argument(
        "--preview-font-size",
        type=float,
        default=20,
    )

    return parser.parse_args()


def _load_im(
    fp: Path,
    resize_percentage: float | None,
):
    im = Image.open(fp).convert("RGBA")
    if resize_percentage:
        w, h = im.size
        w2 = int(w * resize_percentage)
        h2 = int(h * resize_percentage)

        print(f"Resizing {fp.name} from {w}x{h} to {w2}x{h2} before scan")
        im = im.resize((w2, h2), Image.Resampling.BICUBIC)

    return im


def _eval(
    model: OCRPredictor,
    im: Image.Image,
    crop_size: int,
    margin_size: int,
    min_confidence: float,
    desc: str,
) -> list[OcrMatch]:
    windows = calc_windows(im.size, crop_size, margin_size)

    pbar = tqdm(desc=desc, total=len(windows))

    matches: list[OcrMatch] = []
    for w in windows:
        r = eval_window(model, im, w, min_confidence)

        pbar.update()

        matches.extend(r["matches"])

    matches.sort(key=lambda m: m.confidence)

    return matches


def _dump_data(
    pages: list[Page],
    fp: Path,
):
    print(f"Saving to {fp.absolute()}")

    data = dict()

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

        data[pg.filename] = dict(
            filename=pg.filename,
            sha256=im_hash,
            matches=with_ids,
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


def _draw_matches(
    matches: list[OcrMatch],
    im: Image.Image,
    font: ImageFont.FreeTypeFont,
    label_offset_y: int,
):
    overlay = Image.new("RGBA", im.size)
    draw = ImageDraw.Draw(overlay)
    for m in matches:
        a = int(m.confidence * 255)
        y1, x1, y2, x2 = m.bbox

        width = round(m.confidence * 5)
        draw.rectangle(
            (x1, y1, x2, y2),
            outline=(0, 255, 0, a),
            width=width,
        )

    for m in matches:
        a = int(m.confidence * 255)
        y1, x1, y2, x2 = m.bbox
        draw.text(
            (x1, y1 - label_offset_y),
            m.value,
            font=font,
            fill=(255, 50, 50, a),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 255),
        )

    im.paste(overlay, (0, 0), overlay)
    return im


def _draw_blocks(
    blocks: list[StitchedBlock],
    im: Image.Image,
    font: ImageFont.FreeTypeFont,
    label_offset_y: int,
):
    overlay = Image.new("RGBA", im.size)
    draw = ImageDraw.Draw(overlay)
    for blk in blocks:
        a = int(blk.confidence * 255)
        y1, x1, y2, x2 = blk.bbox

        width = round(blk.confidence * 5)
        draw.rectangle(
            (x1, y1, x2, y2),
            outline=(0, 255, 0, a),
            width=width,
        )

    for blk in blocks:
        a = int(blk.confidence * 255)
        y1, x1, y2, x2 = blk.bbox
        draw.text(
            (x1, y2 + label_offset_y),
            blk.value,
            font=font,
            fill=(255, 50, 0, a),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 255),
        )

    im.paste(overlay, (0, 0), overlay)
    return im


if __name__ == "__main__":
    args = parse_args()

    cfg = Config.load_toml(args.config_file)

    kwargs = {
        k: v
        for k, v in vars(args).items()
        if k
        not in [
            "message_type",
            "config_file",
            "preview",
        ]
    }

    ocr(
        det_arch=cfg.det_arch,
        det_weights=cfg.det_weights,
        det_input_size=cfg.det_input_size,
        reco_arch=cfg.reco_arch,
        reco_weights=cfg.reco_weights,
        preview_dir=cfg.debug_dir if args.preview else None,
        **kwargs,
    )
