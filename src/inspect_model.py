import argparse
from itertools import chain
from pathlib import Path

import torch
from doctr import models
from doctr.models import ocr_predictor
from doctr.models.predictor import OCRPredictor
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from lib.config import Config
from lib.constants import KOREAN_ALPHABET
from lib.label_utils import (
    OcrMatch,
    StitchedBlock,
    StitchedLine,
    calc_windows,
    eval_window,
    stitch_blocks,
    stitch_lines,
)

# @todo: avoid text / bbox collision by checking nearest bbox pos


def run(args):
    cfg = Config.load_toml(args.config_file)
    cfg.debug_dir.mkdir(parents=True, exist_ok=True)

    det_model = models.detection.__dict__[cfg.training.det_arch](
        pretrained=False,
        pretrained_backbone=False,
    )
    det_params = torch.load(
        cfg.training.det_model_dir / args.det_weights,
        map_location="cpu",
    )
    det_model.load_state_dict(det_params)

    reco_model = models.recognition.__dict__[cfg.training.reco_arch](
        vocab=KOREAN_ALPHABET,
        pretrained=False,
        pretrained_backbone=False,
    )
    reco_params = torch.load(
        cfg.training.reco_model_dir / args.reco_weights,
        map_location="cpu",
    )
    reco_model.load_state_dict(reco_params)

    predictor = ocr_predictor(
        det_arch=det_model,
        reco_arch=reco_model,
        pretrained=False,
    ).cuda()

    fp_targets = [
        *args.image_dir.glob("**/*.png"),
        *args.image_dir.glob("**/*.jpg"),
    ]

    font_file = args.font_file
    if not font_file:
        font_file = next(
            chain(
                cfg.training.font_dir.glob("**/*.otf"),
                cfg.training.font_dir.glob("**/*.ttf"),
            )
        )
    font = ImageFont.truetype(font_file, args.font_size)

    for fp in fp_targets:
        result = _eval(
            predictor,
            fp,
            font,
            cfg.training.det_input_size,
            args.margin,
            args.min_confidence,
            args.label_offset_y,
        )
        result["match_preview"].save(cfg.debug_dir / f"{fp.stem}_char.png")
        result["line_preview"].save(cfg.debug_dir / f"{fp.stem}_line.png")
        result["block_preview"].save(cfg.debug_dir / f"{fp.stem}_block.png")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config_file",
        type=Path,
    )
    parser.add_argument(
        "det_weights",
        type=Path,
        help="Filename of detection model weights. File should be located in config.det_model_dir",
    )
    parser.add_argument(
        "reco_weights",
        type=Path,
        help="Filename of recognition model weights. File should be located in config.reco_model_dir",
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Folder containing images to generate predictions for",
    )
    parser.add_argument(
        "--font-file",
        type=Path,
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=20,
    )
    parser.add_argument(
        "--label-offset-y",
        type=int,
        default=20,
        help="Offset text labels by this much from bbox edge. Should be a positive integer.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=100,
        help="Input images are sliced into overlapping windows according to margin size before being fed to the predictor.",
    )

    return parser.parse_args()


def _eval(
    model: OCRPredictor,
    fp: Path,
    font: ImageFont.FreeTypeFont,
    crop_size: int,
    margin_size: int,
    min_confidence: float,
    label_offset_y: int,
) -> dict:
    im = Image.open(fp).convert("RGBA")

    windows = calc_windows(im.size, crop_size, margin_size)

    pbar = tqdm(desc=fp.stem, total=len(windows))

    matches: list[OcrMatch] = []
    for w in windows:
        r = eval_window(model, im, w, min_confidence)

        pbar.update()

        matches.extend(r["matches"])

    matches.sort(key=lambda m: m.confidence)
    match_preview = _draw_matches(
        matches,
        im.copy(),
        font,
        label_offset_y,
    )

    lines = stitch_lines(matches)
    line_preview = _draw_lines(
        lines,
        im.copy(),
        font,
        label_offset_y,
    )

    blocks = stitch_blocks(lines)
    block_preview = _draw_blocks(
        blocks,
        im.copy(),
        font,
        label_offset_y,
    )

    return dict(
        im=im,
        match_preview=match_preview,
        line_preview=line_preview,
        block_preview=block_preview,
        matches=matches,
    )


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
            fill=(255, 0, 0, a),
        )

    im.paste(overlay, (0, 0), overlay)
    return im


def _draw_lines(
    lines: list[StitchedLine],
    im: Image.Image,
    font: ImageFont.FreeTypeFont,
    label_offset_y: int,
):
    overlay = Image.new("RGBA", im.size)
    draw = ImageDraw.Draw(overlay)
    for ln in lines:
        a = int(ln.confidence * 255)
        y1, x1, y2, x2 = ln.bbox

        width = round(ln.confidence * 5)
        draw.rectangle(
            (x1, y1, x2, y2),
            outline=(0, 255, 0, a),
            width=width,
        )

    for ln in lines:
        a = int(ln.confidence * 255)
        y1, x1, y2, x2 = ln.bbox
        draw.text(
            (x1, y1 - label_offset_y),
            ln.value,
            font=font,
            fill=(255, 0, 0, a),
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
            fill=(255, 0, 0, a),
            stroke_width=1,
            stroke_fill=(0, 0, 0, 255),
        )

    im.paste(overlay, (0, 0), overlay)
    return im


if __name__ == "__main__":
    args = parse_args()
    run(args)
