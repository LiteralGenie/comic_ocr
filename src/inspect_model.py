import argparse
from itertools import chain
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from doctr.models.predictor import OCRPredictor
import torch
from doctr.models import ocr_predictor, db_resnet50, parseq
from tqdm import tqdm

from lib.config import Config
from lib.constants import KOREAN_ALPHABET
from lib.label_utils import (
    OcrMatch,
    StitchedWord,
    calc_windows,
    eval_window,
    stitch_words,
)


def run(args):
    config = Config.load_toml(args.config_file)
    config.debug_dir.mkdir(parents=True, exist_ok=True)

    det_model = db_resnet50(pretrained=False, pretrained_backbone=False)
    det_params = torch.load(
        config.det_model_dir / args.det_weights,
        map_location="cpu",
    )
    det_model.load_state_dict(det_params)

    reco_model = parseq(
        vocab=KOREAN_ALPHABET,
        pretrained=False,
        pretrained_backbone=False,
    )
    reco_params = torch.load(
        config.reco_model_dir / args.reco_weights,
        map_location="cpu",
    )
    reco_model.load_state_dict(reco_params)

    predictor = ocr_predictor(
        det_arch=det_model,
        reco_arch=reco_model,
        pretrained=False,
    ).cuda()

    fp_tests = [
        *args.test_dir.glob("**/*.png"),
        *args.test_dir.glob("**/*.jpg"),
    ]

    font_file = args.font_file
    if not font_file:
        font_file = next(
            chain(
                config.font_dir.glob("**/*.otf"),
                config.font_dir.glob("**/*.ttf"),
            )
        )
    font = ImageFont.truetype(font_file, args.font_size)

    for fp in fp_tests:
        result = _eval(
            predictor,
            fp,
            font,
            config.det_input_size,
            args.margin,
            args.min_confidence,
            args.label_offset_y,
        )
        result["char_preview"].save(config.debug_dir / f"{fp.stem}_char.png")
        # result["word_preview"].save(config.debug_dir / f"{fp.stem}_word.png")


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
        "test_dir",
        type=Path,
        help="Images to generate predictions for",
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
        default=-20,
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
    char_preview = _draw_chars(
        matches,
        im.copy(),
        font,
        label_offset_y,
    )

    words = stitch_words(matches)
    word_preview = _draw_words(
        words,
        im.copy(),
        font,
        label_offset_y,
    )

    return dict(
        im=im,
        char_preview=char_preview,
        word_preview=word_preview,
        matches=matches,
    )


def _draw_chars(
    matches: list[OcrMatch],
    im: Image.Image,
    font: ImageFont.FreeTypeFont,
    label_offset_y: int,
):
    overlay = Image.new("RGBA", im.size)
    draw = ImageDraw.Draw(overlay)
    for m in matches:
        a = int(m.confidence * 255)

        width = round(m.confidence * 5)

        y1, x1, y2, x2 = m.bbox

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=(255, 0, 0, a),
            width=width,
        )

        draw.text(
            (x1, y1 + label_offset_y),
            m.value,
            font=font,
            fill=(0, 255, 0, a),
        )

    im.paste(overlay, (0, 0), overlay)
    return im


def _draw_words(
    words: list[StitchedWord],
    im: Image.Image,
    font: ImageFont.FreeTypeFont,
    label_offset_y: int,
):
    overlay = Image.new("RGBA", im.size)
    draw = ImageDraw.Draw(overlay)
    for w in words:
        a = int(w.confidence * 255)

        width = round(w.confidence * 5)

        y1, x1, y2, x2 = w.bbox

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=(255, 0, 0, a),
            width=width,
        )

        draw.text(
            (x1, y1 + label_offset_y),
            w.value,
            font=font,
            fill=(0, 255, 0, a),
        )

    im.paste(overlay, (0, 0), overlay)
    return im


if __name__ == "__main__":
    args = parse_args()
    run(args)
