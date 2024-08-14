from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont
from doctr.models.predictor import OCRPredictor
import torch
from doctr.models import ocr_predictor, db_resnet50, parseq
from tqdm import tqdm

from lib.constants import KOREAN_ALPHABET
from lib.label_utils import calc_windows, eval_window
from lib.misc_utils import Bbox

TEST_DIR = Path(sys.argv[1])
DET_WEIGHTS = Path(sys.argv[2])
RECO_WEIGHTS = Path(sys.argv[3])
FONT_FILE = Path(sys.argv[4])

FONT_SIZE = 20
LABEL_OFFSET_Y = -20
MIN_CONFIDENCE = 0.25
CROP_SIZE = 1024
MARGIN_SIZE = 100


def main():
    det_model = db_resnet50(pretrained=False, pretrained_backbone=False)
    det_params = torch.load(DET_WEIGHTS, map_location="cpu")
    det_model.load_state_dict(det_params)

    reco_model = parseq(
        vocab=KOREAN_ALPHABET, pretrained=False, pretrained_backbone=False
    )
    reco_params = torch.load(RECO_WEIGHTS, map_location="cpu")
    reco_model.load_state_dict(reco_params)

    predictor = ocr_predictor(
        det_arch=det_model,
        reco_arch=reco_model,
        pretrained=False,
    ).cuda()

    fp_tests = list(TEST_DIR.glob("**/*.png")) + list(TEST_DIR.glob("**/*.jpg"))

    font = ImageFont.truetype(FONT_FILE, FONT_SIZE)

    for fp in fp_tests:
        result = _eval(predictor, fp, font, CROP_SIZE, MARGIN_SIZE)
        result["preview"].save(f"eval_{fp.stem}.png")


def _eval(
    model: OCRPredictor,
    fp: Path,
    font: ImageFont.FreeTypeFont,
    crop_size: int,
    margin_size: int,
) -> dict:
    im = Image.open(fp).convert("RGBA")

    windows = calc_windows(im.size, crop_size, margin_size)

    pbar = tqdm(desc=fp.stem, total=len(windows))

    matches = []
    for w in windows:
        r = eval_window(model, im, w, MIN_CONFIDENCE)

        pbar.update()

        matches.extend(r["matches"])

    matches.sort(key=lambda m: m["confidence"])

    overlay = Image.new("RGBA", im.size)
    draw = ImageDraw.Draw(overlay)
    for m in matches:
        a = int(m["confidence"] * 255)

        width = round(m["confidence"] * 5)

        y1, x1, y2, x2 = m["bbox"]

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=(255, 0, 0, a),
            width=width,
        )

        draw.text(
            (x1, y1 + LABEL_OFFSET_Y),
            m["value"],
            font=font,
            fill=(0, 255, 0, a),
        )

    preview_im = im.copy()
    preview_im.paste(overlay, (0, 0), overlay)

    return dict(
        im=im,
        preview=preview_im,
        matches=matches,
    )


main()
