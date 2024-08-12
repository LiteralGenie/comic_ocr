from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont
from doctr.io import Document
from doctr.models.predictor import OCRPredictor
import numpy as np
import torch
from doctr.models import ocr_predictor, db_resnet50, parseq
from tqdm import tqdm

from lib.constants import KOREAN_ALPHABET
from lib.misc_utils import Bbox

TEST_DIR = Path(sys.argv[1])
DET_WEIGHTS = Path(sys.argv[2])
RECO_WEIGHTS = Path(sys.argv[3])
FONT_FILE = Path(sys.argv[4])

FONT_SIZE = 20
LABEL_OFFSET_Y = -20
MIN_CONFIDENCE = 0.5
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
    )

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

    windows = _calc_windows(im.size, crop_size, margin_size)

    results = []
    pbar = tqdm(desc=fp.stem, total=len(windows))

    for w in windows:
        r = _eval_window(model, im, font, w)
        results.append(r)

        pbar.update()

    preview_im = Image.new("RGB", im.size)
    for r in results:
        y1, x1, y2, x2 = r["window"]["bbox_cov"]
        offset = (x1 - r["window"]["bbox"][1], y1 - r["window"]["bbox"][0])
        crop = offset + r["preview"].size
        preview_im.paste(r["preview"].crop(crop), (x1, y1))

    return dict(
        im=im,
        preview=preview_im,
        matches=[m for r in results for m in r["matches"]],
    )


def _eval_window(
    model: OCRPredictor,
    im: Image.Image,
    font: ImageFont.FreeTypeFont,
    window: dict,
) -> dict:
    y1, x1, y2, x2 = window["bbox"]
    crop_im = im.crop((x1, y1, x2, y2))
    crop_data = np.asarray(crop_im.convert("RGB"))

    preview_im = crop_im.copy()

    output: Document = model([crop_data])

    matches: list[dict] = []
    for page in output.pages:
        for block in page.blocks:
            for ln in block.lines:
                for w in ln.words:
                    if w.confidence < MIN_CONFIDENCE:
                        continue

                    ((x1, y1), (x2, y2)) = w.geometry
                    x1 *= crop_im.size[0]
                    x2 *= crop_im.size[0]
                    y1 *= crop_im.size[1]
                    y2 *= crop_im.size[1]

                    center_x = window["bbox"][1] + x1 + (x2 - x1) / 2
                    center_y = window["bbox"][0] + y1 + (y2 - y1) / 2
                    if not _is_contained(window["bbox_cov"], (center_x, center_y)):
                        continue

                    canvas = Image.new("RGBA", crop_im.size)
                    draw = ImageDraw.Draw(canvas)

                    a = int(w.confidence * 255)

                    width = round(w.confidence * 5)

                    draw.rectangle(
                        (x1, y1, x2, y2),
                        outline=(255, 0, 0, a),
                        width=width,
                    )

                    draw.text(
                        (x1, y1 + LABEL_OFFSET_Y),
                        w.value,
                        font=font,
                        fill=(0, 255, 0, a),
                    )

                    preview_im.paste(canvas, (0, 0), canvas)

                    bbox = (
                        y1 + window["bbox"][0],
                        x1 + window["bbox"][1],
                        y2 + window["bbox"][2],
                        x2 + window["bbox"][3],
                    )

                    matches.append(
                        dict(
                            value=w.value,
                            confidence=w.confidence,
                            bbox=bbox,
                        )
                    )

    return dict(
        crop=crop_im,
        preview=preview_im,
        window=window,
        matches=matches,
    )


def _calc_windows(
    wh: tuple[int, int],
    crop_size: int,
    margin_size: int,
) -> list[dict]:
    windows = []

    window_xs = _calc_dim_windows(wh[0], crop_size, margin_size)
    window_ys = _calc_dim_windows(wh[1], crop_size, margin_size)

    for y in window_ys:
        for x in window_xs:
            y1, y2 = y["ivl"]
            x1, x2 = x["ivl"]
            bbox = [y1, x1, y2, x2]

            y1_cov, y2_cov = y["ivl_cov"]
            x1_cov, x2_cov = x["ivl_cov"]
            bbox_cov = [y1_cov, x1_cov, y2_cov, x2_cov]

            windows.append(
                dict(
                    # region to crop
                    bbox=bbox,
                    # sub-region to extract bbox's from
                    # any chars detected outside this region will be ignored
                    bbox_cov=bbox_cov,
                )
            )

    return windows


def _calc_dim_windows(dim_size: int, crop_size: int, margin_size: int):
    # max size of each window, minus margins
    # each window will have total size of crop_size,
    #   but bbox's that fall in margins (outside this central coverage area) will be ignored
    max_window_coverage = crop_size - 2 * margin_size

    num_windows = (dim_size - 2 * margin_size) / max_window_coverage

    if num_windows >= 1:
        cov_sizes = []

        for _ in range(int(num_windows)):
            cov_sizes.append(max_window_coverage)

        rem = num_windows % 1
        if rem > 0:
            cov_sizes.append(int(rem * max_window_coverage))
    else:
        cov_sizes = [dim_size - 2 * margin_size]

    windows = []
    curr_window_pos = 0
    curr_cov_pos = margin_size

    for s in cov_sizes:
        ivl = (curr_window_pos, curr_window_pos + s + 2 * margin_size)
        ivl_cov = (curr_cov_pos, curr_cov_pos + s)
        windows.append(
            dict(
                ivl=ivl,
                ivl_cov=ivl_cov,
            )
        )

        curr_window_pos += s
        curr_cov_pos += s

    fst = windows[0]["ivl_cov"]
    windows[0]["ivl_cov"] = (fst[0] - margin_size, fst[1])

    lst = windows[-1]["ivl_cov"]
    windows[-1]["ivl_cov"] = (lst[0], lst[1] + margin_size)

    return windows


def _is_contained(bbox: Bbox, xy: tuple[float, float]):
    y1, x1, y2, x2 = bbox
    x, y = xy

    if x < x1 or x > x2:
        return False

    if y < y1 or y > y2:
        return False

    return True


main()
