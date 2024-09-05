import numpy as np
from doctr.io import Document
from doctr.models.predictor import OCRPredictor
from PIL import Image, ImageDraw, ImageFont

from .label_utils import OcrMatch, StitchedBlock, StitchedLine
from .misc_utils import Bbox


def calc_windows(
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


def eval_window(
    model: OCRPredictor,
    im: Image.Image,
    window: dict,
    min_confidence: float,
) -> dict:
    y1, x1, y2, x2 = window["bbox"]
    crop_im = im.crop((x1, y1, x2, y2))
    crop_data = np.asarray(crop_im.convert("RGB"))

    output: Document = model([crop_data])

    matches: list[OcrMatch] = []
    for page in output.pages:
        for block in page.blocks:
            for ln in block.lines:
                for w in ln.words:
                    if not w.value:
                        continue
                    if w.confidence < min_confidence:
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

                    bbox = (
                        int(y1 + window["bbox"][0]),
                        int(x1 + window["bbox"][1]),
                        int(y2 + window["bbox"][0]),
                        int(x2 + window["bbox"][1]),
                    )

                    matches.append(
                        OcrMatch(
                            bbox=bbox,
                            confidence=w.confidence,
                            value=w.value,
                        )
                    )

    return dict(
        window=window,
        matches=matches,
    )


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


def draw_matches(
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


def draw_lines(
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


def draw_blocks(
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
