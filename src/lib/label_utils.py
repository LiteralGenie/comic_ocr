from pathlib import Path

from PIL import Image, ImageFont
from doctr.io import Document
from doctr.models.predictor import OCRPredictor
import numpy as np
from lib.constants import KOREAN_ALPHABET
from lib.generate_bubbles import generate_bubbles
from lib.generate_panels import generate_panels
from lib.generate_text import InvalidFontFile, generate_texts
from lib.misc_utils import Bbox
from lib.render_page import RenderContext


def make_context(
    font_dir: Path,
    image_dir: Path,
    text_max_bbox_dilation=1,
):
    options = list(font_dir.glob("**/*.otf")) + list(font_dir.glob("**/*.ttf"))
    font_map = {fp.name: fp for fp in options}
    for k, v in list(font_map.items()):
        if not _is_valid_font(v):
            print(f"WARNING: Bad font file: {v}")
            del font_map[k]

    panels, wh = generate_panels()
    panel_map = {p.id: p for p in panels}

    while True:
        bubble_map = {b.id: b for p in panels for b in generate_bubbles(p, font_map)}

        try:
            text_map = {
                t.id: t
                for b in bubble_map.values()
                for t in generate_texts(
                    b,
                    font_map,
                    KOREAN_ALPHABET,
                    max_bbox_dilation=text_max_bbox_dilation,
                )
            }

            break
        except InvalidFontFile as e:
            print(f"Typeset with {e.font_file} failed")
            continue

    ctx = RenderContext(
        font_map,
        image_dir,
        wh,
        panel_map,
        bubble_map,
        text_map,
    )

    return ctx


def _is_valid_font(fp: Path) -> bool:
    try:
        ImageFont.truetype(fp, 12)
        return True
    except:
        return False


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

    matches: list[dict] = []
    for page in output.pages:
        for block in page.blocks:
            for ln in block.lines:
                for w in ln.words:
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
                        y1 + window["bbox"][0],
                        x1 + window["bbox"][1],
                        y2 + window["bbox"][0],
                        x2 + window["bbox"][1],
                    )

                    matches.append(
                        dict(
                            value=w.value,
                            confidence=w.confidence,
                            bbox=bbox,
                        )
                    )

    return dict(
        window=window,
        matches=matches,
    )


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
