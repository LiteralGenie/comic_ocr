import sqlite3
from dataclasses import dataclass
from functools import cached_property
from itertools import accumulate
from pathlib import Path
from typing import Literal

import numpy as np
from doctr.io import Document
from doctr.models.predictor import OCRPredictor
from PIL import Image, ImageFont

from lib.generate_bubbles import generate_bubbles
from lib.generate_panels import generate_panels
from lib.generate_text import InvalidFontFile, generate_texts
from lib.misc_utils import Bbox
from lib.render_page import RenderContext


@dataclass
class OcrMatch:
    bbox: Bbox
    confidence: float
    value: str

    @cached_property
    def width(self):
        return self.bbox[3] - self.bbox[1]

    @cached_property
    def height(self):
        return self.bbox[2] - self.bbox[0]

    @cached_property
    def center(self):
        cx = self.bbox[1] + self.width / 2
        cy = self.bbox[0] + self.height / 2
        return cx, cy


@dataclass
class StitchedLine:
    matches: list[OcrMatch]

    @cached_property
    def value(self):
        value = " ".join(m.value for m in self.matches)
        return value

    @cached_property
    def confidence(self):
        return sum([m.confidence for m in self.matches]) / len(self.matches)

    @cached_property
    def bbox(self):
        x1 = min(m.bbox[1] for m in self.matches)
        x2 = max(m.bbox[3] for m in self.matches)
        y1 = min(m.bbox[0] for m in self.matches)
        y2 = max(m.bbox[2] for m in self.matches)
        return (y1, x1, y2, x2)

    @cached_property
    def width(self):
        return self.bbox[3] - self.bbox[1]

    @cached_property
    def height(self):
        return self.bbox[2] - self.bbox[0]

    @cached_property
    def center(self):
        cx = self.bbox[1] + self.width / 2
        cy = self.bbox[0] + self.height / 2
        return cx, cy


@dataclass
class StitchedBlock:
    lines: list[StitchedLine]

    @cached_property
    def value(self):
        value = "\n".join(ln.value for ln in self.lines)
        return value

    @cached_property
    def confidence(self):
        return sum([m.confidence for m in self.lines]) / len(self.lines)

    @cached_property
    def bbox(self):
        x1 = min(m.bbox[1] for m in self.lines)
        x2 = max(m.bbox[3] for m in self.lines)
        y1 = min(m.bbox[0] for m in self.lines)
        y2 = max(m.bbox[2] for m in self.lines)
        return (y1, x1, y2, x2)


def make_context(
    font_dir: Path,
    image_dir: Path,
    vocab: dict[str, int],
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

    words, word_weights = zip(*vocab.items())
    word_weight_acc = list(accumulate(word_weights))

    while True:
        bubble_map = {b.id: b for p in panels for b in generate_bubbles(p, font_map)}

        try:
            text_map = {
                t.id: t
                for b in bubble_map.values()
                for t in generate_texts(
                    b,
                    font_map,
                    words,
                    word_weight_acc,
                    max_bbox_dilation=text_max_bbox_dilation,
                )
            }

            if len(text_map) == 0:
                continue

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
                        y1 + window["bbox"][0],
                        x1 + window["bbox"][1],
                        y2 + window["bbox"][0],
                        x2 + window["bbox"][1],
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


def stitch_lines(
    matches: list[OcrMatch],
    max_edge_dx=0.5,
    max_center_dy=0.25,
) -> list[StitchedLine]:
    """
    Group words into lines

    The maximum edge-to-edge x-distance is (line_height * max_edge_dx)
    The maximum center-to-center y-distance is (line_height * max_center_dy)
    """

    lines: list[list[OcrMatch]] = []

    rem = matches.copy()
    while rem:
        # Pick arbitrary starting point
        base = rem.pop()
        ln = [base]

        while True:
            x1 = min(m.bbox[1] for m in ln)
            x2 = max(m.bbox[3] for m in ln)
            y1 = min(m.bbox[0] for m in ln)
            y2 = max(m.bbox[2] for m in ln)
            bbox = (y1, x1, y2, x2)

            to_add = []
            for idx, m in enumerate(rem):
                dy = abs(base.center[1] - m.center[1])
                if dy > base.height * max_center_dy:
                    continue

                dx = _edge_to_edge_dist(bbox, m.bbox, "x")
                if dx > base.height * max_edge_dx:
                    continue

                to_add.append(idx)

            if not to_add:
                break

            for idx in to_add:
                ln.append(rem[idx])

            rem = [m for idx, m in enumerate(rem) if idx not in to_add]

        ln.sort(key=lambda w: w.bbox[1])
        lines.append(ln)

    stitched_lines = [StitchedLine(ln) for ln in lines]
    return stitched_lines


def stitch_blocks(
    lines: list[StitchedLine],
    max_edge_dx=0.5,
    max_edge_dy=0.5,
) -> list[StitchedBlock]:
    """
    Group lines into blocks

    The maximum edge-to-edge x-distance is (line_height * max_edge_dx)
    The maximum center-to-center y-distance is (line_height * max_center_dy)
    """

    blocks: list[list[StitchedLine]] = []

    rem = lines.copy()
    while rem:
        # Pick arbitrary starting point
        base: StitchedLine = rem.pop()
        blk = [base]

        while True:
            x1 = min(m.bbox[1] for m in blk)
            x2 = max(m.bbox[3] for m in blk)
            y1 = min(m.bbox[0] for m in blk)
            y2 = max(m.bbox[2] for m in blk)
            bbox = (y1, x1, y2, x2)

            to_add = []
            for idx, m in enumerate(rem):
                dy = _edge_to_edge_dist(bbox, m.bbox, "y")
                if dy > base.height * max_edge_dy:
                    continue

                dx = _edge_to_edge_dist(bbox, m.bbox, "x")
                if dx > base.height * max_edge_dx:
                    continue

                to_add.append(idx)

            if not to_add:
                break

            for idx in to_add:
                blk.append(rem[idx])

            rem = [m for idx, m in enumerate(rem) if idx not in to_add]

        blk.sort(key=lambda ln: ln.bbox[0])
        blocks.append(blk)

    stitched_blocks = [StitchedBlock(blk) for blk in blocks]
    return stitched_blocks


def load_vocab(vocab_file: Path, max_freq_frac=0.0003):
    db = sqlite3.connect(vocab_file)
    db.row_factory = sqlite3.Row

    vocab = {r["id"]: r["count"] for r in db.execute("SELECT id, count FROM vocab")}

    max_freq = int(len(vocab) * max_freq_frac)
    vocab = {k: min(v, max_freq) for k, v in vocab.items()}

    total_freq = sum(vocab.values())
    print(
        f"Limiting vocab frequency to {max_freq / total_freq:.3%} ({max_freq:,} / {total_freq:,})",
    )

    return vocab


def _between(start: float, end: float, x: float):
    return x >= start and x <= end


def _any_between(start: float, end: float, xs: list[float]):
    return any(_between(start, end, x) for x in xs)


def _edge_to_edge_dist(a: Bbox, b: Bbox, axis: Literal["x", "y"]) -> float:
    if axis == "x":
        av1, av2 = a[1], a[3]
        bv1, bv2 = b[1], b[3]
    else:
        av1, av2 = a[0], a[2]
        bv1, bv2 = b[0], b[2]

    has_overlap = _any_between(av1, av2, [bv1, bv2])
    has_overlap |= _any_between(bv1, bv2, [av1, av2])
    if has_overlap:
        return 0

    dist = min(
        abs(av1 - bv2),
        abs(bv1 - av2),
    )
    return dist
