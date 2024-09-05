import sqlite3
from dataclasses import dataclass
from functools import cached_property
from itertools import accumulate
from pathlib import Path
from typing import Literal

from PIL import ImageFont

from .generate_bubbles import generate_bubbles
from .generate_panels import generate_panels
from .generate_text import InvalidFontFile, generate_texts
from .misc_utils import Bbox
from .render_page import RenderContext


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
