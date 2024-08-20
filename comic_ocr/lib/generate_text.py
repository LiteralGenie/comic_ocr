import random
from bisect import bisect
from dataclasses import dataclass
from pathlib import Path
from random import randint
from uuid import uuid4

import cv2
import numpy as np
from cv2.typing import MatLike
from PIL import Image, ImageDraw, ImageFont

from lib.generate_bubbles import Bubble
from lib.misc_utils import Bbox, dilate


class InvalidFontFile(Exception):
    def __init__(self, font_file: Path):
        self.font_file = font_file


@dataclass
class Text:
    id: str
    id_bubble: str
    letter: str
    xy: tuple[int, int]
    font_size: float
    bbox: Bbox
    angle: int


def generate_texts(
    bubble: Bubble,
    font_map: dict[str, Path],
    vocab: list[str],
    vocab_weight_acc: list[int],
    max_tries=500,
    min_font_size=20,
    max_font_size=50,
    min_angle=-20,
    max_angle=20,
    max_bbox_dilation=0,
    mask_dilation=2,
):
    mask = np.zeros((bubble.height, bubble.width, 3), np.uint8)
    mask.fill(255)

    pts = [
        (
            x - bubble.bbox[1],
            y - bubble.bbox[0],
        )
        for x, y in bubble.poly
    ]
    pts = np.array(pts)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], (0, 0, 0))

    # pad by 1 on each side to detect overflow
    mask = np.pad(mask, ((1, 1), (1, 1), (0, 0)), "constant", constant_values=255)

    texts: list[Text] = []
    for idx in range(max_tries):
        word_idx = random.random() * vocab_weight_acc[-1]
        word = vocab[bisect(vocab_weight_acc, word_idx)]

        font_size = randint(min_font_size, max_font_size)
        x = randint(0, bubble.width)
        y = randint(0, bubble.height)
        angle = randint(min_angle, max_angle)

        render = Image.new("RGB", (mask.shape[1], mask.shape[0]))

        try:
            fp_font = font_map[bubble.font_file]
            font = ImageFont.truetype(fp_font, font_size)
            draw = ImageDraw.Draw(render)
            draw.text((x + 1, y + 1), word, font=font, fill=(255, 255, 255))
        except OSError:
            raise InvalidFontFile(fp_font)

        if _touches_edge(render):
            continue

        render = render.rotate(angle, fillcolor=(0, 0, 0))
        if _touches_edge(render):
            continue

        render = np.array(render)

        # check if letter extends outside bubble or intersects with existing letters
        render_dilated = dilate(render.copy(), mask_dilation)

        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        intersection = cv2.bitwise_and(render_dilated, mask, mask=mask_gray)
        is_valid = np.all(intersection == 0)
        if not is_valid:
            continue

        mask = cv2.bitwise_or(render_dilated, mask)

        try:
            y1, x1, y2, x2 = _get_bbox(render)

            x1 += bubble.bbox[1] - 1
            x2 += bubble.bbox[1] - 1
            y1 += bubble.bbox[0] - 1
            y2 = min(y2, bubble.bbox[2] - 1)

            if max_bbox_dilation > 0:
                x1 -= randint(1, max_bbox_dilation)
                x2 += randint(1, max_bbox_dilation)
                y1 -= randint(1, max_bbox_dilation)
                y2 += randint(1, max_bbox_dilation)

            x1 = max(x1, 0)
            x2 = min(x2, bubble.bbox[3] - 1)
            y1 = max(y1, 0)
            y2 += bubble.bbox[0] - 1

            bbox = (y1, x1, y2, x2)
        except:
            # render is empty or has 1-pixel height / width
            continue

        texts.append(
            Text(
                uuid4().hex,
                bubble.id,
                word,
                (x + bubble.bbox[1], y + bubble.bbox[0]),
                font_size,
                bbox,
                angle,
            )
        )

    return texts


def _get_bbox(im: MatLike) -> Bbox:
    h, w, _ = im.shape

    data = np.array(im)

    for x1 in range(w):
        if np.any(data[:, x1, :] != (0, 0, 0)):
            break

    for x2 in range(w - 1, -1, -1):
        if np.any(data[:, x2, :] != (0, 0, 0)):
            break

    for y1 in range(h):
        if np.any(data[y1, :, :] != (0, 0, 0)):
            break

    for y2 in range(h - 1, -1, -1):
        if np.any(data[y2, :, :] != (0, 0, 0)):
            break

    assert y1 < y2  # type: ignore
    assert x1 < x2  # type: ignore
    return (y1, x1, y2, x2)


def _touches_edge(im: Image.Image):
    w, h = im.size

    data = np.array(im)

    if np.any(data[0, :, :] != (0, 0, 0)):
        return True

    if np.any(data[h - 1, :, :] != (0, 0, 0)):
        return True

    if np.any(data[:, 0, :] != (0, 0, 0)):
        return True

    if np.any(data[:, w - 1, :] != (0, 0, 0)):
        return True

    return False
