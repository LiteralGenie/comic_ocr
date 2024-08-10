from dataclasses import dataclass
from pathlib import Path
from random import randint
import random
from uuid import uuid4

import cv2
from cv2.typing import MatLike
from PIL import ImageFont, ImageDraw, Image
import numpy as np

from lib.generate_bubbles import Bubble
from lib.misc_utils import Bbox


@dataclass
class Text:
    id: str
    letter: str
    xy: tuple[int, int]
    fp_font: Path
    font_size: float
    bbox: Bbox


def generate_texts(
    bubble: Bubble,
    fp_fonts: list[Path],
    alphabet: list[str],
    max_tries=2000,
    min_font_size=10,
    max_font_size=40,
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
        letter = random.choice(alphabet)
        fp_font = random.choice(fp_fonts)
        font_size = randint(min_font_size, max_font_size)
        x = randint(0, bubble.width)
        y = randint(0, bubble.height)

        render = Image.new("RGB", (mask.shape[1], mask.shape[0]))
        font = ImageFont.truetype(fp_font, font_size)
        draw = ImageDraw.Draw(render)
        draw.text((x + 1, y + 1), letter, font=font, fill=(255, 255, 255))
        render = np.array(render)

        # check if letter extends outside bubble or intersects with existing letters
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        intersection = cv2.bitwise_and(render, mask, mask=mask_gray)
        is_valid = np.max(intersection) == 0
        if not is_valid:
            continue

        mask = cv2.bitwise_or(render, mask)

        try:
            bbox = _get_bbox(render)
        except:
            # render is empty or has 1-pixel in height / width
            continue

        # print(letter, (x + bubble.bbox[1], y + bubble.bbox[0]), bubble.id, fp_font.stem)
        texts.append(
            Text(
                uuid4().hex,
                letter,
                (x + bubble.bbox[1], y + bubble.bbox[0]),
                fp_font,
                font_size,
                bbox,
            )
        )

    return texts


def _get_bbox(im: MatLike) -> Bbox:
    for idx in range(im.shape[1]):
        x1 = idx

        slice = im[:, x1, :]
        is_empty = np.all(slice == (0, 0, 0))
        if not is_empty:
            break

    for idx in range(x1 + 1, im.shape[1]):
        x2 = idx

        slice = im[:, x2, :]
        is_empty = np.all(slice == (0, 0, 0))
        if is_empty:
            break

    for idx in range(im.shape[1]):
        y1 = idx

        slice = im[:, y1, :]
        is_empty = np.all(slice == (0, 0, 0))
        if not is_empty:
            break

    for idx in range(y1 + 1, im.shape[1]):
        y2 = idx

        slice = im[:, y2, :]
        is_empty = np.all(slice == (0, 0, 0))
        if is_empty:
            break

    return (y1, x1, y2, x2)
