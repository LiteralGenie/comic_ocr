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
    id_bubble: str
    letter: str
    xy: tuple[int, int]
    font_file: str
    font_size: float
    bbox: Bbox
    angle: int


def generate_texts(
    bubble: Bubble,
    font_map: dict[str, Path],
    alphabet: list[str],
    max_tries=1000,
    min_font_size=20,
    max_font_size=50,
    min_angle=-30,
    max_angle=30,
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
        font_file = random.choice(list(font_map.keys()))
        font_size = randint(min_font_size, max_font_size)
        x = randint(0, bubble.width)
        y = randint(0, bubble.height)
        angle = randint(min_angle, max_angle)

        render = Image.new("RGB", (mask.shape[1], mask.shape[0]))

        font = ImageFont.truetype(font_map[font_file], font_size)
        draw = ImageDraw.Draw(render)
        draw.text((x + 1, y + 1), letter, font=font, fill=(255, 255, 255))
        if _touches_edge(render):
            continue

        render = render.rotate(angle, fillcolor=(0, 0, 0))
        if _touches_edge(render):
            continue

        render = np.array(render)

        # check if letter extends outside bubble or intersects with existing letters
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        intersection = cv2.bitwise_and(render, mask, mask=mask_gray)
        is_valid = np.max(intersection) == 0
        if not is_valid:
            continue

        _, tmp = cv2.threshold(render, 1, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(tmp, mask)

        try:
            y1, x1, y2, x2 = _get_bbox(render)

            x1 += bubble.bbox[1] - 1
            x2 += bubble.bbox[1] - 1
            y1 += bubble.bbox[0] - 1
            y2 += bubble.bbox[0] - 1

            bbox = (y1, x1, y2, x2)
        except:
            # render is empty or has 1-pixel height / width
            continue

        texts.append(
            Text(
                uuid4().hex,
                bubble.id,
                letter,
                (x + bubble.bbox[1], y + bubble.bbox[0]),
                font_file,
                font_size,
                bbox,
                angle,
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

    for idx in range(im.shape[1] - 1, -1, -1):
        x2 = idx

        slice = im[:, x2, :]
        is_empty = np.all(slice == (0, 0, 0))
        if not is_empty:
            break

    for idx in range(im.shape[0]):
        y1 = idx

        slice = im[y1, :, :]
        is_empty = np.all(slice == (0, 0, 0))
        if not is_empty:
            break

    for idx in range(im.shape[0] - 1, -1, -1):
        y2 = idx

        slice = im[y2, :, :]
        is_empty = np.all(slice == (0, 0, 0))
        if not is_empty:
            break

    assert y1 < y2  # type: ignore
    assert x1 < x2  # type: ignore
    return (y1, x1, y2, x2)


# @todo: _touches_edge() is slow
def _touches_edge(im: Image.Image):
    w, h = im.size

    for x in range(w):
        if im.getpixel((x, 0)) != (0, 0, 0):
            return True

        if im.getpixel((x, h - 1)) != (0, 0, 0):
            return True

    for y in range(h):
        if im.getpixel((0, y)) != (0, 0, 0):
            return True

        if im.getpixel((w - 1, y)) != (0, 0, 0):
            return True

    return False
