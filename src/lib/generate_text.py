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

        try:
            font = ImageFont.truetype(font_map[font_file], font_size)
            draw = ImageDraw.Draw(render)
            draw.text((x + 1, y + 1), letter, font=font, fill=(255, 255, 255))
        except OSError:
            print(f"Typeset with {font_map[font_file]} failed")
            continue

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
