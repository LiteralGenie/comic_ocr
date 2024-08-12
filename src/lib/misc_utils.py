import math
import random
from random import randint
from typing import TypeAlias

import cv2
from cv2.typing import MatLike

Bbox: TypeAlias = tuple[int, int, int, int]  # top, left, bot, right
Xywh: TypeAlias = tuple[int, int, int, int]


def generate_poly(
    xywh: Xywh,
    max_points: int,
    mu=0.85,
    sigma=0.25,
):
    x, y, w, h = xywh
    center_x = x + w / 2
    center_y = y + h / 2

    max_r = math.sqrt(w**2 + h**2) / 2

    num_points = randint(3, max_points)

    poly: list[tuple[int, int]] = []
    for idx in range(num_points):
        angle = idx * (2 * math.pi / num_points)

        while True:
            r = max_r * rand_gauss(mu, sigma, 0, 1)

            pt = (
                int(center_x + r * math.cos(angle)),
                int(center_y + r * math.sin(angle)),
            )

            if _is_contained(xywh, pt):
                poly.append(pt)
                break

    return poly


def _is_contained(xywh, xy):
    x, y, w, h = xywh
    x2, y2 = xy

    if x2 < x or x2 > x + w:
        return False

    if y2 < y or y2 > y + h:
        return False

    return True


def rand_gauss(mu: float, sigma: float, min: float, max: float):
    while True:
        x = random.gauss(mu, sigma)

        if x >= min and x <= max:
            return x


def rotate(im: MatLike, deg: int):
    center_x = im.shape[1] // 2
    center_y = im.shape[0] // 2

    rot_mat = cv2.getRotationMatrix2D((center_x, center_y), deg, 1.0)

    result = cv2.warpAffine(im, rot_mat, im.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
