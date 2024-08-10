from dataclasses import dataclass
import functools
import itertools
import math
from random import randint
import random
from typing import Literal, TypeAlias

Bbox: TypeAlias = tuple[int, int, int, int]  # top, left, bot, right
Xywh: TypeAlias = tuple[int, int, int, int]


@dataclass
class Panel:
    poly: list[tuple[int, int]]
    bbox: Bbox


def generate_panels(
    num_rows=8,
    num_cols=6,
    max_panels=10,
    scale_factor=100,
    max_poly_points=10,
) -> tuple[list[Panel], tuple[int, int]]:
    xywh_list = _generate_panel_rects(
        num_rows,
        num_cols,
        max_panels,
        scale_factor,
    )

    polys = [
        _generate_poly(
            xywh,
            max_poly_points,
        )
        for xywh in xywh_list
    ]

    panels = []
    for xywh, poly in zip(xywh_list, polys):
        x, y, w, h = xywh
        bbox = (y, x, y + h, x + w)
        p = Panel(poly=poly, bbox=bbox)
        panels.append(p)

    grid_wh = (
        num_rows * scale_factor,
        num_cols * scale_factor,
    )

    return panels, grid_wh


def _generate_panel_rects(
    num_rows: int,
    num_cols: int,
    num_tries: int,
    scale_factor: float,
):
    panels_xywh = []

    iter_cells = itertools.product(range(num_rows), range(num_cols))

    # random, non-overlapping rects
    for _ in range(num_tries):
        w = randint(1, num_cols)
        h = randint(1, num_rows)

        for x, y in iter_cells:
            if not _has_collision((x, y, w, h), panels_xywh):
                panels_xywh.append((x, y, w, h))
                break

    s = scale_factor
    after_scaling = [
        (
            x * s,
            y * s,
            w * s,
            h * s,
        )
        for x, y, w, h in panels_xywh
    ]

    return after_scaling


def _has_collision(candidate: Xywh, panels: list[Xywh]):
    x, y, w, h = candidate

    for x2, y2, w2, h2 in panels:
        x_check = x < x2 + w2 and x + w > x2
        y_check = y < y2 + h2 and y + h > y2
        if x_check and y_check:
            return True

    return False


def _generate_poly(
    xywh: Xywh,
    max_points: int,
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
            while True:
                r = max_r * random.gauss(0.85, 0.25)
                if r < 0:
                    continue
                break

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
