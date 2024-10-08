import itertools
from dataclasses import dataclass
from functools import cached_property
from random import randint
from uuid import uuid4

import cv2
import numpy as np

from .misc_utils import Bbox, Xywh, generate_poly


@dataclass
class Panel:
    id: str
    poly: list[tuple[int, int]]
    bbox: Bbox

    @cached_property
    def width(self):
        return self.bbox[3] - self.bbox[1]

    @cached_property
    def height(self):
        return self.bbox[2] - self.bbox[0]

    @cached_property
    def mask(self):
        canvas = np.zeros((self.height, self.width, 3), np.uint8)

        pts = np.array(
            [
                (
                    x - self.bbox[1],
                    y - self.bbox[0],
                )
                for x, y in self.poly
            ]
        )
        pts = pts.reshape((-1, 1, 2))

        cv2.fillPoly(canvas, [pts], (255, 255, 255))

        return canvas


def generate_panels(
    min_rows=3,
    max_rows=12,
    min_cols=3,
    max_cols=12,
    panel_tries=50,
    scale_factor=120,
    max_poly_points=15,
    max_margin=1.5,
) -> tuple[list[Panel], tuple[int, int]]:
    mt = randint(0, int(max_margin * scale_factor))
    ml = randint(0, int(max_margin * scale_factor))
    mb = randint(0, int(max_margin * scale_factor))
    mr = randint(0, int(max_margin * scale_factor))

    num_rows = randint(min_rows, max_rows)
    num_cols = randint(min_cols, max_cols)

    xywh_list = _generate_panel_rects(
        num_rows,
        num_cols,
        panel_tries,
        scale_factor,
    )

    polys = [
        generate_poly(
            xywh,
            max_poly_points,
            1.1,
            0.25,
        )
        for xywh in xywh_list
    ]

    panels = []
    for idx, (xywh, poly) in enumerate(zip(xywh_list, polys)):
        x, y, w, h = xywh

        x += ml
        y += mt

        poly = [(x + ml, y + mt) for x, y in poly]

        bbox = (y, x, y + h, x + w)
        p = Panel(uuid4().hex, poly, bbox)
        panels.append(p)

    grid_wh = (
        ml + mr + num_cols * scale_factor,
        mt + mb + num_rows * scale_factor,
    )

    return panels, grid_wh


def _generate_panel_rects(
    num_rows: int,
    num_cols: int,
    num_tries: int,
    scale_factor: float,
):
    panels_xywh = []

    iter_cells = lambda: itertools.product(range(num_rows), range(num_cols))

    # random, non-overlapping rects
    for _ in range(num_tries):
        w = randint(1, num_cols)
        h = randint(1, num_rows)

        for x, y in iter_cells():
            if not _has_collision((x, y, w, h), panels_xywh, num_rows, num_cols):
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


def _has_collision(
    candidate: Xywh,
    panels: list[Xywh],
    num_rows: int,
    num_cols: int,
):
    x, y, w, h = candidate

    for x2, y2, w2, h2 in panels:
        x_check = x < x2 + w2 and x + w > x2
        y_check = y < y2 + h2 and y + h > y2
        if x_check and y_check:
            return True

    if x + w > num_cols:
        return True
    if y + h > num_rows:
        return True

    return False
