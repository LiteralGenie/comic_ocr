from dataclasses import dataclass
from functools import cached_property
from random import randint
from uuid import uuid4

import cv2
import numpy as np

from lib.generate_panels import Panel
from lib.misc_utils import generate_poly, rand_gauss


@dataclass
class Bubble:
    id: str
    id_panel: str
    poly: list[tuple[int, int]]

    @cached_property
    def bbox(self):
        y_min = min(xy[1] for xy in self.poly)
        y_max = max(xy[1] for xy in self.poly)
        x_min = min(xy[0] for xy in self.poly)
        x_max = max(xy[0] for xy in self.poly)

        return (y_min, x_min, y_max, x_max)

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


def generate_bubbles(
    panel: Panel,
    max_tries=6,
    max_points=15,
) -> list[Bubble]:
    bubbles = []

    # generate bubbles
    for idx in range(max_tries):
        w = int(panel.width * rand_gauss(0.75, 0.25, 0, 1)) + 1
        h = int(panel.height * rand_gauss(0.75, 0.25, 0, 1)) + 1

        x = randint(panel.bbox[1], panel.bbox[3] - w)
        y = randint(panel.bbox[0], panel.bbox[2] - h)

        poly = generate_poly((x, y, w, h), max_points)

        b = Bubble(uuid4().hex, panel.id, poly)
        bubbles.append(b)

    panel_mask = panel.mask

    # filter bubbles without panel overlap
    after_panel_filter = []
    for b in bubbles:
        y1, x1, y2, x2 = b.bbox
        x1 -= panel.bbox[1]
        x2 -= panel.bbox[1]
        y1 -= panel.bbox[0]
        y2 -= panel.bbox[0]

        b_mask = np.zeros(panel_mask.shape, np.uint8)
        b_mask[y1:y2, x1:x2, :] = b.mask

        clip_mask = cv2.cvtColor(panel_mask, cv2.COLOR_BGR2GRAY)
        _, clip_mask = cv2.threshold(clip_mask, 127, 255, cv2.THRESH_BINARY)
        clipped = cv2.bitwise_and(b_mask, b_mask, mask=clip_mask)
        if np.max(clipped) == 0:
            continue

        clipped = cv2.cvtColor(clipped, cv2.COLOR_BGR2GRAY)
        ctrs, _ = cv2.findContours(clipped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = ctrs[0]

        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if approx.shape[0] < 3:
            continue

        new_poly = [tuple(x[0]) for x in approx]
        new_poly = [
            (
                xy[0] + panel.bbox[1],
                xy[1] + panel.bbox[0],
            )
            for xy in new_poly
        ]
        after_panel_filter.append(Bubble(b.id, panel.id, new_poly))

    # filter bubbles that intersect
    canvas = np.zeros(panel.mask.shape, np.uint8)
    after_bubble_filter = []
    for idx, b in enumerate(after_panel_filter):
        y1, x1, y2, x2 = b.bbox
        x1 -= panel.bbox[1]
        x2 -= panel.bbox[1]
        y1 -= panel.bbox[0]
        y2 -= panel.bbox[0]

        b_mask = np.zeros(canvas.shape, np.uint8)
        b_mask[y1:y2, x1:x2, :] = b.mask

        intersection = cv2.bitwise_and(canvas, b_mask)
        has_overlap = np.max(intersection) == 255
        if not has_overlap:
            after_bubble_filter.append(idx)
            canvas = cv2.bitwise_or(canvas, b_mask)

    return [after_panel_filter[idx] for idx in after_bubble_filter]
