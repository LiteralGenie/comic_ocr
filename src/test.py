import cv2
from cv2.typing import MatLike
import numpy as np
from lib.generate_bubbles import Bubble, generate_bubbles
from lib.generate_panels import Panel, generate_panels


def main():
    panels, wh = generate_panels()

    canvas = np.zeros((wh[1], wh[0], 3), np.uint8)
    canvas.fill(255)
    draw_panels(canvas, panels)
    print(len(panels))

    bubbles = {p.id: generate_bubbles(p) for p in panels}
    draw_bubbles(canvas, bubbles)

    cv2.imwrite("./tmp.png", canvas)


def draw_panels(canvas: MatLike, panels: list[Panel]):
    for p in panels:
        cv2.rectangle(
            canvas,
            (p.bbox[1], p.bbox[0]),
            (p.bbox[3], p.bbox[2]),
            (0, 255, 255),
            3,
        )

        pts = np.array(p.poly)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], True, (0, 0, 0), 5)


def draw_bubbles(canvas: MatLike, bubbles: dict[int, list[Bubble]]):
    for grp in bubbles.values():
        for b in grp:
            pts = np.array(b.poly)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], True, (0, 255, 0), 3)


main()
