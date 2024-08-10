import cv2
import numpy as np
from lib.generate_panels import generate_panels


panels, wh = generate_panels()

canvas = np.zeros((wh[1], wh[0], 3), np.uint8)
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
    cv2.polylines(canvas, [pts], True, (255, 0, 0))

cv2.imwrite("./tmp.png", canvas)
