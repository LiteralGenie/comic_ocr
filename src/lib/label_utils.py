import hashlib
from PIL import Image
import numpy as np
from lib.render_page import RenderContext


def export_detection_label(ctx: RenderContext, im: Image.Image):
    img_dimensions = ctx.wh

    arr = np.array(im).astype(np.uint16)
    img_hash = hashlib.sha256(arr.tobytes()).hexdigest()

    polygons = []
    for t in ctx.text_map.values():
        y1, x1, y2, x2 = t.bbox
        polygons.append(
            [
                (x1, y1),
                (x1, y2),
                (x2, y2),
                (x2, y1),
            ]
        )

    return dict(
        img_dimensions=img_dimensions,
        img_hash=img_hash,
        polygons=polygons,
    )
