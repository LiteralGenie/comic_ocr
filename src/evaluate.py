from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont
from doctr.io import Document, DocumentFile
import numpy as np
import torch
from doctr.models import ocr_predictor, db_resnet50, parseq

from lib.constants import KOREAN_ALPHABET

TEST_DIR = Path(sys.argv[1])
DET_WEIGHTS = Path(sys.argv[2])
RECO_WEIGHTS = Path(sys.argv[3])
FONT_FILE = Path(sys.argv[4])
FONT_SIZE = 20
LABEL_OFFSET_Y = -20

det_model = db_resnet50(pretrained=False, pretrained_backbone=False)
det_params = torch.load(DET_WEIGHTS, map_location="cpu")
det_model.load_state_dict(det_params)

reco_model = parseq(vocab=KOREAN_ALPHABET, pretrained=False, pretrained_backbone=False)
reco_params = torch.load(RECO_WEIGHTS, map_location="cpu")
reco_model.load_state_dict(reco_params)

predictor = ocr_predictor(
    det_arch=det_model,
    reco_arch=reco_model,
    pretrained=False,
)

fp_tests = list(TEST_DIR.glob("**/*.png")) + list(TEST_DIR.glob("**/*.jpg"))

font = ImageFont.truetype(FONT_FILE, FONT_SIZE)

for fp in fp_tests:
    im = Image.open(fp).convert("RGBA")

    crop_x = min(im.size[0], 1024)
    crop_y = min(im.size[1], 1024)
    crop_data = np.asarray(im.convert("RGB").crop((0, 0, crop_x, crop_y)))

    output: Document = predictor([crop_data])

    lines: list[str] = []
    for page in output.pages:
        for block in page.blocks:
            for ln in block.lines:
                lines.append(ln.render())

                for w in ln.words:
                    canvas = Image.new("RGBA", im.size)
                    draw = ImageDraw.Draw(canvas)

                    ((x1, y1), (x2, y2)) = w.geometry
                    x1 *= crop_x  # im.size[0]
                    x2 *= crop_x  # im.size[0]
                    y1 *= crop_y  # im.size[1]
                    y2 *= crop_y  # im.size[1]

                    a = int(w.confidence * 255)

                    width = round(w.confidence * 5)

                    draw.rectangle(
                        (x1, y1, x2, y2),
                        outline=(255, 0, 0, a),
                        width=width,
                    )

                    draw.text(
                        (x1, y1 + LABEL_OFFSET_Y),
                        w.value,
                        font=font,
                        fill=(0, 255, 0, a),
                    )

                    im.paste(canvas, (0, 0), canvas)

    result = "\n".join(lines)

    im.save(f"eval_{fp.stem}.png")

    print(fp.stem)
    print(result)
    print("\n---------\n")
