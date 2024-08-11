from pathlib import Path
import pickle
import sys
from lib.constants import HANGUL_SYLLABLES
from lib.generate_bubbles import generate_bubbles
from lib.generate_panels import generate_panels
from lib.generate_text import generate_texts
from lib.render_page import RenderContext, build_render_info, render_page

FONT_DIR = Path(sys.argv[1])
IMAGE_DIR = Path(sys.argv[2])

# random.seed(99)


def main():
    # font_map = {fp.name: fp for fp in FONT_DIR.glob("**/*.ttf")}

    # panels, wh = generate_panels()
    # panel_map = {p.id: p for p in panels}

    # bubble_map = {b.id: b for p in panels for b in generate_bubbles(p)}

    # text_map = {
    #     t.id: t
    #     for b in bubble_map.values()
    #     for t in generate_texts(
    #         b,
    #         font_map,
    #         HANGUL_SYLLABLES,
    #     )
    # }

    # ctx = RenderContext(
    #     font_map,
    #     IMAGE_DIR,
    #     wh,
    #     panel_map,
    #     bubble_map,
    #     text_map,
    # )
    # with open(Path("./ctx.pkl"), "wb") as file:
    #     pickle.dump(ctx, file)

    with open(Path("./ctx.pkl"), "rb") as file:
        ctx: RenderContext = pickle.load(file)

    info = build_render_info(ctx)
    im = render_page(ctx, info)
    im.save("tmp.png")


main()
