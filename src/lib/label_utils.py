from pathlib import Path
from lib.constants import KOREAN_ALPHABET
from lib.generate_bubbles import generate_bubbles
from lib.generate_panels import generate_panels
from lib.generate_text import generate_texts
from lib.render_page import RenderContext


def make_context(font_dir: Path, image_dir: Path):
    font_map = {fp.name: fp for fp in font_dir.glob("**/*.ttf")}

    panels, wh = generate_panels()
    panel_map = {p.id: p for p in panels}

    bubble_map = {b.id: b for p in panels for b in generate_bubbles(p)}

    text_map = {
        t.id: t
        for b in bubble_map.values()
        for t in generate_texts(
            b,
            font_map,
            KOREAN_ALPHABET,
        )
    }

    ctx = RenderContext(
        font_map,
        image_dir,
        wh,
        panel_map,
        bubble_map,
        text_map,
    )

    return ctx
