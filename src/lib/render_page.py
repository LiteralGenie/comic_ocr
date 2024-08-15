from dataclasses import dataclass, fields, is_dataclass
from functools import cached_property
from pathlib import Path
import random
from typing import TypeAlias

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

from lib.generate_bubbles import Bubble
from lib.generate_panels import Panel
from lib.generate_text import Text
from lib.misc_utils import Bbox

Rgba: TypeAlias = tuple[int, int, int, int]

# @todo: stroke styles


@dataclass
class RenderContext:
    font_map: dict[str, Path]
    image_dir: Path

    wh: tuple[int, int]
    panel_map: dict[str, Panel]
    bubble_map: dict[str, Bubble]
    text_map: dict[str, Text]

    max_image_noise: float = 0.75
    max_panel_stroke_width: int = 5
    max_bubble_stroke_width: int = 10
    max_image_rotate: int = 45
    min_text_opacity: int = 180
    min_bubble_opacity: int = 127
    min_image_crop_frac: float = 0.2
    p_image_flip_x: float = 0.5
    p_image_flip_y: float = 0.5
    p_image_inv: float = 0.5
    p_page_image: float = 0.8
    p_page_fill: float = 0.1
    p_panel_image: float = 0.6
    p_panel_fill: float = 0.2

    @cached_property
    def fp_images(self):
        return list(self.image_dir.glob("*.png"))

    def dump(self):
        return dump_dataclass(self)


@dataclass
class ImageRenderInfo:
    filename: str
    noise: float
    crop: Bbox | None
    flip_x: bool
    flip_y: bool
    rotate_deg: float
    inv_r: bool
    inv_g: bool
    inv_b: bool
    tint: Rgba


@dataclass
class PageRenderInfo:
    wh: tuple[int, int]
    image: ImageRenderInfo | None
    fill: Rgba | None

    panels: dict[str, "PanelRenderInfo"]

    def dump(self):
        return dump_dataclass(self)


@dataclass
class PanelRenderInfo:
    stroke_color: Rgba
    stroke_width: int
    image: ImageRenderInfo | None
    fill: Rgba | None

    bubbles: dict[str, "BubbleRenderInfo"]


@dataclass
class BubbleRenderInfo:
    stroke_color: Rgba
    stroke_width: int
    fill: Rgba
    text_color: Rgba


def build_render_info(ctx: RenderContext) -> PageRenderInfo:
    info = _pick_page_render(ctx)

    for p in ctx.panel_map.values():
        info.panels[p.id] = _pick_panel_render(ctx)

    for b in ctx.bubble_map.values():
        panel_info = info.panels[b.id_panel]
        panel_info.bubbles[b.id] = _pick_bubble_render(ctx)

    return info


def render_page(ctx: RenderContext, info: PageRenderInfo) -> Image.Image:
    canvas = Image.new("RGBA", ctx.wh)

    if info.image:
        fill_layer = _render_image(ctx, info.image)
        fill_layer = fill_layer.resize(canvas.size, Image.Resampling.BICUBIC)
        canvas = Image.alpha_composite(canvas, fill_layer)
    elif info.fill:
        fill_layer = Image.new("RGBA", canvas.size, info.fill)
        canvas = Image.alpha_composite(canvas, fill_layer)

    for pid, panel in info.panels.items():
        canvas = _render_panel(ctx, ctx.panel_map[pid], panel, canvas)

        for bid, bubble in panel.bubbles.items():
            canvas = _render_bubble(ctx, ctx.bubble_map[bid], bubble, canvas)

            for text in ctx.text_map.values():
                if text.id_bubble != bid:
                    continue

                canvas = _render_text(ctx, bubble, text, canvas)

    return canvas


def _pick_page_render(ctx: RenderContext) -> PageRenderInfo:
    image, fill = _pick_image_or_fill(ctx, ctx.p_page_image, ctx.p_page_fill)

    return PageRenderInfo(
        wh=ctx.wh,
        image=image,
        fill=fill,
        panels=dict(),
    )


def _pick_panel_render(ctx: RenderContext) -> PanelRenderInfo:
    stroke_color = _pick_color()
    stroke_width = random.randint(0, ctx.max_panel_stroke_width)
    image, fill = _pick_image_or_fill(ctx, ctx.p_panel_image, ctx.p_panel_fill)

    return PanelRenderInfo(
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        image=image,
        fill=fill,
        bubbles=dict(),
    )


def _pick_bubble_render(ctx: RenderContext) -> BubbleRenderInfo:
    stroke_color = _pick_color()
    stroke_width = random.randint(0, ctx.max_panel_stroke_width)

    contrast = 0
    while contrast < 3:
        fill = _pick_color()
        fill = fill[:3] + (random.randint(ctx.min_bubble_opacity, 255),)

        text_color = _pick_color()
        text_color = text_color[:3] + (random.randint(ctx.min_text_opacity, 255),)

        contrast = _guess_contrast(fill, text_color)

    return BubbleRenderInfo(
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        fill=fill,
        text_color=text_color,
    )


def _pick_image(ctx: RenderContext) -> ImageRenderInfo:
    fp = random.choice(ctx.fp_images)
    filename = fp.name

    noise = random.random() * ctx.max_image_noise

    im = Image.open(fp)

    x1, x2 = _pick_subrange(0, im.size[0], ctx.min_image_crop_frac)
    y1, y2 = _pick_subrange(0, im.size[1], ctx.min_image_crop_frac)
    crop = (y1, x1, y2, x2)

    flip_x = random.random() < ctx.p_image_flip_x
    flip_y = random.random() < ctx.p_image_flip_y

    rotate_deg = random.random() * ctx.max_image_rotate

    inv_r = random.random() < ctx.p_image_inv
    inv_g = random.random() < ctx.p_image_inv
    inv_b = random.random() < ctx.p_image_inv

    tint = _pick_color()

    return ImageRenderInfo(
        filename=filename,
        noise=noise,
        crop=crop,
        flip_x=flip_x,
        flip_y=flip_y,
        rotate_deg=rotate_deg,
        inv_r=inv_r,
        inv_g=inv_g,
        inv_b=inv_b,
        tint=tint,
    )


def _pick_color():
    return (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
    )


def _pick_subrange(mn: int, mx: int, min_dist_frac: float):
    while True:
        x1 = random.randint(mn, mx)
        x2 = random.randint(mn, mx)

        if x1 == x2:
            continue

        if x1 > x2:
            x1, x2 = x2, x1

        dist_frac = (x2 - x1) / (mx - mn)
        if dist_frac < min_dist_frac:
            continue

        return (x1, x2)


def _pick_image_or_fill(
    ctx: RenderContext,
    p_image: float,
    p_fill: float,
):
    bg_weights = [
        int(100 * p)
        for p in [
            p_image,
            p_fill,
            1 - (p_image + p_fill),
        ]
    ]
    bg_type = random.sample(
        ["image", "fill", "none"],
        k=1,
        counts=bg_weights,
    )[0]
    match bg_type:
        case "image":
            image = _pick_image(ctx)
            fill = None
        case "fill":
            image = None
            fill = _pick_color()
        case _:
            image = None
            fill = None

    return image, fill


def _render_image(
    ctx: RenderContext,
    info: ImageRenderInfo,
) -> Image.Image:
    im = Image.open(ctx.image_dir / info.filename).convert("RGBA")

    if info.noise:
        rgb = 255 * np.random.rand(im.size[1], im.size[0], 3)
        alpha = info.noise * 255 * np.random.rand(im.size[1], im.size[0], 1)
        noise = np.concatenate((rgb, alpha), axis=2)
        noise = Image.fromarray(noise.astype("uint8"))

        im = Image.alpha_composite(im, noise)

    if info.tint:
        layer = Image.new("RGBA", im.size, info.tint)
        im = Image.alpha_composite(im, layer)

    if info.flip_x:
        im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if info.flip_y:
        im = im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    if info.rotate_deg:
        im = im.rotate(
            info.rotate_deg,
            resample=Image.Resampling.BICUBIC,
        )

    if info.inv_r or info.inv_g or info.inv_b:
        data = np.array(im)

        if info.inv_r:
            data[:, :, 0] = 255 - data[:, :, 0]
        if info.inv_g:
            data[:, :, 1] = 255 - data[:, :, 1]
        if info.inv_b:
            data[:, :, 2] = 255 - data[:, :, 2]

        im = Image.fromarray(data)

    if info.crop:
        y1, x1, y2, x2 = info.crop
        im = im.crop((x1, y1, x2, y2))

    return im


def _render_panel(
    ctx: RenderContext,
    panel: Panel,
    info: PanelRenderInfo,
    canvas: Image.Image,
) -> Image.Image:
    pts = np.array(panel.poly)
    pts = pts.reshape((-1, 1, 2))

    fill_mask = np.array(Image.new("RGBA", canvas.size, (0, 0, 0, 0)))
    fill_mask = cv2.fillPoly(
        fill_mask,
        [pts],
        (255, 255, 255, 255),
    )
    fill_mask_im = Image.fromarray(fill_mask)

    if info.image:
        fill_layer = _render_image(ctx, info.image)
        fill_layer = fill_layer.resize(
            (panel.width, panel.height), Image.Resampling.BICUBIC
        )

        tmp = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        tmp.paste(fill_layer, (panel.bbox[1], panel.bbox[0]))

        tmp2 = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        tmp2.paste(tmp, (0, 0), fill_mask_im)

        canvas = Image.alpha_composite(canvas, tmp2)
    elif info.fill:
        fill_layer = Image.new("RGBA", canvas.size, info.fill)

        tmp = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        tmp.paste(fill_layer, (panel.bbox[1], panel.bbox[0]))

        tmp2 = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        tmp2.paste(tmp, (0, 0), fill_mask_im)

        canvas = Image.alpha_composite(canvas, tmp2)

    if info.stroke_width > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (2 * info.stroke_width + 1, 2 * info.stroke_width + 1),
            (info.stroke_width, info.stroke_width),
        )
        stroke_mask = cv2.dilate(fill_mask, kernel)
        stroke_mask = np.subtract(stroke_mask, fill_mask)
        stroke_mask_im = Image.fromarray(stroke_mask)

        fill_layer = Image.new("RGBA", canvas.size, info.fill)

        tmp = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        tmp.paste(fill_layer, (0, 0), stroke_mask_im)

        canvas = Image.alpha_composite(canvas, tmp)

    return canvas


def _render_bubble(
    ctx: RenderContext,
    bubble: Bubble,
    info: BubbleRenderInfo,
    canvas: Image.Image,
) -> Image.Image:
    fill_layer = np.array(Image.new("RGBA", canvas.size, (0, 0, 0, 0)))

    pts = np.array(bubble.poly)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(
        fill_layer,
        [pts],
        info.fill[::-1],
    )
    cv2.polylines(
        fill_layer,
        [pts],
        True,
        info.stroke_color[::-1],
        info.stroke_width,
    )

    fill_layer_im = Image.fromarray(fill_layer)

    canvas = Image.alpha_composite(canvas, fill_layer_im)

    return canvas


def _render_text(
    ctx: RenderContext,
    bubble_info: BubbleRenderInfo,
    text: Text,
    canvas: Image.Image,
) -> Image.Image:
    bubble = ctx.bubble_map[text.id_bubble]
    bubble_center = (bubble.width // 2, bubble.height // 2)

    render = Image.new("RGBA", (bubble.width, bubble.height), (0, 0, 0, 0))
    font = ImageFont.truetype(
        ctx.font_map[bubble.font_file],
        text.font_size,
    )

    x, y = text.xy
    x -= bubble.bbox[1]
    y -= bubble.bbox[0]

    draw = ImageDraw.Draw(render)
    draw.text(
        (x, y),
        text.letter,
        font=font,
        fill=bubble_info.text_color,
    )

    render = render.rotate(
        text.angle,
        resample=Image.Resampling.BICUBIC,
        center=bubble_center,
    )

    canvas.paste(render, (bubble.bbox[1], bubble.bbox[0]), render)

    return canvas


def dump_dataclass(instance, exclude: list[str] | None = None) -> dict:
    exclude = exclude or []

    data = dict()

    for f in fields(instance):
        if f.name in exclude:
            continue

        val = getattr(instance, f.name)

        if is_dataclass(val):
            data[f.name] = dump_dataclass(val, exclude)
        else:
            data[f.name] = _dump(val, exclude)

    return data


def _dump(x, exclude: list[str] | None):
    exclude = exclude or []

    if isinstance(x, (str, int, float, bool)):
        return x
    elif isinstance(x, (np.int64, np.int32, np.int16, np.int8)):  # type: ignore
        return int(x)
    elif isinstance(x, Path):
        return str(x)
    elif isinstance(x, list):
        return [_dump(v, exclude) for v in x]
    elif isinstance(x, tuple):
        return tuple(_dump(v, exclude) for v in x)
    elif isinstance(x, dict):
        return {k: _dump(v, exclude) for k, v in x.items() if k not in exclude}
    elif is_dataclass(x):
        return dump_dataclass(x, exclude)
    elif x is None:
        return None
    else:
        raise Exception()


def _guess_contrast(a: Rgba, b: Rgba) -> float:
    return a[3] * b[3] * _calc_constrast(a[:3], b[:3])


def _calc_constrast(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    la = _calc_luminance(a)
    lb = _calc_luminance(b)

    brightest = max(la, lb)
    darkest = min(la, lb)

    return (brightest + 0.05) / (darkest + 0.05)


def _calc_luminance(rgb: tuple[int, int, int]) -> float:
    def f(x):
        x /= 255
        if x < 0.03928:
            return x / 12.92
        else:
            return ((x + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
