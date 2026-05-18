import os
import numpy as np

_OVERLAY_FONT_CACHE = {}

def _get_overlay_font(size: int = 18):
    """Return a PIL ImageFont, falling back to the default bitmap font."""
    if size in _OVERLAY_FONT_CACHE:
        return _OVERLAY_FONT_CACHE[size]
    from PIL import ImageFont
    candidates = (
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "Arial.ttf",
    )
    font = None
    for name in candidates:
        try:
            font = ImageFont.truetype(name, size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    _OVERLAY_FONT_CACHE[size] = font
    return font


def _draw_overlay(frame: np.ndarray, text: str, corner: str = 'top_left') -> np.ndarray:
    """Draw a small white-on-black text box at the requested corner of the frame."""
    if not text or corner == 'none':
        return frame
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = _get_overlay_font(18)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 10
    h, w = frame.shape[:2]
    if corner == 'top_right':
        x = w - tw - 2 * pad
    else:
        x = pad
    y = pad
    draw.rectangle([x - 6, y - 4, x + tw + 6, y + th + 6], fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)