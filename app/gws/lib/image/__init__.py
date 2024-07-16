"""Wrapper for PIL objects"""

from typing import Optional, Literal, cast

import base64
import io
import re

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import numpy as np

import gws
import gws.lib.mime

# https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open
# max 10k x 10k RGBA
PIL.Image.MAX_IMAGE_PIXELS = 10_000 * 10_000 * 4


class Error(gws.Error):
    pass


def from_size(size: gws.Size, color=None) -> 'Image':
    """Creates a monochrome image object.

    Args:
        size: `(width, height)`
        color: `(red, green, blue, alpha)`

    Returns:
        An image object.
    """
    img = PIL.Image.new('RGBA', _int_size(size), color or (0, 0, 0, 0))
    return _new(img)


def from_bytes(r: bytes) -> 'Image':
    """Creates an image object from bytes.

    Args:
        r: Bytes encoding an image.

    Returns:
        An image object.
    """
    with io.BytesIO(r) as fp:
        return _new(PIL.Image.open(fp))


def from_raw_data(r: bytes, mode: str, size: gws.Size) -> 'Image':
    """Creates an image object in a given mode from raw pixel data in arrays.

    Args:
        r: Bytes encoding an image in arrays of pixels.
        mode: PIL image mode.
        size: `(width, height)`

    Returns:
        An image object.
    """
    return _new(PIL.Image.frombytes(mode, _int_size(size), r))


def from_path(path: str) -> 'Image':
    """Creates an image object from a path.

    Args:
        path: Path to an existing image.

    Returns:
        An image object.
    """
    with open(path, 'rb') as fp:
        return from_bytes(fp.read())


_DATA_URL_RE = r'data:image/(png|gif|jpeg|jpg);base64,'


def from_data_url(url: str) -> Optional['Image']:
    """Creates an image object from a URL.

    Args:
        url: URL encoding an image.

    Returns:
        An image object.
    """
    m = re.match(_DATA_URL_RE, url)
    if not m:
        raise Error(f'invalid data url')
    r = base64.standard_b64decode(url[m.end():])
    return from_bytes(r)


def from_svg(xmlstr: str, size: gws.Size, mime=None) -> 'Image':
    """Not implemented yet. Should create an image object from a URL.

    Args:
        xmlstr: XML String of the image.

        size: `(width, height)`

        mime: Mime type.

    Returns:
        An image object.
    """
    # @TODO rasterize svg
    raise NotImplemented


def _new(img: PIL.Image.Image):
    try:
        img.load()
    except Exception as exc:
        raise Error from exc
    return Image(img)


class Image(gws.Image):
    """Class to convert, save and do basic manipulations on images."""

    def __init__(self, img: PIL.Image.Image):
        self.img: PIL.Image.Image = img

    def mode(self):
        return self.img.mode

    def size(self):
        return self.img.size

    def resize(self, size, **kwargs):
        kwargs.setdefault('resample', PIL.Image.BICUBIC)
        self.img = self.img.resize(_int_size(size), **kwargs)
        return self

    def rotate(self, angle, **kwargs):
        kwargs.setdefault('resample', PIL.Image.BICUBIC)
        self.img = self.img.rotate(angle, **kwargs)
        return self

    def crop(self, box):
        self.img = self.img.crop(box)
        return self

    def paste(self, other, where=None):
        self.img.paste(cast('Image', other).img, where)
        return self

    def compose(self, other, opacity=1):
        oth = cast('Image', other).img.convert('RGBA')

        if oth.size != self.img.size:
            oth = oth.resize(size=self.img.size, resample=PIL.Image.BICUBIC)

        if opacity < 1:
            alpha = oth.getchannel('A').point(lambda x: int(x * opacity))
            oth.putalpha(alpha)

        self.img = PIL.Image.alpha_composite(self.img, oth)
        return self

    def to_bytes(self, mime=None, options=None):
        with io.BytesIO() as fp:
            self._save(fp, mime, options)
            return fp.getvalue()

    def to_path(self, path, mime=None, options=None):
        with open(path, 'wb') as fp:
            self._save(fp, mime, options)
        return path

    def _save(self, fp, mime: str, options: dict):
        fmt = _mime_to_format(mime)
        opts = dict(options or {})
        img = self.img

        if self.img.mode == 'RGBA' and fmt == 'JPEG':
            background = opts.pop('background', '#FFFFFF')
            img = PIL.Image.new('RGBA', self.img.size, background)
            img.alpha_composite(self.img)
            img = img.convert('RGB')

        mode = opts.pop('mode', '')
        if mode and self.img.mode != mode:
            img = img.convert(mode, palette=PIL.Image.ADAPTIVE)

        img.save(fp, fmt, **opts)

    def to_array(self):
        return np.array(self.img)

    def add_text(self, text, x=0, y=0, color=None):
        self.img = self.img.convert('RGBA')
        draw = PIL.ImageDraw.Draw(self.img)
        font = PIL.ImageFont.load_default()
        color = color or (0, 0, 0, 255)
        draw.multiline_text((x, y), text, font=font, fill=color)
        return self

    def add_box(self, color=None):
        self.img = self.img.convert('RGBA')
        draw = PIL.ImageDraw.Draw(self.img)
        color = color or (0, 0, 0, 255)
        x, y = self.img.size
        draw.rectangle((0, 0) + (x - 1, y - 1), outline=color)  # box goes around all edges
        return self

    def compare_to(self, other):
        error = 0
        x, y = self.size()
        for i in range(int(x)):
            for j in range(int(y)):
                a_r, a_g, a_b, a_a = self.img.getpixel((i, j))
                b_r, b_g, b_b, b_a = cast(Image, other).img.getpixel((i, j))
                error += (a_r - b_r) ** 2
                error += (a_g - b_g) ** 2
                error += (a_b - b_b) ** 2
                error += (a_a - b_a) ** 2
        return error / (3 * x * y)


_MIME_TO_FORMAT = {
    gws.lib.mime.PNG: 'PNG',
    gws.lib.mime.JPEG: 'JPEG',
    gws.lib.mime.GIF: 'GIF',
    gws.lib.mime.WEBP: 'WEBP',
}


def _mime_to_format(mime):
    if not mime:
        return 'PNG'
    m = mime.split(';')[0].strip()
    if m in _MIME_TO_FORMAT:
        return _MIME_TO_FORMAT[m]
    m = m.split('/')
    if len(m) == 2 and m[0] == 'image':
        return m[1].upper()
    raise Error(f'unknown mime type {mime!r}')


def _int_size(size: gws.Size):
    w, h = size
    return int(w), int(h)


_PIXELS = {}
_ERROR_COLOR = '#ffa1b4'


def empty_pixel(mime: str = None):
    return pixel(mime, '#ffffff' if mime == gws.lib.mime.JPEG else None)


def error_pixel(mime: str = None):
    return pixel(mime, _ERROR_COLOR)


def pixel(mime, color):
    fmt = _mime_to_format(mime)
    key = fmt, str(color)

    if key not in _PIXELS:
        img = PIL.Image.new('RGBA' if color is None else 'RGB', (1, 1), color)
        with io.BytesIO() as fp:
            img.save(fp, fmt)
            _PIXELS[key] = fp.getvalue()

    return _PIXELS[key]
