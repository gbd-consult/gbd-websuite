"""Raster images.

Provides ``Image``, the implementation of ``gws.Image`` on top of a PIL image, and
functions to create images from various sources and to encode them.

Images are created with the ``from_*`` functions (an empty image of a given size, encoded
bytes, raw pixel data, a file, a data URL or a numpy array) or with ``qr_code``. The
``Image`` methods resize, crop, rotate, paste and compose images and encode them as bytes,
base64, data URLs or files. The output format is selected by a MIME type; PNG is the default.

Image sizes are limited to ``MAX_PIXELS``. ``thumbnail`` scales encoded images down,
``pixel``, ``empty_pixel`` and ``error_pixel`` return cached encoded 1x1 images.
``get_draw`` and ``get_font`` give access to PIL drawing on an image.

Example::

    img = gws.lib.image.from_size((256, 256))
    img.compose(gws.lib.image.from_path('/data/overlay.png'), opacity=0.5)
    png = img.to_bytes(gws.lib.mime.PNG, {'mode': 'P'})
"""

import base64
import io
import re
from typing import Optional, cast

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import numpy as np
import qrcode.main
import qrcode.constants

import gws
import gws.lib.mime

# https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open
# up to ~4 GB RGBA images
MAX_PIXELS = 1_000_000_000
"""Maximum number of pixels of an image created from a size or raw data."""
PIL.Image.MAX_IMAGE_PIXELS = MAX_PIXELS // 2


class Error(gws.Error):
    """Image error."""

    pass


class FormatConfig(gws.Config):
    """Image format with its encoding options."""

    name: str = ''
    """Name of the format."""
    mimeTypes: list[gws.MimeType]
    """MIME types for this format."""
    options: Optional[dict]
    """Image encoding options."""


def from_size(size: gws.Size, color=None) -> 'Image':
    """Create an RGBA image filled with one color.

    Args:
        size: Image size ``(width, height)``.
        color: Fill color ``(red, green, blue, alpha)``, transparent by default.

    Returns:
        An image object.

    Raises:
        ``Error``: If the image has more than ``MAX_PIXELS`` pixels.
    """
    w, h = _int_size(size)
    if w * h > MAX_PIXELS:
        raise Error(f'image too large: {w}x{h}')
    img = PIL.Image.new('RGBA', (w, h), color or (0, 0, 0, 0))
    return _new(img)


def from_bytes(r: bytes) -> 'Image':
    """Create an image from encoded bytes.

    Args:
        r: Encoded image, in any format PIL can read.

    Returns:
        An image object.

    Raises:
        ``Error``: If the image data cannot be loaded.
    """
    with io.BytesIO(r) as fp:
        return _new(PIL.Image.open(fp))


def from_raw_data(r: bytes, mode: str, size: gws.Size) -> 'Image':
    """Create an image from raw pixel data.

    Args:
        r: Raw pixel data.
        mode: PIL image mode of the data.
        size: Image size ``(width, height)``.

    Returns:
        An image object.

    Raises:
        ``Error``: If the image has more than ``MAX_PIXELS`` pixels.
    """

    w, h = _int_size(size)
    if w * h > MAX_PIXELS:
        raise Error(f'image too large: {w}x{h}')
    return _new(PIL.Image.frombytes(mode, (w, h), r))


def from_path(path: str) -> 'Image':
    """Create an image from a file.

    Args:
        path: Path to an image file.

    Returns:
        An image object.

    Raises:
        ``Error``: If the image data cannot be loaded.
    """
    with open(path, 'rb') as fp:
        return from_bytes(fp.read())


_DATA_URL_RE = r'data:image/(png|gif|jpeg|jpg);base64,'


def from_data_url(url: str) -> Optional['Image']:
    """Create an image from a base64 data URL.

    Only PNG, GIF and JPEG data URLs are accepted.

    Args:
        url: A ``data:image/...;base64,`` URL.

    Returns:
        An image object.

    Raises:
        ``Error``: If the URL is not a supported data URL or the image cannot be loaded.
    """
    m = re.match(_DATA_URL_RE, url)
    if not m:
        raise Error(f'invalid data url')
    r = base64.standard_b64decode(url[m.end() :])
    return from_bytes(r)


def from_array(arr: np.ndarray, mode: str = None) -> 'Image':
    """Create an image from a numpy array.

    Args:
        arr: Pixel array, as returned by ``Image.to_array``.
        mode: Not used.

    Returns:
        An image object.

    Raises:
        ``Error``: If the image data cannot be loaded.
    """
    img = PIL.Image.fromarray(arr)
    return _new(img)


def from_svg(xmlstr: str, size: gws.Size, mime_type=None) -> 'Image':
    """Create an image from an SVG document. Not implemented.

    Args:
        xmlstr: SVG source.
        size: Image size ``(width, height)``.
        mime_type: MIME type.

    Returns:
        An image object.

    Raises:
        ``NotImplementedError``: Always.
    """
    # @TODO rasterize svg
    raise NotImplementedError


def thumbnail(r: bytes, size: gws.Size, max_pixels=0, mime_type=None, options=None) -> bytes:
    """Create a thumbnail from an encoded image.

    The image is scaled to fit into ``size``, keeping the aspect ratio. Small images are not scaled up.

    Args:
        r: Encoded image.
        size: Maximum thumbnail size ``(width, height)``.
        max_pixels: Maximum number of source pixels, ``0`` for no limit.
        mime_type: MIME type of the thumbnail, PNG by default.
        options: Encoding options, as in ``Image.to_bytes``.

    Returns:
        The encoded thumbnail.

    Raises:
        ``Error``: If the image is too big or cannot be processed.
    """

    sz = _int_size(size)

    try:
        with io.BytesIO(r) as fp:
            img = PIL.Image.open(fp)
            w, h = img.size
            if max_pixels and w * h > max_pixels:
                raise Error(f'image too big: {w}x{h}')
            img.draft(img.mode, sz)
            img.thumbnail(sz, resample=PIL.Image.Resampling.BICUBIC)
            if img.mode not in {'1', 'L', 'LA', 'P', 'RGB', 'RGBA'}:
                img = img.convert('RGB')
            return Image(img).to_bytes(mime_type, options)
    except Error:
        raise
    except Exception as exc:
        raise Error from exc


def qr_code(
    data: str,
    level='M',
    scale=4,
    border=True,
    color='black',
    background='white',
) -> 'Image':
    """Create an image with a QR code.

    See https://github.com/lincolnloop/python-qrcode/blob/main/README.rst#advanced-usage.

    Args:
        data: Data to encode.
        level: Error correction level, one of ``L``, ``M``, ``Q``, ``H``.
        scale: Box size in pixels.
        border: If ``True``, include a quiet zone of 4 boxes.
        color: Foreground color.
        background: Background color.

    Returns:
        An image object.

    Raises:
        ``Error``: If the image cannot be created.
    """

    ec_map = {
        'L': qrcode.constants.ERROR_CORRECT_L,
        'M': qrcode.constants.ERROR_CORRECT_M,
        'Q': qrcode.constants.ERROR_CORRECT_Q,
        'H': qrcode.constants.ERROR_CORRECT_H,
    }

    qr = qrcode.main.QRCode(
        version=None,
        error_correction=ec_map[level],
        box_size=scale,
        border=4 if border else 0,
    )

    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color=color, back_color=background)
    return _new(img)


def get_draw(img: 'Image') -> PIL.ImageDraw.ImageDraw:
    """Return a PIL drawing object for an image.

    Args:
        img: An image object.

    Returns:
        A ``PIL.ImageDraw.ImageDraw`` that draws on the image in place.
    """

    return PIL.ImageDraw.Draw(img.img)


def get_font(size: int = 12, font: Optional[str] = None) -> PIL.ImageFont.ImageFont | PIL.ImageFont.FreeTypeFont:
    """Return a PIL font object.

    Args:
        size: Font size, used for TrueType fonts only.
        font: Path to a TrueType font file, or ``None`` for the PIL default font.

    Returns:
        A PIL font object.
    """

    if font:
        return PIL.ImageFont.truetype(font, size)
    return PIL.ImageFont.load_default()


def _new(img: PIL.Image.Image):
    """Load the PIL image data and wrap the image in an ``Image``."""
    try:
        img.load()
    except Exception as exc:
        raise Error from exc
    return Image(img)


class Image(gws.Image):
    """Image object, a wrapper around a PIL image."""

    def __init__(self, img: PIL.Image.Image):
        """Create an image object.

        Args:
            img: A PIL image.
        """

        self.img: PIL.Image.Image = img

    def mode(self):
        return self.img.mode

    def size(self):
        return self.img.size

    def resize(self, size, **kwargs):
        kwargs.setdefault('resample', PIL.Image.Resampling.BICUBIC)
        self.img = self.img.resize(_int_size(size), **kwargs)
        return self

    def resize_to(self, width=0, height=0, **kwargs):
        w, h = self.img.size
        if width and height:
            sz = (width, height)
        elif width:
            sz = (width, int(h * width / w))
        elif height:
            sz = (int(w * height / h), height)
        else:
            return self
        return self.resize(sz, **kwargs)

    def rotate(self, angle, **kwargs):
        kwargs.setdefault('resample', PIL.Image.Resampling.BICUBIC)
        self.img = self.img.rotate(angle, **kwargs)
        return self

    def crop(self, box):
        self.img = self.img.crop(box)
        return self

    def convert(self, mode):
        if self.img.mode != mode:
            self.img = self.img.convert(mode)
        return self

    def paste(self, other, where=None):
        self.img.paste(cast('Image', other).img, where)
        return self

    def compose(self, other, opacity=1):
        oth = cast('Image', other).img.convert('RGBA')

        if oth.size != self.img.size:
            oth = oth.resize(size=self.img.size, resample=PIL.Image.Resampling.BICUBIC)

        if opacity < 1:
            alpha = oth.getchannel('A').point(lambda x: int(x * opacity))
            oth.putalpha(alpha)

        self.img = PIL.Image.alpha_composite(self.img, oth)
        return self

    def to_bytes(self, mime_type=None, options=None):
        with io.BytesIO() as fp:
            self._save(fp, mime_type, options)
            return fp.getvalue()

    def to_base64(self, mime_type=None, options=None):
        b = base64.standard_b64encode(self.to_bytes(mime_type, options))
        return b.decode('ascii')

    def to_data_url(self, mime_type=None, options=None):
        mime_type = mime_type or gws.lib.mime.PNG
        return f'data:{mime_type};base64,' + self.to_base64(mime_type, options)

    def to_path(self, path, mime_type=None, options=None):
        with open(path, 'wb') as fp:
            self._save(fp, mime_type, options)
        return path

    def _save(self, fp, mime_type: str, options: dict):
        fmt = _mime_to_format(mime_type)
        opts = dict(options or {})
        img = self.img

        if self.img.mode == 'RGBA' and fmt == 'JPEG':
            background = opts.pop('background', '#FFFFFF')
            img = PIL.Image.new('RGBA', self.img.size, background)
            img.alpha_composite(self.img)
            img = img.convert('RGB')

        mode = opts.pop('mode', '')
        if mode and self.img.mode != mode:
            img = img.convert(mode, palette=PIL.Image.Palette.ADAPTIVE)

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
        draw.rectangle((0, 0) + (x - 1, y - 1), outline=color)
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
        return error / (4 * x * y * 255 * 255)


_MIME_TO_FORMAT = {
    gws.lib.mime.PNG: 'PNG',
    gws.lib.mime.JPEG: 'JPEG',
    gws.lib.mime.GIF: 'GIF',
    gws.lib.mime.WEBP: 'WEBP',
}


def _mime_to_format(mime_type):
    """Return the PIL format name for a MIME type, PNG if no MIME type is given."""

    if not mime_type:
        return 'PNG'
    m = mime_type.split(';')[0].strip()
    if m in _MIME_TO_FORMAT:
        return _MIME_TO_FORMAT[m]
    m = m.split('/')
    if len(m) == 2 and m[0] == 'image':
        return m[1].upper()
    raise Error(f'unknown mime type {mime_type!r}')


def _int_size(size: gws.Size):
    w, h = size
    return int(w), int(h)


_PIXELS = {}
_ERROR_COLOR = '#ffa1b4'


def empty_pixel(mime_type: str = None):
    """Return an encoded empty 1x1 image.

    The pixel is transparent, or white for JPEG.

    Args:
        mime_type: MIME type, PNG by default.

    Returns:
        The encoded image.

    Raises:
        ``Error``: If the MIME type is not an image type.
    """

    return pixel(mime_type, '#ffffff' if mime_type == gws.lib.mime.JPEG else None)


def error_pixel(mime_type: str = None):
    """Return an encoded 1x1 image in the error color.

    Args:
        mime_type: MIME type, PNG by default.

    Returns:
        The encoded image.

    Raises:
        ``Error``: If the MIME type is not an image type.
    """

    return pixel(mime_type, _ERROR_COLOR)


def pixel(mime_type, color):
    """Return an encoded 1x1 image of a color.

    Results are cached.

    Args:
        mime_type: MIME type, PNG by default.
        color: Pixel color, or ``None`` for a transparent pixel.

    Returns:
        The encoded image.

    Raises:
        ``Error``: If the MIME type is not an image type.
    """

    fmt = _mime_to_format(mime_type)
    key = fmt, str(color)

    if key not in _PIXELS:
        img = PIL.Image.new('RGBA' if color is None else 'RGB', (1, 1), color)
        with io.BytesIO() as fp:
            img.save(fp, fmt)
            _PIXELS[key] = fp.getvalue()

    return _PIXELS[key]
