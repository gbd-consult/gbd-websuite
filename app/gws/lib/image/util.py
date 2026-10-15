"""Image utilities.

``thumbnail`` scales encoded images down, ``pixel``, ``empty_pixel`` and ``error_pixel`` return
cached encoded 1x1 images. ``qr_code`` creates an image with a QR code.
``get_draw`` and ``get_font`` give access to PIL drawing on an image.
"""

import io
from typing import Optional

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import qrcode.main
import qrcode.constants

import gws
import gws.lib.mime

from . import core


def thumbnail(
    r: bytes,
    size: gws.Size,
    max_pixels=0,
    mime: gws.MimeType = '',
    options=None,
) -> bytes:
    """Create a thumbnail from an encoded image.

    The image is scaled to fit into ``size``, keeping the aspect ratio. Small images are not scaled up.

    Args:
        r: Encoded image.
        size: Maximum thumbnail size ``(width, height)``.
        max_pixels: Maximum number of source pixels, ``0`` for no limit.
        mime: MIME type of the thumbnail, PNG by default.
        options: Encoding options, as in ``Image.to_bytes``.

    Returns:
        The encoded thumbnail.

    Raises:
        ``Error``: If the image is too big or cannot be processed.
    """

    w, h = size
    sz = int(w), int(h)

    try:
        with io.BytesIO(r) as fp:
            img = PIL.Image.open(fp)
            w, h = img.size
            if max_pixels and w * h > max_pixels:
                raise core.Error(f'image too big: {w}x{h}')
            img.draft(img.mode, sz)
            img.thumbnail(sz, resample=PIL.Image.Resampling.BICUBIC)
            if img.mode not in {'1', 'L', 'LA', 'P', 'RGB', 'RGBA'}:
                img = img.convert('RGB')
            return core.Image(img).to_bytes(mime, options)
    except core.Error:
        raise
    except Exception as exc:
        raise core.Error from exc


def qr_code(
    data: str,
    level='M',
    scale=4,
    border=True,
    color='black',
    background='white',
) -> core.Image:
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
    return core._new(img)


def get_draw(img: core.Image) -> PIL.ImageDraw.ImageDraw:
    """Return a PIL drawing object for an image.

    Args:
        img: An image object.

    Returns:
        A ``PIL.ImageDraw.ImageDraw`` that draws on the image in place.
    """

    return PIL.ImageDraw.Draw(img.img)


def get_font(
    size: int = 12,
    font: Optional[str] = None,
) -> PIL.ImageFont.ImageFont | PIL.ImageFont.FreeTypeFont:
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


_PIXELS = {}
_ERROR_COLOR = '#ffa1b4'


def empty_pixel(mime: gws.MimeType = ''):
    """Return an encoded empty 1x1 image.

    The pixel is transparent, or white for JPEG.

    Args:
        mime: MIME type, PNG by default.

    Returns:
        The encoded image.

    Raises:
        ``Error``: If the MIME type is not an image type.
    """

    return pixel(mime, '#ffffff' if mime == gws.lib.mime.JPEG else None)


def error_pixel(mime: gws.MimeType = ''):
    """Return an encoded 1x1 image in the error color.

    Args:
        mime: MIME type, PNG by default.

    Returns:
        The encoded image.

    Raises:
        ``Error``: If the MIME type is not an image type.
    """

    return pixel(mime, _ERROR_COLOR)


def pixel(mime: gws.MimeType, color):
    """Return an encoded 1x1 image of a color.

    Results are cached.

    Args:
        mime: MIME type, PNG by default.
        color: Pixel color, or ``None`` for a transparent pixel.

    Returns:
        The encoded image.

    Raises:
        ``Error``: If the MIME type is not an image type.
    """

    fmt = core.mime_to_pil_format(mime)
    if not fmt:
        raise core.Error(f'unknown mime type {mime!r}')
    key = fmt, str(color)

    if key not in _PIXELS:
        img = PIL.Image.new('RGBA' if color is None else 'RGB', (1, 1), color)
        with io.BytesIO() as fp:
            img.save(fp, fmt)
            _PIXELS[key] = fp.getvalue()

    return _PIXELS[key]
