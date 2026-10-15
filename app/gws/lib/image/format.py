"""Image formats.

``PNG8``, ``PNG24``, ``JPEG_LOW``, ``JPEG``, ``JPEG_HIGH``, ``WEBP`` and ``WEBP_LOSSLESS`` are predefined ``gws.ImageFormat`` objects. The first
MIME type of a format is its content type, the others are names under which services offer it
(``image/png; mode=8bit``). ``from_config`` creates a format from a ``Config``, and ``best_mime_type``
selects the MIME type to request from a remote source (WMS, QGIS Server) for a format.

Encoding options
----------------

The options of a format are passed to ``gws.Image.to_bytes``. ``mode``, ``background`` and ``quality``
are also ``Config`` fields. ``mode`` and ``background`` are handled by ``gws.lib.image.core.Image``,
all other keys are passed to the PIL encoder (Pillow 12). Configured options replace those of a
predefined format, except its ``mode``, ``background`` and ``quality``.

==================== ================================================================= ================
option               effect                                                            formats
==================== ================================================================= ================
``mode``             convert to this PIL mode before encoding, with an adaptive        all
                     palette: ``P`` (8-bit palette), ``RGBA``, ``RGB``, ``L``
                     (grayscale); JPEG accepts only ``RGB``, ``L``, ``CMYK``
``background``       color of transparent areas when RGBA is converted to RGB,         JPEG
                     default white
``quality``          JPEG 1-95 (default 75), WebP 0-100 (default 80); for lossless     JPEG, WebP, TIFF
                     WebP the encoder effort; TIFF with ``compression: jpeg`` only
``compress_level``   zlib level 0-9, default 6; predefined PNG formats use 3           PNG
``optimize``         optimized Huffman tables                                          JPEG
``progressive``      progressive JPEG                                                  JPEG
``subsampling``      chroma subsampling: 0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0               JPEG
``lossless``         lossless WebP                                                     WebP
``method``           encoder effort 0-6, default 4 (higher = smaller, slower)          WebP
``alpha_quality``    alpha channel quality 0-100, default 100                          WebP
``compression``      ``raw``, ``tiff_lzw``, ``tiff_adobe_deflate``, ``jpeg``,          TIFF
                     ``packbits``
==================== ================================================================= ================

Not to be used: PNG ``optimize`` and ``compress_level: 9`` (6-10x slower than level 6, at most 2%
smaller), PNG ``bits`` and ``transparency`` (break the palette created by ``mode: P``), WebP ``exact``,
JPEG ``quality`` above 95, ``smooth``, ``qtables``, ``restart_marker_*``, ``dpi`` (header only, no
effect on maps), metadata and animation options, ``mode: P`` with WebP (converted back to RGBA).

Measured on map tiles and a 1024 px aerial image: ``compress_level: 3`` encodes PNG24 2-3x and PNG8 3x
faster than the default 6, at most 8% larger. Lossy WebP is 15-50% smaller than PNG8 and 6x slower to
encode; lossless WebP is 2-4x smaller than PNG24, but larger and much slower than PNG8. JPEG on a
1024 px map: quality 50 is 35% smaller than 75, quality 90 is 70% larger; ``subsampling: 0`` adds
another 40% at quality 90.
"""

from typing import Optional

import gws
import gws.lib.mime

from . import core


class ImageMode(gws.Enum):
    """Image mode the image is converted to before encoding."""

    P = 'P'
    """8-bit palette."""
    RGBA = 'RGBA'
    """Full color with transparency."""
    RGB = 'RGB'
    """Full color without transparency."""
    L = 'L'
    """Grayscale."""


class ImageFormatName(gws.Enum):
    """Predefined image format."""

    PNG8 = 'PNG8'
    """PNG with an 8-bit palette."""
    PNG24 = 'PNG24'
    """PNG with full color and alpha."""
    JPEG_LOW = 'JPEG_LOW'
    """JPEG with quality 50."""
    JPEG = 'JPEG'
    """JPEG with quality 75."""
    JPEG_HIGH = 'JPEG_HIGH'
    """JPEG with quality 90."""
    WEBP = 'WEBP'
    """Lossy WebP."""
    WEBP_LOSSLESS = 'WEBP_LOSSLESS'
    """Lossless WebP."""


class Config(gws.Config):
    """Image format with its encoding options."""

    name: ImageFormatName
    """Predefined format."""
    mimeTypes: Optional[list[gws.MimeType]]
    """MIME types for this format, replace those of the predefined format."""
    mode: Optional[ImageMode]
    """Image mode."""
    background: Optional[gws.Color]
    """Color of transparent areas in formats without transparency (JPEG)."""
    quality: Optional[int]
    """Encoding quality: JPEG 1-95, WebP 0-100."""
    options: Optional[dict]
    """Further image encoding options."""


PNG8 = gws.ImageFormat(
    name=ImageFormatName.PNG8,
    mimeTypes=[gws.lib.mime.PNG, gws.lib.mime.PNG8],
    options={'mode': 'P', 'compress_level': 3},
)
PNG24 = gws.ImageFormat(
    name=ImageFormatName.PNG24,
    mimeTypes=[gws.lib.mime.PNG, gws.lib.mime.PNG24],
    options={'compress_level': 3},
)
JPEG_LOW = gws.ImageFormat(
    name=ImageFormatName.JPEG_LOW,
    mimeTypes=[gws.lib.mime.JPEG],
    options={'quality': 50},
)
JPEG = gws.ImageFormat(
    name=ImageFormatName.JPEG,
    mimeTypes=[gws.lib.mime.JPEG],
    options={'quality': 75},
)
JPEG_HIGH = gws.ImageFormat(
    name=ImageFormatName.JPEG_HIGH,
    mimeTypes=[gws.lib.mime.JPEG],
    options={'quality': 90},
)
WEBP = gws.ImageFormat(
    name=ImageFormatName.WEBP,
    mimeTypes=[gws.lib.mime.WEBP],
    options={},
)
WEBP_LOSSLESS = gws.ImageFormat(
    name=ImageFormatName.WEBP_LOSSLESS,
    mimeTypes=[gws.lib.mime.WEBP],
    options={'lossless': True},
)


_FORMATS = {f.name: f for f in [PNG8, PNG24, JPEG_LOW, JPEG, JPEG_HIGH, WEBP, WEBP_LOSSLESS]}
_CONFIG_KEYS = ['mode', 'background', 'quality']


def from_config(cfg: Config) -> gws.ImageFormat:
    """Create an image format from its configuration.

    The format is based on the predefined format ``name``. Configured MIME types replace those
    of the predefined format, and so do configured ``options``, except that the predefined ``mode``,
    ``background`` and ``quality`` are kept unless the configured ``options`` set them. The
    configured ``mode``, ``background`` and ``quality`` fields are applied last.

    Args:
        cfg: Format configuration.

    Returns:
        The image format.

    Raises:
        ``Error``: If the name is not a predefined format or the configuration has no known MIME types.
    """

    name = cfg.get('name') or ''
    base = _FORMATS.get(name)
    if not base:
        raise core.Error(f'unknown image format {name!r}')

    mimes = base.mimeTypes
    p = cfg.get('mimeTypes')
    if p is not None:
        mimes = []
        for s in p:
            mime = gws.lib.mime.get(s)
            if not mime or not core.mime_to_pil_format(mime):
                gws.log.warning(f'image format {name!r}: unknown MIME type {s!r} ignored')
                continue
            mimes.append(mime)

    if not mimes:
        raise core.Error(f'image format {name!r}: no known MIME types')

    p = cfg.get('options')
    options = dict(gws.u.to_dict(p) if p is not None else base.options)

    for key in _CONFIG_KEYS:
        p = cfg.get(key)
        if p is not None:
            options[key] = p
        elif key not in options and key in base.options:
            options[key] = base.options[key]

    return gws.ImageFormat(name=name, mimeTypes=mimes, options=options)


def find_by_mime_type(mime: gws.MimeType, formats: list[gws.ImageFormat]) -> Optional[gws.ImageFormat]:
    """Find an image format by one of its MIME types.

    Args:
        mime: MIME type, normalized with ``gws.lib.mime.get``.
        formats: Image formats.

    Returns:
        The first format that has the MIME type, or ``None`` if there is none.
    """

    for ifo in formats:
        if mime in ifo.mimeTypes:
            return ifo


def best_mime_type(ifo: gws.ImageFormat, supported: list[str]) -> str:
    """Select the MIME type to request from a source for an image format.

    The supported MIME types are tried longest first, so that a specific one (``image/png; mode=8bit``)
    wins over a plain one (``image/png``). The first one that is among the MIME types of the format
    is selected.

    Args:
        ifo: Image format.
        supported: MIME types the source supports, normalized with ``gws.lib.mime.get``.

    Returns:
        The selected MIME type, or an empty string if none matches.
    """

    for mime in sorted(supported, key=len, reverse=True):
        if mime in ifo.mimeTypes:
            return mime

    return ''
