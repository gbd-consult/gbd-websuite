"""Image formats.

``PNG8``, ``PNG24``, ``JPEG`` and ``WEBP`` are predefined ``gws.ImageFormat`` objects. The first
MIME type of a format is its content type, the others are names under which services offer it
(``image/png; mode=8bit``). ``from_config`` creates a format from a ``Config``, and
``best_mime_type`` selects the MIME type to request from a remote source (WMS, QGIS Server) for a format.
"""

from typing import Optional

import gws
import gws.lib.mime

from . import core


class Config(gws.Config):
    """Image format with its encoding options."""

    name: str = ''
    """Name of the format: "png8", "png24", "jpeg", "webp" or a custom name."""
    mimeTypes: Optional[list[gws.MimeType]]
    """MIME types for this format, required for a custom name."""
    options: Optional[dict]
    """Image encoding options."""


PNG8 = gws.ImageFormat(name='png8', mimeTypes=[gws.lib.mime.PNG, gws.lib.mime.PNG8], options={'mode': 'P'})
"""PNG with an 8-bit palette."""
PNG24 = gws.ImageFormat(name='png24', mimeTypes=[gws.lib.mime.PNG, gws.lib.mime.PNG24], options={})
"""PNG with full color and alpha."""
JPEG = gws.ImageFormat(name='jpeg', mimeTypes=[gws.lib.mime.JPEG], options={})
"""JPEG, transparency rendered as a white background."""
WEBP = gws.ImageFormat(name='webp', mimeTypes=[gws.lib.mime.WEBP], options={})
"""WebP."""


_FORMATS = {f.name: f for f in [PNG8, PNG24, JPEG, WEBP]}


def from_config(cfg: Config) -> gws.ImageFormat:
    """Create an image format from its configuration.

    A predefined format (``png8``, ``png24``, ``jpeg``, ``webp``) can be given by name alone;
    configured MIME types and options replace those of the predefined format.

    Args:
        cfg: Format configuration.

    Returns:
        The image format.

    Raises:
        ``Error``: If the configuration has no known MIME types and the name is not a predefined format.
    """

    name = cfg.get('name') or ''

    mimes, options = [], {}
    for f in [PNG8, PNG24, JPEG, WEBP]:
        if f.name == name:
            mimes, options = f.mimeTypes, f.options

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
        raise core.Error(f'unknown image format {name!r}')

    
    p = cfg.get('options')
    if p is not None:
        options = p
    
    return gws.ImageFormat(name=name, mimeTypes=mimes, options=dict(gws.u.to_dict(options)))


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
