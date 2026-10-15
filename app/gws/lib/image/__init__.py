"""Raster images.

- ``core``: ``Image``, the implementation of ``gws.Image`` on top of a PIL image, and the factories
  that create images (exported by this package).
- ``format``: predefined image formats, formats from configuration, MIME type selection.
- ``util``: thumbnails, cached 1x1 pixels, QR codes, PIL drawing helpers.

Example::

    img = gws.lib.image.from_size((256, 256))
    img.compose(gws.lib.image.from_path('/data/overlay.png'), opacity=0.5)
    png = img.to_bytes(gws.lib.mime.PNG, {'mode': 'P'})
"""

from .core import *
from . import format, util
