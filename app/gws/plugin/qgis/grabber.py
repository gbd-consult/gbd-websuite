"""Raster grabber for QGIS Server."""

import gws
import gws.base.grabber.box

from . import provider


class Object(gws.base.grabber.box.Object):
    """Grabber that renders boxes with GetMap requests to QGIS Server."""

    provider: provider.Object
    """QGIS provider."""
    params: dict
    """Fixed GetMap parameters, e.g. ``LAYERS`` and ``FILTER``."""

    def __init__(self, opts: gws.base.grabber.Options, provider: provider.Object, params: dict):
        """Create the grabber.

        Args:
            opts: Grabber options.
            provider: QGIS provider.
            params: Fixed GetMap parameters, merged with per-request parameters.
        """
        super().__init__(opts)
        self.provider = provider
        self.params = params
        self.sourceCrs = self.targetCrs

    def fetch_box_as_bytes(self, bounds, w, h, params=None):
        return self.provider.get_map(
            bounds,
            w,
            h,
            gws.u.merge(self.params, params),
        )

    def fetch_box_as_image(self, bounds, w, h, params=None):
        return self.to_image(self.fetch_box_as_bytes(bounds, w, h, params))
