"""WMS grabber."""

import gws
import gws.base.grabber.box

from . import provider


class Object(gws.base.grabber.box.Object):
    """Box grabber that fetches map images from a WMS service with GetMap."""

    provider: provider.Object
    """WMS service provider."""
    sourceLayers: list[gws.SourceLayer]
    """Source layers to render."""

    def __init__(self, opts: gws.base.grabber.Options, provider: provider.Object, sourceLayers: list[gws.SourceLayer], sourceCrs: gws.Crs):
        """Create the grabber.

        Args:
            opts: Grabber options.
            provider: WMS service provider.
            sourceLayers: Source layers to render.
            sourceCrs: CRS of the GetMap requests.
        """
        super().__init__(opts)
        self.provider = provider
        self.sourceLayers = sourceLayers
        self.sourceCrs = sourceCrs
        self.maxRequestPixels = self.provider.maxRequestPixels

    def fetch_box_as_bytes(self, bounds, w, h, params=None):
        return self.provider.get_map(
            bounds,
            w,
            h,
            self.sourceLayers,
            self.mimeType,
        )

    def fetch_box_as_image(self, bounds, w, h, params=None):
        return self.to_image(self.fetch_box_as_bytes(bounds, w, h, params))
