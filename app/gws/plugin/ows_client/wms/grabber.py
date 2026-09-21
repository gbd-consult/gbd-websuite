"""WMS grabber."""

import gws
import gws.base.grabber.box

from . import provider


class Object(gws.base.grabber.box.Object):
    provider: provider.Object
    sourceLayers: list[gws.SourceLayer]

    def __init__(self, opts: gws.base.grabber.Options, provider: provider.Object, sourceLayers: list[gws.SourceLayer], sourceCrs: gws.Crs):
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
