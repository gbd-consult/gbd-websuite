"""WMS grabber."""

import gws
import gws.base.grabber.box
import gws.lib.image

from . import provider


class Object(gws.base.grabber.box.Object):
    serviceProvider: provider.Object
    sourceLayers: list[gws.SourceLayer]

    def __init__(self, opts: gws.base.grabber.Options, sourceLayers: list[gws.SourceLayer], sourceCrs: gws.Crs):
        super().__init__(opts)
        self.serviceProvider = opts.provider
        self.sourceLayers = sourceLayers
        self.sourceCrs = sourceCrs
        self.maxRequestPixels = self.serviceProvider.maxRequestPixels

    def fetch_box(self, bounds, width, height, params=None):
        blob = self.serviceProvider.get_map(
            bounds,
            width,
            height,
            self.sourceLayers,
            self.mime,
        )
        return gws.lib.image.from_bytes(blob)
