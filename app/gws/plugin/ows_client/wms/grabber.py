"""WMS grabber."""

import gws
import gws.base.grabber.box

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

    def fetch_box_as_bytes(self, bounds, width, height, params=None):
        return self.serviceProvider.get_map(
            bounds,
            width,
            height,
            self.sourceLayers,
            self.mime,
        )
