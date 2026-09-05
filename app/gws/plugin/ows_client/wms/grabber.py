"""WMS grabber."""

import gws
import gws.base.grabber.box
import gws.lib.image

from . import provider


class Object(gws.base.grabber.box.Object):
    serviceProvider: provider.Object
    sourceLayers: list[gws.SourceLayer]

    def configure(self):
        self.serviceProvider = self.cfg('_defaultProvider')
        self.sourceLayers = self.cfg('_defaultSourceLayers')
        self.sourceCrs = self.cfg('_defaultSourceCrs')
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
