"""MBTiles grabber."""

import gws
import gws.base.grabber.box
import gws.lib.mapserver.core

from . import provider


class Object(gws.base.grabber.box.Object):
    serviceProvider: provider.Object
    msOptions: gws.MapServerLayerOptions

    def __init__(self, opts: gws.base.grabber.Options, msOptions: gws.MapServerLayerOptions):
        super().__init__(opts)
        self.serviceProvider = opts.provider
        self.msOptions = msOptions
        self.sourceCrs = self.targetCrs
        self.maxRequestPixels = 9000

    def fetch_box_as_image(self, bounds, width, height, params=None):
        ms_map = gws.lib.mapserver.core.new_map()
        ms_map.add_layer(self.msOptions)
        return ms_map.draw(bounds, (width, height))

    def fetch_box_as_bytes(self, bounds, width, height, params=None):
        return self.as_bytes((None, self.fetch_box_as_image(bounds, width, height, params)))
