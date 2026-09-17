"""MBTiles grabber."""

import gws
import gws.base.grabber.box
import gws.lib.mapserver.core


class Object(gws.base.grabber.box.Object):
    msOptions: gws.MapServerLayerOptions

    def __init__(self, opts: gws.base.grabber.Options, msOptions: gws.MapServerLayerOptions):
        super().__init__(opts)
        self.msOptions = msOptions
        self.sourceCrs = self.targetCrs
        self.maxRequestPixels = 9000

    def fetch_box_as_image(self, bounds, w, h, params=None):
        ms_map = gws.lib.mapserver.core.new_map()
        ms_map.add_layer(self.msOptions)
        return ms_map.draw(bounds, (w, h))

    def fetch_box_as_bytes(self, bounds, w, h, params=None):
        return self.to_bytes(self.fetch_box_as_image(bounds, w, h, params))
