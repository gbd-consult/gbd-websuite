"""Raster layer grabber."""

import gws
import gws.base.grabber.box
import gws.lib.image
import gws.lib.mapserver.core

from . import provider


class Object(gws.base.grabber.box.Object):
    serviceProvider: provider.Object
    msOptions: gws.MapServerLayerOptions

    def configure(self):
        self.serviceProvider = self.cfg('_defaultProvider')
        self.msOptions = self.cfg('_defaultMsOptions')
        self.sourceCrs = self.targetCrs
        self.maxRequestPixels = 9000

    def fetch_box(self, bounds, width, height, params=None):
        ms_map = gws.lib.mapserver.core.new_map()
        ms_map.add_layer(self.msOptions)
        return ms_map.draw(bounds, (width, height))
