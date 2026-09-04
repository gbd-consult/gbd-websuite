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

    def fetch_box(self, extent, width, height):
        w = gws.u.to_rounded_int(width)
        h = gws.u.to_rounded_int(height)

        ms_map = gws.lib.mapserver.core.new_map()
        ms_map.add_layer(self.msOptions)
        img = ms_map.draw(gws.Bounds(crs=self.targetCrs, extent=extent), (w, h))

        canvas = gws.lib.image.from_size((w, h))
        canvas.paste(img, (0, 0))
        return canvas
