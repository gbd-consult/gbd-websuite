"""QGIS grabber."""

import gws
import gws.base.grabber.box
import gws.lib.image

from . import provider


class Object(gws.base.grabber.box.Object):
    serviceProvider: provider.Object
    params: dict

    def configure(self):
        self.serviceProvider = self.cfg('_defaultProvider')
        self.params = self.cfg('_defaultParams')

    def fetch_box(self, extent, width, height):
        w = gws.u.to_rounded_int(width)
        h = gws.u.to_rounded_int(height)

        blob = self.serviceProvider.get_map(
            None,
            gws.Bounds(crs=self.targetCrs, extent=extent),
            w,
            h,
            self.params,
        )

        img = gws.lib.image.from_size((w, h))
        img.paste(gws.lib.image.from_bytes(blob), (0, 0))
        return img
