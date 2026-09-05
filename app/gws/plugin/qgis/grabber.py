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
        self.sourceCrs = self.targetCrs

    def fetch_box(self, bounds, width, height, params=None):
        blob = self.serviceProvider.get_map(
            None,
            bounds,
            width,
            height,
            gws.u.merge(self.params, params),
        )
        return gws.lib.image.from_bytes(blob)
