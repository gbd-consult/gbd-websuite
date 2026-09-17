"""QGIS grabber."""

import gws
import gws.base.grabber.box

from . import provider


class Object(gws.base.grabber.box.Object):
    serviceProvider: provider.Object
    params: dict

    def __init__(self, opts: gws.base.grabber.Options, serviceProvider: provider.Object, params: dict):
        super().__init__(opts)
        self.serviceProvider = serviceProvider
        self.params = params
        self.sourceCrs = self.targetCrs

    def fetch_box_as_bytes(self, bounds, w, h, params=None):
        return self.serviceProvider.get_map(
            bounds,
            w,
            h,
            gws.u.merge(self.params, params),
        )

    def fetch_box_as_image(self, bounds, w, h, params=None):
        return self.to_image(self.fetch_box_as_bytes(bounds, w, h, params))
