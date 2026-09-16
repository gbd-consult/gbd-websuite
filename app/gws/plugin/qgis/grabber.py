"""QGIS grabber."""

import gws
import gws.base.grabber.box

from . import provider


class Object(gws.base.grabber.box.Object):
    serviceProvider: provider.Object
    params: dict

    def __init__(self, opts: gws.base.grabber.Options, params: dict):
        super().__init__(opts)
        self.serviceProvider = opts.provider
        self.params = params
        self.sourceCrs = self.targetCrs

    def fetch_box_as_bytes(self, bounds, width, height, params=None):
        return self.serviceProvider.get_map(
            bounds,
            width,
            height,
            gws.u.merge(self.params, params),
        )

    def fetch_box_as_image(self, bounds, width, height, params=None):
        return self.as_image((self.fetch_box_as_bytes(bounds, width, height, params), None))
