"""WMTS grabber."""

import gws
import gws.base.grabber.tile

from . import provider


class Object(gws.base.grabber.tile.Object):
    serviceProvider: provider.Object
    tms: gws.TileMatrixSet
    urlTemplate: str

    def __init__(self, opts: gws.base.grabber.Options, serviceProvider: provider.Object, tms: gws.TileMatrixSet, urlTemplate: str):
        super().__init__(opts)
        self.serviceProvider = serviceProvider
        self.tms = tms
        self.urlTemplate = urlTemplate

        self.sourceCrs = self.tms.crs
        self.sourceMatrices = self.tms.matrices

    def fetch_tile_as_bytes(self, tm, col, row):
        return self.serviceProvider.get_tile(self.urlTemplate, tm.identifier, col, row)

    def fetch_tile_as_image(self, tm, col, row):
        return self.to_image(self.fetch_tile_as_bytes(tm, col, row))
