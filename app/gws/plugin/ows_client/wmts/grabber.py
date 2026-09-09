"""WMTS grabber."""

import gws
import gws.base.grabber.tile

from . import provider


class Object(gws.base.grabber.tile.Object):
    serviceProvider: provider.Object
    tms: gws.TileMatrixSet
    urlTemplate: str

    def __init__(self, opts: gws.base.grabber.Options, tms: gws.TileMatrixSet, urlTemplate: str):
        super().__init__(opts)
        self.serviceProvider = opts.provider
        self.tms = tms
        self.urlTemplate = urlTemplate

        self.sourceCrs = self.tms.crs
        self.sourceMatrices = self.tms.matrices

    def fetch_source_tile(self, m, col, row):
        return self.serviceProvider.get_tile(self.urlTemplate, m.identifier, col, row)
