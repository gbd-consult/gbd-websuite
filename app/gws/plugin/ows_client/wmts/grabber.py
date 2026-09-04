"""WMTS grabber."""

import gws
import gws.base.grabber.tile

from . import provider


class Object(gws.base.grabber.tile.Object):
    serviceProvider: provider.Object
    tms: gws.TileMatrixSet
    urlTemplate: str

    def configure(self):
        self.serviceProvider = self.cfg('_defaultProvider')
        self.tms = self.cfg('_defaultTms')
        self.urlTemplate = self.cfg('_defaultUrlTemplate')

        self.sourceCrs = self.tms.crs
        self.sourceMatrices = self.tms.matrices

    def fetch_source_tile(self, m, col, row):
        return self.serviceProvider.get_tile(self.urlTemplate, m.uid, col, row)
