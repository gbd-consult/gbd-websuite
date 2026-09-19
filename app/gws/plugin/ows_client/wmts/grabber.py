"""WMTS grabber."""

import gws
import gws.base.grabber.tile

from . import provider


class Object(gws.base.grabber.tile.Object):
    provider: provider.Object
    tms: gws.TileMatrixSet
    urlTemplate: str

    def __init__(self, opts: gws.base.grabber.Options, provider: provider.Object, tms: gws.TileMatrixSet, urlTemplate: str):
        super().__init__(opts)
        self.provider = provider
        self.tms = tms
        self.urlTemplate = urlTemplate

        self.sourceCrs = self.tms.crs
        self.sourceTms = self.tms

    def fetch_tile_as_bytes(self, tm, col, row):
        return self.provider.get_tile(self.urlTemplate, tm.identifier, col, row)
