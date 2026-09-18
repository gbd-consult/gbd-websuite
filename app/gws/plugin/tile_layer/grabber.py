"""Tile layer grabber."""

import gws
import gws.base.grabber.tile
import gws.lib.grid

from . import provider


class Object(gws.base.grabber.tile.Object):
    serviceProvider: provider.Object

    def __init__(self, opts: gws.base.grabber.Options, serviceProvider: provider.Object):
        super().__init__(opts)
        self.serviceProvider = serviceProvider

        sg = self.serviceProvider.grid

        self.sourceCrs = sg.crs
        self.sourceTms = gws.lib.grid.matrix_set_for_grid(sg, self.serviceProvider.maxLevel)

    def fetch_tile_as_bytes(self, tm, col, row):
        return self.serviceProvider.get_tile(col, row, int(tm.identifier))
