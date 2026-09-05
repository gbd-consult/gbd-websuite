"""Tile layer grabber."""

import gws
import gws.base.grabber.tile
import gws.lib.grid

from . import provider


class Object(gws.base.grabber.tile.Object):
    serviceProvider: provider.Object

    def configure(self):
        self.serviceProvider = self.cfg('_defaultProvider')

        sg = self.serviceProvider.grid
        self.sourceCrs = sg.crs
        self.sourceMatrices = []
        for z in range(self.serviceProvider.maxLevel + 1):
            nx, ny = gws.lib.grid.tile_count_for_level(sg, z)
            self.sourceMatrices.append(gws.TileMatrix(
                identifier=str(z),
                x=sg.extent[0],
                y=sg.extent[3],
                width=nx,
                height=ny,
                tileWidth=sg.tileSize,
                tileHeight=sg.tileSize,
                extent=sg.extent,
            ))

    def fetch_source_tile(self, m, col, row):
        return self.serviceProvider.get_tile(col, row, int(m.identifier))
