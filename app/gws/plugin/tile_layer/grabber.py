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
        self.sourceMatrices = []
        
        for z in range(self.serviceProvider.maxLevel + 1):
            nx, ny = gws.lib.grid.tile_count_for_level(sg, z)
            self.sourceMatrices.append(gws.TileMatrix(
                identifier=str(z),
                resolution=gws.lib.grid.resolution_for_level(sg, z),
                x=sg.extent[0],
                y=sg.extent[3],
                width=nx,
                height=ny,
                tileWidth=sg.tileSize,
                tileHeight=sg.tileSize,
                extent=sg.extent,
            ))

    def fetch_tile_as_bytes(self, tm, col, row):
        return self.serviceProvider.get_tile(col, row, int(tm.identifier))

    def fetch_tile_as_image(self, tm, col, row):
        return self.to_image(self.fetch_tile_as_bytes(tm, col, row))
