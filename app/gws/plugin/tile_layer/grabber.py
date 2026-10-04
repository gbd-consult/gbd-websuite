"""Tile layer grabber."""

import gws
import gws.base.grabber.tile
import gws.lib.grid

from . import provider


class Object(gws.base.grabber.tile.Object):
    """Grabber for XYZ tile services."""

    provider: provider.Object
    """Tile provider."""

    def __init__(self, opts: gws.base.grabber.Options, provider: provider.Object):
        """Create the grabber.

        Args:
            opts: Grabber options.
            provider: Tile provider.
        """
        super().__init__(opts)
        self.provider = provider

        sg = self.provider.grid

        self.sourceCrs = sg.crs
        self.sourceTms = gws.lib.grid.matrix_set_for_grid(sg, self.provider.maxLevel)

    def fetch_tile_as_bytes(self, tm, col, row):
        return self.provider.get_tile(col, row, int(tm.identifier))
