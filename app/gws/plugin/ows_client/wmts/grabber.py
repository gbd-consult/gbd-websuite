"""WMTS grabber."""

import gws
import gws.base.grabber.tile

from . import provider


class Object(gws.base.grabber.tile.Object):
    """Tile grabber that fetches source tiles from a WMTS service."""

    provider: provider.Object
    """WMTS service provider."""
    tms: gws.TileMatrixSet
    """Source tile matrix set."""
    urlTemplate: str
    """Tile URL template with ``{TileMatrix}``, ``{TileCol}`` and ``{TileRow}`` placeholders."""

    def __init__(self, opts: gws.base.grabber.Options, provider: provider.Object, tms: gws.TileMatrixSet, urlTemplate: str):
        """Create the grabber.

        The source CRS is the CRS of the tile matrix set.

        Args:
            opts: Grabber options.
            provider: WMTS service provider.
            tms: Source tile matrix set.
            urlTemplate: Tile URL template.
        """
        super().__init__(opts)
        self.provider = provider
        self.tms = tms
        self.urlTemplate = urlTemplate

        self.sourceCrs = self.tms.crs
        self.sourceTms = self.tms

    def fetch_tile_as_bytes(self, tm, col, row):
        return self.provider.get_tile(self.urlTemplate, tm.identifier, col, row)
