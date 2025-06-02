"""Tile provider."""

from typing import Optional, cast

import gws
import gws.base.layer
import gws.config.util
import gws.lib.crs
import gws.lib.extent
import gws.gis.zoom
import gws.lib.metadata
import gws.lib.net


class Config(gws.Config):
    """Tile provider configuration."""
    
    maxRequests: int = 0
    """Max concurrent requests to this source."""
    url: gws.Url
    """Rest url with placeholders {x}, {y} and {z}."""
    grid: Optional[gws.base.layer.GridConfig]
    """Source grid."""


class Object(gws.Node):
    url: gws.Url
    grid: Optional[gws.TileGrid]
    maxRequests: int

    def configure(self):
        self.url = self.cfg('url')
        self.maxRequests = self.cfg('maxRequests')

        p = self.cfg('grid', default=gws.Config())
        self.grid = gws.TileGrid(
            origin=p.origin or gws.Origin.nw,
            tileSize=p.tileSize or 256,
        )
        crs = p.crs or gws.lib.crs.WEBMERCATOR
        extent = p.extent or (gws.lib.crs.WEBMERCATOR_SQUARE if crs == gws.lib.crs.WEBMERCATOR else crs.extent)
        self.grid.bounds = gws.Bounds(crs=crs, extent=extent)
        self.grid.resolutions = (
                p.resolutions or
                gws.gis.zoom.resolutions_from_bounds(self.grid.bounds, self.grid.tileSize))
