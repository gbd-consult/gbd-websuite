"""Tile provider."""

from typing import Optional, cast

import gws
import gws.base.grabber
import gws.base.layer
import gws.base.layer
import gws.lib.grid
import gws.base.metadata
import gws.config.util
import gws.gis.zoom
import gws.lib.crs
import gws.lib.extent
import gws.lib.net


class Config(gws.Config):
    """Tile provider configuration."""
    
    maxLevel: int = 19
    """Max zoom level the source provides."""
    maxRequests: int = 0
    """Max concurrent requests to this source."""
    url: gws.Url
    """Rest url with placeholders {x}, {y} and {z}."""
    grid: Optional[gws.lib.grid.MapGridConfig]
    """Source grid."""


class Object(gws.Node):
    url: gws.Url
    grid: gws.MapGrid
    maxLevel: int
    maxRequests: int

    def configure(self):
        self.url = self.cfg('url')
        self.maxLevel = self.cfg('maxLevel')
        self.maxRequests = self.cfg('maxRequests')

        p = cast(gws.lib.grid.MapGridConfig, self.cfg('grid', default=gws.Config()))
        opts = gws.lib.grid.MapGridOptions(
            crs=gws.lib.crs.require(p.crs) if p.crs else gws.lib.crs.WEBMERCATOR,
            extent=p.extent,
            baseResolution=p.baseResolution,
            tileSize=p.tileSize,
        )
        self.grid = gws.lib.grid.new(opts)

    def get_tile(self, x: int, y: int, z: int) -> bytes:
        url = self.url
        url = url.replace('{x}', str(x))
        url = url.replace('{y}', str(y))
        url = url.replace('{z}', str(z))

        res = gws.lib.net.http_request(url)
        if not res.ok:
            raise gws.ExternalServiceError(f'tile request failed: status={res.status_code} url={url!r}')
        if not res.content_type.startswith('image/'):
            raise gws.ExternalServiceError(f'tile request failed: content type {res.content_type!r} url={url!r}')
        return res.content
