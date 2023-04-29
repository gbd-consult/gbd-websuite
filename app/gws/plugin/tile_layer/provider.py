"""Tile provider."""

import gws
import gws.base.layer
import gws.gis.crs
import gws.gis.extent
import gws.gis.zoom
import gws.lib.metadata
import gws.lib.net
import gws.types as t


class Config(gws.Config):
    maxRequests: int = 0
    """max concurrent requests to this source"""
    url: gws.Url
    """rest url with placeholders {x}, {y} and {z}"""
    grid: t.Optional[gws.base.layer.GridConfig]
    """source grid"""


class Object(gws.Node, gws.IProvider):
    url: gws.Url
    grid: t.Optional[gws.TileGrid]
    maxRequests: int

    def configure(self):
        self.url = self.cfg('url')
        self.maxRequests = self.cfg('maxRequests')

        p = self.cfg('grid', default=gws.Config())
        self.grid = gws.TileGrid(
            corner=p.corner or gws.Corner.nw,
            tileSize=p.tileSize or 256,
        )
        crs = p.crs or gws.gis.crs.WEBMERCATOR
        extent = p.extent or (gws.gis.crs.WEBMERCATOR_SQUARE if crs == gws.gis.crs.WEBMERCATOR else crs.extent)
        self.grid.bounds = gws.Bounds(crs=crs, extent=extent)
        self.grid.resolutions = (
                p.resolutions or
                gws.gis.zoom.resolutions_from_bounds(self.grid.bounds, self.grid.tileSize))


##

def get_for(obj: gws.INode) -> Object:
    p = obj.cfg('provider')
    if p:
        return obj.root.create_shared(Object, p)
    p = obj.cfg('_defaultProvider')
    if p:
        return p
    raise gws.Error(f'no provider found for {obj!r}')