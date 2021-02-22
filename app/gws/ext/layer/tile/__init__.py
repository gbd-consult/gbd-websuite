import re
import math

import gws
import gws.common.layer
import gws.types as t
import gws.gis.source
import gws.tools.json2

_EPSG_3857_RADIUS = 6378137

_EPSG_3857_EXTENT = [
    -(math.pi * _EPSG_3857_RADIUS),
    -(math.pi * _EPSG_3857_RADIUS),
    +(math.pi * _EPSG_3857_RADIUS),
    +(math.pi * _EPSG_3857_RADIUS),
]


class ServiceConfig:
    """Tile service configuration"""

    extent: t.Optional[t.Extent]  #: service extent
    crs: t.Crs = 'EPSG:3857'  #: service CRS
    origin: str = 'nw'  #: position of the first tile (nw or sw)
    tileSize: int = 256  #: tile size


class Config(gws.common.layer.ImageTileConfig):
    """Tile layer"""

    maxRequests: int = 0  #: max concurrent requests to this source
    service: t.Optional[ServiceConfig] = {}  #: service configuration
    url: t.Url  #: rest url with placeholders {x}, {y} and {z}


class Object(gws.common.layer.ImageTile):
    def configure(self):
        super().configure()

        self.url = self.var('url')
        self.service: ServiceConfig = self.var('service')

        if not self.service.extent:
            if self.service.crs == gws.EPSG_3857:
                self.service.extent = _EPSG_3857_EXTENT
            else:
                raise gws.Error(r'service extent required for crs {self.service.crs!r}')

    @property
    def props(self):
        if self.display == 'client':
            return gws.merge(super().props, type='xyz', url=self.url)
        return super().props

    @property
    def own_bounds(self):
        # in the "native" projection, use the service extent
        # otherwise, the map extent
        if self.service.crs == self.crs:
            return t.Bounds(
                crs=self.service.crs,
                extent=self.service.extent)

    def mapproxy_config(self, mc, options=None):
        # we use {x} like in Ol, mapproxy wants %(x)s
        url = re.sub(
            r'{([xyz])}',
            r'%(\1)s',
            self.url)

        grid_uid = mc.grid(gws.compact({
            'origin': self.service.origin,
            'bbox': self.service.extent,
            # 'res': res,
            'srs': self.service.crs,
            'tile_size': [self.service.tileSize, self.service.tileSize],
        }))

        src = self.mapproxy_back_cache_config(mc, url, grid_uid)
        self.mapproxy_layer_config(mc, src)
