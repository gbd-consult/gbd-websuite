import math
import re

import gws
import gws.types as t
import gws.base.layer
import gws.lib.gis
import gws.lib.json2

from . import core, image

_EPSG_3857_RADIUS = 6378137

_EPSG_3857_EXTENT = [
    -(math.pi * _EPSG_3857_RADIUS),
    -(math.pi * _EPSG_3857_RADIUS),
    +(math.pi * _EPSG_3857_RADIUS),
    +(math.pi * _EPSG_3857_RADIUS),
]


class ServiceConfig:
    """Tile service configuration"""
    extent: t.Optional[gws.Extent]  #: service extent
    crs: gws.Crs = 'EPSG:3857'  #: service CRS
    origin: str = 'nw'  #: position of the first tile (nw or sw)
    tileSize: int = 256  #: tile size


@gws.ext.Config('layer.tile')
class Config(image.Config):
    """Tile layer"""
    display: core.DisplayMode = core.DisplayMode.tile  #: layer display mode
    maxRequests: int = 0  #: max concurrent requests to this source
    service: t.Optional[ServiceConfig] = {}  #: service configuration
    url: gws.Url  #: rest url with placeholders {x}, {y} and {z}


@gws.ext.Object('layer.tile')
class Object(image.Object):
    url: gws.Url
    service: ServiceConfig

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
            return gws.Bounds(
                crs=self.service.crs,
                extent=self.service.extent)

    def configure(self):
        # with reqSize=1 MP will request the same tile multiple times
        # reqSize=4 is more efficient, however, meta=1 yields the first tile faster
        # which is crucial when browsing non-cached low resoltions
        # so, let's use 1 as default, overridable in the config
        #
        # @TODO make MP cache network requests

        self.grid.reqSize = self.grid.reqSize or 1

        self.url = self.var('url')
        self.service: ServiceConfig = self.var('service')

        if not self.service.extent:
            if self.service.crs == gws.EPSG_3857:
                self.service.extent = _EPSG_3857_EXTENT
            else:
                raise gws.Error(r'service extent required for crs {self.service.crs!r}')

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
