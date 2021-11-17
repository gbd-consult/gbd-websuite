import re

import gws
import gws.gis.crs
import gws.types as t

from . import image, types


class ServiceConfig:
    """Tile service configuration"""
    extent: t.Optional[gws.Extent]  #: service extent
    crs: gws.CrsId = 'EPSG:3857'  #: service CRS
    origin: str = 'nw'  #: position of the first tile (nw or sw)
    tileSize: int = 256  #: tile size


class Service(gws.Data):
    extent: gws.Extent
    crs: gws.ICrs
    origin: str
    tile_size: int


@gws.ext.Config('layer.tile')
class Config(image.Config):
    """Tile layer"""
    display: types.DisplayMode = types.DisplayMode.tile  #: layer display mode
    maxRequests: int = 0  #: max concurrent requests to this source
    service: t.Optional[ServiceConfig] = {}  # type:ignore #: service configuration
    url: gws.Url  #: rest url with placeholders {x}, {y} and {z}


@gws.ext.Object('layer.tile')
class Object(image.Object):
    url: gws.Url
    service: Service

    def props_for(self, user):
        p = super().props_for(user)
        if self.display == 'client':
            return gws.merge(p, type='xyz', url=self.url)
        return p

    @property
    def own_bounds(self):
        # in the "native" projection, use the service extent
        # otherwise, the map extent
        if self.service.crs.same_as(self.crs):
            return gws.Bounds(crs=self.service.crs, extent=self.service.extent)

    def configure(self):
        # with reqSize=1 MP will request the same tile multiple times
        # reqSize=4 is more efficient, however, reqSize=1 yields the first tile faster
        # which is crucial when browsing non-cached low resoltions
        # so, let's use 1 as default, overridable in the config
        #
        # @TODO make MP cache network requests

        self.grid.reqSize = self.grid.reqSize or 1
        self.url = self.var('url')

        p = self.var('service', default=gws.Data())
        self.service = Service(
            crs=gws.gis.crs.get(p.crs) or gws.gis.crs.get3857(),
            origin=p.origin,
            tile_size=p.tileSize,
            extent=p.extent)

        if not self.service.extent:
            if self.service.crs.srid == gws.gis.crs.c3857:
                self.service.extent = gws.gis.crs.c3857_extent
            else:
                raise gws.Error(f'service extent required for crs {self.service.crs.srid!r}')

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
            'srs': self.service.crs.epsg,
            'tile_size': [self.service.tile_size, self.service.tile_size],
        }))

        src = self.mapproxy_back_cache_config(mc, url, grid_uid)
        self.mapproxy_layer_config(mc, src)
