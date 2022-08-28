import re

import gws
import gws.gis.crs
import gws.base.layer
import gws.types as t


class ServiceConfig:
    """Tile service configuration"""
    extent: t.Optional[gws.Extent]  #: service extent
    crs: gws.CRS = 'EPSG:3857'  #: service CRS
    origin: str = 'nw'  #: position of the first tile (nw or sw)
    tileSize: int = 256  #: tile size


class Service(gws.Data):
    extent: gws.Extent
    crs: gws.ICrs
    origin: str
    tileSize: int


@gws.ext.config.layer('tile')
class Config(gws.base.layer.Config):
    """Tile layer"""
    display: gws.LayerDisplayMode = gws.LayerDisplayMode.tile  #: layer display mode
    maxRequests: int = 0  #: max concurrent requests to this source
    service: t.Optional[ServiceConfig] = {}  # type:ignore #: service configuration
    url: gws.Url  #: rest url with placeholders {x}, {y} and {z}


@gws.ext.object.layer('tile')
class Object(gws.base.layer.Object):
    url: gws.Url
    service: Service

    def configure_source(self):
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
            tileSize=p.tileSize,
            extent=p.extent)

        if not self.service.extent:
            if self.service.crs.srid == 3857:
                self.service.extent = gws.gis.crs.CRS_3857_EXTENT
            else:
                raise gws.ConfigurationError(f'service extent required for crs {self.service.crs.srid!r}')

    def props(self, user):
        p = super().props(user)

        if self.displayMode == gws.LayerDisplayMode.client:
            p.type = 'xyz'
            p.url = self.url

        return p

    def own_bounds(self):
        # in the "native" projection, use the service extent
        # otherwise, the map extent
        if self.service.crs == self.crs:
            return gws.Bounds(crs=self.service.crs, extent=self.service.extent)

    def render(self, lri):
        return gws.base.layer.lib.generic_raster_render(self, lri)

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
            'tile_size': [self.service.tileSize, self.service.tileSize],
        }))

        src_uid = gws.base.layer.lib.mapproxy_back_cache_config(self, mc, url, grid_uid)
        gws.base.layer.lib.mapproxy_layer_config(self, mc, src_uid)
