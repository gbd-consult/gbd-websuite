"""Tile layer."""

import gws
import gws.base.layer
import gws.config.util
import gws.gis.bounds
import gws.gis.crs
import gws.gis.zoom
from . import provider

gws.ext.new.layer('tile')


class Config(gws.base.layer.Config):
    """Tile layer"""
    provider: provider.Config
    """tile service provider"""
    display: gws.LayerDisplayMode = gws.LayerDisplayMode.tile
    """layer display mode"""


_GRID_DEFAULTS = gws.TileGrid(
    bounds=gws.Bounds(
        crs=gws.gis.crs.WEBMERCATOR,
        extent=gws.gis.crs.WEBMERCATOR_SQUARE,
    ),
    origin=gws.Origin.nw,
    tileSize=256,
)


class Object(gws.base.layer.image.Object):
    serviceProvider: provider.Object

    def configure(self):
        self.configure_layer()

    def configure_provider(self):
        return gws.config.util.configure_service_provider_for(self, provider.Object)

    #
    # reprojecting the world doesn't make sense, just use the map extent here
    # see also ows_provider/wmts
    #
    # def configure_bounds(self):
    #     if super().configure_bounds():
    #         return True
    #     self.bounds = gws.gis.bounds.transform(self.serviceProvider.grid.bounds, self.mapCrs)
    #     return True

    def configure_grid(self):
        p = self.cfg('grid', default=gws.Config())

        self.grid = gws.TileGrid(
            origin=p.origin or gws.Origin.nw,
            tileSize=p.tileSize or 256,
        )

        if p.extent:
            extent = p.extent
        elif self.bounds.crs == self.serviceProvider.grid.bounds.crs:
            extent = self.serviceProvider.grid.bounds.extent
        else:
            extent = self.parentBounds.extent
        self.grid.bounds = gws.Bounds(crs=self.bounds.crs, extent=extent)

        if p.resolutions:
            self.grid.resolutions = p.resolutions
        else:
            self.grid.resolutions = gws.gis.zoom.resolutions_from_bounds(self.grid.bounds, self.grid.tileSize)

    def mapproxy_config(self, mc, options=None):
        if self.displayMode == gws.LayerDisplayMode.client:
            return

        # we use {x} like in Ol, mapproxy wants %(x)s
        url = self.serviceProvider.url
        url = url.replace('{x}', '%(x)s')
        url = url.replace('{y}', '%(y)s')
        url = url.replace('{z}', '%(z)s')

        sg = self.serviceProvider.grid

        if sg.origin == gws.Origin.nw:
            origin = 'nw'
        elif sg.origin == gws.Origin.sw:
            origin = 'sw'
        else:
            raise gws.Error(f'invalid grid origin {sg.origin!r}')

        back_grid_uid = mc.grid(gws.u.compact({
            'origin': origin,
            'srs': sg.bounds.crs.epsg,
            'bbox': sg.bounds.extent,
            'res': sg.resolutions,
            'tile_size': [sg.tileSize, sg.tileSize],
        }))

        src_uid = gws.base.layer.util.mapproxy_back_cache_config(self, mc, url, back_grid_uid)
        gws.base.layer.util.mapproxy_layer_config(self, mc, src_uid)

    ##

    def props(self, user):
        p = super().props(user)
        if self.displayMode == gws.LayerDisplayMode.client:
            return gws.u.merge(p, type='xyz', url=self.serviceProvider.url)
        return p

    def render(self, lri):
        return gws.base.layer.util.mpx_raster_render(self, lri)
