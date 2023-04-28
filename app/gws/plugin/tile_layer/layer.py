"""Tile layer."""

import gws
import gws.base.layer
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
    corner=gws.Corner.nw,
    tileSize=256,
)


class Object(gws.base.layer.Object):
    canRenderBox = True
    canRenderXyz = True
    canRenderSvg = False

    provider: provider.Object

    def configure(self):
        self.configure_layer()

    def configure_provider(self):
        self.provider = provider.get_for(self)
        return True

    def configure_bounds(self):
        if super().configure_bounds():
            return True
        self.bounds = gws.gis.bounds.transform(
            self.provider.grid.bounds,
            self.defaultBounds.crs)
        return True

    def configure_grid(self):
        p = self.cfg('grid', default=gws.Config())

        self.grid = gws.TileGrid(
            corner=p.corner or gws.Corner.nw,
            tileSize=p.tileSize or 256,
        )

        if p.extent:
            extent = p.extent
        elif self.bounds.crs == self.provider.grid.bounds.crs:
            extent = self.provider.grid.bounds.extent
        else:
            extent = self.defaultBounds.extent
        self.grid.bounds = gws.Bounds(crs=self.bounds.crs, extent=extent)

        if p.resolutions:
            self.grid.resolutions = p.resolutions
        else:
            self.grid.resolutions = gws.gis.zoom.resolutions_from_bounds(self.grid.bounds, self.grid.tileSize)

    def mapproxy_config(self, mc, options=None):
        if self.displayMode == gws.LayerDisplayMode.client:
            return

        # we use {x} like in Ol, mapproxy wants %(x)s
        url = self.provider.url
        url = url.replace('{x}', '%(x)s')
        url = url.replace('{y}', '%(y)s')
        url = url.replace('{z}', '%(z)s')

        sg = self.provider.grid

        if sg.corner == gws.Corner.nw:
            origin = 'nw'
        elif sg.corner == gws.Corner.sw:
            origin = 'sw'
        else:
            raise gws.Error(f'invalid grid corner {sg.corner!r}')

        back_grid_uid = mc.grid(gws.compact({
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
            return gws.merge(p, type='xyz', url=self.provider.url)
        return p

    def render(self, lri):
        return gws.base.layer.util.mpx_raster_render(self, lri)
