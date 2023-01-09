import re

import gws
import gws.gis.crs
import gws.gis.bounds
import gws.gis.zoom
import gws.base.layer
import gws.types as t


@gws.ext.config.layer('tile')
class Config(gws.base.layer.Config):
    """Tile layer"""
    display: gws.LayerDisplayMode = gws.LayerDisplayMode.tile 
    """layer display mode"""
    maxRequests: int = 0 
    """max concurrent requests to this source"""
    url: gws.Url 
    """rest url with placeholders {x}, {y} and {z}"""


_GRID_DEFAULTS = gws.TileGrid(
    bounds=gws.Bounds(
        crs=gws.gis.crs.WEBMERCATOR,
        extent=gws.gis.crs.WEBMERCATOR_SQUARE,
    ),
    corner='lt',
    tileSize=256,
)


@gws.ext.object.layer('tile')
class Object(gws.base.layer.Object):
    url: gws.Url

    def configure(self):
        # with reqSize=1 MP will request the same tile multiple times
        # reqSize=4 is more efficient, however, reqSize=1 yields the first tile faster
        # which is crucial when browsing non-cached low resolutions
        # so, let's use 1 as default, overridable in the config
        #
        # @TODO make MP cache network requests

        self.url = self.var('url')

        p = self.var('sourceGrid', default=gws.Config())
        self.sourceGrid = gws.TileGrid(
            corner=p.corner or 'lt',
            tileSize=p.tileSize or 256,
        )
        crs = p.crs or gws.gis.crs.WEBMERCATOR
        extent = p.extent or (gws.gis.crs.WEBMERCATOR_SQUARE if crs == gws.gis.crs.WEBMERCATOR else crs.extent)
        self.sourceGrid.bounds = gws.Bounds(crs=crs, extent=extent)
        self.sourceGrid.resolutions = (
                p.resolutions or
                gws.gis.zoom.resolutions_from_bounds(self.sourceGrid.bounds, self.sourceGrid.tileSize))

        p = self.var('grid', default=gws.Config())
        self.grid = gws.TileGrid(
            corner=p.corner or 'lt',
            tileSize=p.tileSize or 256,
        )
        crs = self.parentBounds.crs
        extent = (
            p.extent or
            self.sourceGrid.bounds.extent if crs == self.sourceGrid.bounds.crs else self.parentBounds.extent)
        self.grid.bounds = gws.Bounds(crs=crs, extent=extent)
        self.grid.resolutions = (
                p.resolutions or
                gws.gis.zoom.resolutions_from_bounds(self.grid.bounds, self.grid.tileSize))

        if not self.configure_bounds():
            self.bounds = self.parentBounds

        self.configure_metadata()
        self.configure_legend()

    def props(self, user):
        p = super().props(user)
        if self.displayMode == gws.LayerDisplayMode.client:
            return gws.merge(p, type='xyz', url=self.url)
        return p

    def render(self, lri):
        return gws.base.layer.util.generic_raster_render(self, lri)

    def mapproxy_config(self, mc, options=None):
        if self.displayMode == gws.LayerDisplayMode.client:
            return

        # we use {x} like in Ol, mapproxy wants %(x)s
        url = self.url
        url = url.replace('{x}', '%(x)s')
        url = url.replace('{y}', '%(y)s')
        url = url.replace('{z}', '%(z)s')

        sg = self.sourceGrid

        if sg.corner == 'lt':
            origin = 'nw'
        elif sg.corner == 'lb':
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
