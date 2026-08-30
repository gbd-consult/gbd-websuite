"""Tile layer."""

from typing import Optional

import gws
import gws.base.layer
import gws.config.util
import gws.lib.bounds
import gws.lib.crs
import gws.lib.grid
import gws.gis.zoom
from . import grabber, provider

gws.ext.new.layer('tile')


class Config(gws.base.layer.Config):
    """Tile layer"""

    provider: provider.Config
    """Tile service provider."""
    display: gws.LayerDisplayMode = gws.LayerDisplayMode.tile
    """Layer display mode."""
    devGrabber: bool = False
    """Use the grabber instead of MapProxy."""


_GRID_DEFAULTS = gws.TileGrid(
    bounds=gws.Bounds(
        crs=gws.lib.crs.WEBMERCATOR,
        extent=gws.lib.crs.WEBMERCATOR_SQUARE,
    ),
    origin=gws.Origin.nw,
    tileSize=256,
)


class Object(gws.base.layer.image.Object):
    serviceProvider: provider.Object
    devGrabber: Optional[grabber.Object]

    def configure(self):
        self.configure_layer()
        self.devGrabber = None
        if self.cfg('devGrabber'):
            self.devGrabber = self.create_grabber()

    def create_grabber(self):
        cache = self.cache or gws.LayerCache(maxAge=0, maxLevel=0)
        uid = 'grabber_' + gws.u.sha256([
            self.serviceProvider.uid,
            self.mapCrs.srid,
            vars(self.imageFormat),
            list(self.bounds.extent),
            cache.maxAge or 0,
            cache.maxLevel or 0,
            cache.requestTiles or 0,
        ])
        return self.root.create_shared(
            grabber.Object,
            crs=self.mapCrs.srid,
            extent=self.bounds.extent,
            imageFormat=self.imageFormat,
            blockSize=cache.requestTiles or 1,
            cacheMaxAge=cache.maxAge or 0,
            cacheMaxLevel=cache.maxLevel or 0,
            cacheUid=uid,
            _defaultProvider=self.serviceProvider,
        )

    def configure_provider(self):
        return gws.config.util.configure_service_provider_for(self, provider.Object)

    #
    # reprojecting the world doesn't make sense, just use the map extent here
    # see also ows_provider/wmts
    #
    # def configure_bounds(self):
    #     if super().configure_bounds():
    #         return True
    #     self.bounds = gws.lib.bounds.transform(self.serviceProvider.grid.bounds, self.mapCrs)
    #     return True

    def configure_grid(self):
        p = self.cfg('grid', default=gws.Config())

        self.grid = gws.TileGrid(
            origin=p.origin or gws.Origin.nw,
            tileSize=p.tileSize or 256,
        )

        if p.extent:
            extent = p.extent
        elif self.bounds.crs == self.serviceProvider.grid.crs:
            extent = self.serviceProvider.grid.extent
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

        back_grid_uid = mc.grid(gws.u.compact({
            'origin': 'nw',
            'srs': sg.crs.epsg,
            'bbox': sg.extent,
            'res': [gws.lib.grid.resolution_for_level(sg, z) for z in range(self.serviceProvider.maxLevel + 1)],
            'tile_size': [sg.tileSize, sg.tileSize],
        }))

        src_uid = gws.base.layer.util.mapproxy_back_cache_config(self, mc, url, back_grid_uid)
        gws.base.layer.util.mapproxy_layer_config(self, mc, src_uid)

    ##

    def props(self, user):
        p = super().props(user)
        if self.displayMode == gws.LayerDisplayMode.client:
            return gws.u.merge(p, type='xyz', url=self.serviceProvider.url)
        if self.devGrabber:
            g = self.devGrabber.grid
            zmax = gws.lib.grid.level_for_resolution(g, min(self.resolutions))
            p.grid = gws.base.layer.core.GridProps(
                origin=gws.Origin.nw,
                extent=g.extent,
                resolutions=[gws.lib.grid.resolution_for_level(g, z) for z in range(zmax + 1)],
                tileSize=g.tileSize,
            )
        return p

    def render(self, lri):
        if self.devGrabber:
            return self.render_with_grabber(lri)
        return gws.base.layer.util.mpx_raster_render(self, lri)

    def render_with_grabber(self, lri):

        if lri.type == gws.LayerRenderInputType.xyz:
            return gws.LayerRenderOutput(content=self.devGrabber.get_tile((lri.x, lri.y, lri.z)))
        if lri.type == gws.LayerRenderInputType.box:
            def get_box(bounds, width, height):
                return self.devGrabber.get_box(bounds.extent, width, height)

            content = gws.base.layer.util.generic_render_box(self, lri, get_box)
            return gws.LayerRenderOutput(content=content)
