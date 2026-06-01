"""MBTiles based layer."""

from typing import Optional

import gws
import gws.base.layer
import gws.config.util
import gws.lib.gdalx
import gws.lib.mapserver
import gws.gis.zoom
import gws.lib.bounds

from . import provider

gws.ext.new.layer('mbtiles')


class Config(gws.base.layer.Config):
    """MBTiles layer"""

    provider: provider.Config
    """Provider configuration."""
    processing: Optional[list[str]]
    """Processing directives."""
    transparentColor: Optional[str]
    """Color to treat as transparent in the layer."""


class Object(gws.base.layer.image.Object):
    serviceProvider: provider.Object

    def configure(self):
        self.msOptions = gws.MapServerLayerOptions(
            type=gws.MapServerLayerType.raster,
            processing=self.cfg('processing', default=[]),
            transparentColor=self.cfg('transparentColor', default=None),
        )
        self.configure_layer()

    def configure_provider(self):
        gws.config.util.configure_service_provider_for(self, provider.Object)
        self.msOptions.path = self.serviceProvider.path

    def configure_bounds(self):
        if super().configure_bounds():
            return True
        with gws.lib.gdalx.open_raster(self.msOptions.path) as gd:
            self.bounds = gws.lib.bounds.transform(gd.bounds(), self.mapCrs)
        self.msOptions.crs = self.bounds.crs
        return True

    def configure_grid(self):
        p = self.cfg('grid', default=gws.Config())

        self.grid = gws.TileGrid(
            origin=p.origin or gws.Origin.nw,
            tileSize=p.tileSize or 256,
            bounds=self.bounds,
        )

        if p.resolutions:
            self.grid.resolutions = p.resolutions
        else:
            self.grid.resolutions = gws.gis.zoom.resolutions_from_bounds(self.grid.bounds, self.grid.tileSize)

    ##

    def render(self, lri):
        return gws.lib.mapserver.util.raster_render(self, lri)
