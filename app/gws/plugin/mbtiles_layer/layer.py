"""mbtiles based layer."""

from typing import Optional

import fnmatch

import gws
import gws.base.shape
import gws.base.layer
import gws.lib.image
import gws.lib.osx
import gws.lib.bounds
import gws.lib.crs
import gws.gis.zoom
import gws.gis.ms
import gws.gis.ms.util
import gws.gis.gdalx

gws.ext.new.layer('mbtiles')


class ProviderConfig(gws.Config):
    """Data provider."""

    type: str
    """Type ('file')"""
    path: gws.FilePath
    """List of image file paths."""


class Config(gws.base.layer.Config):
    """mbtiles layer"""

    provider: ProviderConfig
    """Picture provider"""
    processing: Optional[list[str]]
    """Processing directives."""
    transparentColor: Optional[str]
    """Color to treat as transparent in the layer."""


class Object(gws.base.layer.image.Object):
    msOptions: gws.gis.ms.LayerOptions

    def configure(self):
        self.msOptions = gws.gis.ms.LayerOptions(
            type=gws.gis.ms.LayerType.raster,
            processing=self.cfg('processing', default=[]),
            transparentColor=self.cfg('transparentColor', default=None),
        )
        self.configure_layer()

    def configure_provider(self):
        p = self.cfg('provider')
        self.msOptions.path = p.path

    def configure_bounds(self):
        if super().configure_bounds():
            return True
        with gws.gis.gdalx.open_raster(self.msOptions.path) as gd:
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
        return gws.gis.ms.util.raster_render(self, lri)
