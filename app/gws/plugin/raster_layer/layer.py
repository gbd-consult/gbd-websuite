"""Raster image layer."""

from typing import Optional

import gws
import gws.base.layer
import gws.base.shape
import gws.config.util
import gws.gis.gdalx
import gws.gis.ms
import gws.gis.ms.util
import gws.gis.zoom
import gws.lib.bounds
import gws.lib.crs
import gws.lib.osx

from . import provider

gws.ext.new.layer('raster')


class Config(gws.base.layer.Config):
    """Raster layer"""

    provider: provider.Config
    """Raster provider"""
    processing: Optional[list[str]]
    """Processing directives."""
    transparentColor: Optional[str]
    """Color to treat as transparent in the layer."""
    sldPath: Optional[gws.FilePath]
    """Path to SLD file for styling the layer."""
    sldName: Optional[str]
    """Name of an SLD NamedLayer to apply."""


class Object(gws.base.layer.image.Object):
    serviceProvider: provider.Object
    entries: list[provider.ImageEntry]
    msOptions: gws.gis.ms.LayerOptions

    def configure(self):
        self.msOptions = gws.gis.ms.LayerOptions(
            type=gws.gis.ms.LayerType.raster,
            processing=self.cfg('processing', default=[]),
            transparentColor=self.cfg('transparentColor'),
            sldPath=self.cfg('sldPath'),
            sldName=self.cfg('sldName'),
        )
        self.configure_layer()

    def configure_provider(self):
        gws.config.util.configure_service_provider_for(self, provider.Object)
        
        default_crs = self.serviceProvider.crs or self.parentBounds.crs
        self.entries = self.serviceProvider.enumerate_images(default_crs)
        if not self.entries:
            raise gws.ConfigurationError('no images found')

        self.msOptions.crs = self.entries[0].bounds.crs
        self.msOptions.tileIndex = self.serviceProvider.make_tile_index(
            self.entries,
            file_name=f'raster_layer_{self.uid}',
        )
        # self.msOptions.path = self.entries[0].path

    def configure_bounds(self):
        if super().configure_bounds():
            return True
        b = gws.lib.bounds.union([e.bounds for e in self.entries])
        self.bounds = gws.lib.bounds.transform(b, self.parentBounds.crs)
        return True

    def configure_grid(self):
        p = self.cfg('grid') or gws.base.layer.GridConfig()

        self.grid = gws.TileGrid(
            origin=p.origin or gws.Origin.nw,
            tileSize=p.tileSize or 256,
            bounds=self.bounds,
        )

        if p.resolutions:
            self.grid.resolutions = p.resolutions
        else:
            self.grid.resolutions = gws.gis.zoom.resolutions_from_bounds(self.grid.bounds, self.grid.tileSize)

    def render(self, lri):
        return gws.gis.ms.util.raster_render(self, lri)
