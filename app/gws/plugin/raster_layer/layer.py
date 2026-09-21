"""Raster image layer."""

from typing import Optional

import gws
import gws.base.layer
import gws.base.shape
import gws.config.util
import gws.lib.bounds
import gws.lib.extent

from . import grabber, provider


@gws.ext.config.layer('raster')
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


@gws.ext.object.layer('raster')
class Object(gws.base.layer.image.Object):
    provider: provider.Object
    msOptions: gws.MapServerLayerOptions
    entries: list[provider.ImageEntry]

    def configure(self):
        self.msOptions = gws.MapServerLayerOptions(
            type=gws.MapServerLayerType.raster,
            processing=self.cfg('processing', default=[]),
            transparentColor=self.cfg('transparentColor'),
            sldPath=self.cfg('sldPath'),
            sldName=self.cfg('sldName'),
        )
        self.configure_layer()

    def create_cache_name(self, cache):
        return gws.u.sha256([
            self.provider.cache_hash(),
            [e.path for e in self.entries],
            self.cfg('processing', default=[]),
            self.cfg('transparentColor') or '',
            self.cfg('sldPath') or '',
            self.cfg('sldName') or '',
            vars(self.imageFormat),
            list(self.wgsExtent),
            cache.requestBuffer,
            cache.requestTiles,
        ])[: gws.base.layer.image.CACHE_NAME_LENGTH]

    def create_grabber(self, opts):
        return grabber.Object(opts, msOptions=self.msOptions)

    def configure_provider(self):
        gws.config.util.configure_provider_for(self, provider.Object)
        
        default_crs = self.provider.crs or self.mapCrs
        self.entries = self.provider.enumerate_images(default_crs)
        if not self.entries:
            raise gws.ConfigurationError('no images found')

        self.msOptions.crs = self.entries[0].bounds.crs
        self.msOptions.tileIndex = self.provider.make_tile_index(
            self.entries,
            file_name=f'raster_layer_{self.uid}',
        )
        # self.msOptions.path = self.entries[0].path

    def configure_extent(self):
        if super().configure_extent():
            return True
        b = gws.lib.bounds.union([e.bounds for e in self.entries])
        self.wgsExtent = gws.lib.extent.transform_to_wgs(b.extent, b.crs)
        return True
