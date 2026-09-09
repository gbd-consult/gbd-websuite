"""MBTiles based layer."""

from typing import Optional

import gws
import gws.base.layer
import gws.config.util
import gws.lib.gdalx
import gws.lib.bounds
import gws.lib.extent

from . import grabber, provider

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

    def configure_extent(self):
        with gws.lib.gdalx.open_raster(self.msOptions.path) as gd:
            b = gd.bounds()
        self.msOptions.crs = b.crs
        if super().configure_extent():
            return True
        self.wgsExtent = gws.lib.extent.transform_to_wgs(b.extent, b.crs)
        return True

    def create_cache_name(self, cache):
        return gws.u.sha256([
            self.serviceProvider.cache_hash(),
            self.cfg('processing', default=[]),
            self.cfg('transparentColor') or '',
            vars(self.imageFormat),
            list(self.wgsExtent),
            cache.requestBuffer,
            cache.requestTiles,
        ])[: gws.base.layer.core.CACHE_NAME_LENGTH]

    def create_grabber(self, opts):
        return grabber.Object(opts, msOptions=self.msOptions)
