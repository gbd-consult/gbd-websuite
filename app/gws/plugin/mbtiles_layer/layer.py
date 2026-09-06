"""MBTiles based layer."""

from typing import Optional

import gws
import gws.base.layer
import gws.config.util
import gws.lib.gdalx
import gws.lib.bounds

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

    def configure_bounds(self):
        if super().configure_bounds():
            return True
        with gws.lib.gdalx.open_raster(self.msOptions.path) as gd:
            self.bounds = gws.lib.bounds.transform(gd.bounds(), self.mapCrs)
        self.msOptions.crs = self.bounds.crs
        return True

    def configure_grabber(self):
        self.grabber = self.create_grabber()
        return True

    def create_cache_name(self):
        return gws.u.sha256([
            self.serviceProvider.cache_hash(),
            self.cfg('processing', default=[]),
            self.cfg('transparentColor') or '',
            self.mapCrs.srid,
            vars(self.imageFormat),
            list(self.bounds.extent),
            self.cache.requestBuffer,
            self.cache.requestTiles,
        ], maxlen=gws.base.layer.core.CACHE_NAME_LENGTH)

    def create_grabber(self):
        return self.root.create_shared(
            grabber.Object,
            crs=self.mapCrs.srid,
            extent=self.bounds.extent,
            imageFormat=self.imageFormat,
            _defaultCache=self.cache,
            _defaultProvider=self.serviceProvider,
            _defaultMsOptions=self.msOptions,
        )
