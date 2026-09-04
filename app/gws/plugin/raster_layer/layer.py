"""Raster image layer."""

from typing import Optional

import gws
import gws.base.grabber.box
import gws.base.layer
import gws.base.shape
import gws.config.util
import gws.lib.gdalx
import gws.lib.grid
import gws.lib.mapserver
import gws.gis.zoom
import gws.lib.bounds
import gws.lib.crs
import gws.lib.osx

from . import grabber, provider

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

    def configure(self):
        self.msOptions = gws.MapServerLayerOptions(
            type=gws.MapServerLayerType.raster,
            processing=self.cfg('processing', default=[]),
            transparentColor=self.cfg('transparentColor'),
            sldPath=self.cfg('sldPath'),
            sldName=self.cfg('sldName'),
        )
        self.configure_layer()

    def configure_grabber(self):
        self.grabber = self.create_grabber()
        return True

    def create_grabber(self):
        cache = self.cache or gws.LayerCache(maxAge=0, maxLevel=0)
        uid = 'grabber_' + gws.u.sha256([
            self.serviceProvider.uid,
            [e.path for e in self.entries],
            self.cfg('processing', default=[]),
            self.cfg('transparentColor') or '',
            self.cfg('sldPath') or '',
            self.cfg('sldName') or '',
            self.mapCrs.srid,
            vars(self.imageFormat),
            list(self.bounds.extent),
            cache.maxAge or 0,
            cache.maxLevel or 0,
            cache.requestTiles or 0,
            cache.requestBuffer or 0,
        ])
        return self.root.create_shared(
            grabber.Object,
            uid=uid,
            crs=self.mapCrs.srid,
            extent=self.bounds.extent,
            imageFormat=self.imageFormat,
            blockSize=cache.requestTiles or gws.base.grabber.box.DEFAULT_BLOCK_SIZE,
            cacheMaxAge=cache.maxAge or 0,
            cacheMaxLevel=cache.maxLevel or 0,
            edgeBuffer=cache.requestBuffer or 0,
            _defaultProvider=self.serviceProvider,
            _defaultMsOptions=self.msOptions,
        )

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
