"""Tile layer."""

from typing import Optional

import gws
import gws.base.layer
import gws.config.util
import gws.lib.bounds
import gws.lib.extent
from . import grabber, provider

gws.ext.new.layer('tile')


class Config(gws.base.layer.Config):
    """Tile layer"""

    provider: provider.Config
    """Tile service provider."""
    display: gws.LayerDisplayMode = gws.LayerDisplayMode.tile
    """Layer display mode."""


class Object(gws.base.layer.image.Object):
    provider: provider.Object

    canRenderInClient = True

    def configure(self):
        self.configure_layer()

    def create_cache_name(self, cache):
        return gws.u.sha256([
            self.provider.cache_hash(),
            vars(self.imageFormat),
            list(self.wgsExtent),
            cache.requestBuffer,
            cache.requestTiles,
        ])[: gws.base.layer.image.CACHE_NAME_LENGTH]

    def create_grabber(self, opts):
        return grabber.Object(opts, provider=self.provider)

    def configure_provider(self):
        return gws.config.util.configure_provider_for(self, provider.Object)

    def configure_extent(self):
        if super().configure_extent():
            return True
        grid = self.provider.grid
        ext = gws.lib.extent.transform_to_wgs(grid.extent, grid.crs)
        if gws.lib.extent.is_valid_wgs(ext):
            self.wgsExtent = grid.crs.clip_wgs_extent(ext) or grid.crs.wgsMaxExtent
        else:
            self.wgsExtent = grid.crs.wgsMaxExtent
        return True

    ##

    def props(self, user):
        p = super().props(user)
        if self.displayMode == gws.LayerDisplayMode.client:
            return gws.u.merge(p, type='xyz', url=self.provider.url)
        return p
