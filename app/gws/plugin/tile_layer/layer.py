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
    serviceProvider: provider.Object

    canRenderInClient = True

    def configure(self):
        self.configure_layer()

    def create_grabber(self, opts):
        return grabber.Object(opts, serviceProvider=self.serviceProvider)

    def configure_provider(self):
        return gws.config.util.configure_service_provider_for(self, provider.Object)

    def configure_extent(self):
        if super().configure_extent():
            return True
        grid = self.serviceProvider.grid
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
            return gws.u.merge(p, type='xyz', url=self.serviceProvider.url)
        return p
