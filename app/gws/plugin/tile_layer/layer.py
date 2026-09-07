"""Tile layer."""

from typing import Optional

import gws
import gws.base.layer
import gws.config.util
import gws.lib.bounds
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

    def configure(self):
        self.configure_layer()

    def configure_grabber(self):
        self.grabber = self.create_grabber()
        return True

    def create_grabber(self):
        return self.root.create_shared(
            grabber.Object,
            crs=self.mapCrs.srid,
            extent=self.bounds.extent,
            imageFormat=self.imageFormat,
            _defaultCache=self.cache,
            _defaultProvider=self.serviceProvider,
        )

    def configure_provider(self):
        return gws.config.util.configure_service_provider_for(self, provider.Object)

    def configure_bounds(self):
        if super().configure_bounds():
            return True
        grid = self.serviceProvider.grid
        self.bounds = gws.lib.bounds.transform(
            gws.Bounds(crs=grid.crs, extent=grid.extent),
            self.mapCrs,
        )
        return True

    ##

    def props(self, user):
        p = super().props(user)
        if self.displayMode == gws.LayerDisplayMode.client:
            return gws.u.merge(p, type='xyz', url=self.serviceProvider.url)
        return p
