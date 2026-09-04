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
        cache = self.cache or gws.LayerCache(maxAge=0, maxLevel=0)
        uid = 'grabber_' + gws.u.sha256([
            self.serviceProvider.uid,
            self.mapCrs.srid,
            vars(self.imageFormat),
            list(self.bounds.extent),
            cache.maxAge or 0,
            cache.maxLevel or 0,
            cache.requestTiles or 0,
        ])
        return self.root.create_shared(
            grabber.Object,
            crs=self.mapCrs.srid,
            extent=self.bounds.extent,
            imageFormat=self.imageFormat,
            blockSize=cache.requestTiles or 1,
            cacheMaxAge=cache.maxAge or 0,
            cacheMaxLevel=cache.maxLevel or 0,
            cacheUid=uid,
            _defaultProvider=self.serviceProvider,
        )

    def configure_provider(self):
        return gws.config.util.configure_service_provider_for(self, provider.Object)

    #
    # reprojecting the world doesn't make sense, just use the map extent here
    # see also ows_provider/wmts
    #
    # def configure_bounds(self):
    #     if super().configure_bounds():
    #         return True
    #     self.bounds = gws.lib.bounds.transform(self.serviceProvider.grid.bounds, self.mapCrs)
    #     return True

    ##

    def props(self, user):
        p = super().props(user)
        if self.displayMode == gws.LayerDisplayMode.client:
            return gws.u.merge(p, type='xyz', url=self.serviceProvider.url)
        return p
