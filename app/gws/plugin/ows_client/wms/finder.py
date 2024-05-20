"""WMS Finder."""

from typing import Optional

import gws
import gws.base.ows.client
import gws.config.util
import gws.base.search
import gws.gis.source

from . import provider

gws.ext.new.finder('wms')


class Config(gws.base.search.finder.Config):
    provider: Optional[provider.Config]
    """Provider configuration."""
    sourceLayers: Optional[gws.gis.source.LayerFilter]
    """Source layers to search for."""


class Object(gws.base.ows.client.finder.Object):
    supportsGeometrySearch = True
    serviceProvider: provider.Object

    def configure_provider(self):
        return gws.config.util.configure_service_provider_for(self, provider.Object)

    def can_run(self, search, user):
        return (
                super().can_run(search, user)
                and bool(search.shape)
                and search.shape.type == gws.GeometryType.point)
