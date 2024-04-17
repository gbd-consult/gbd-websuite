"""QGIS Server-based Finder."""

from typing import Optional

import gws
import gws.base.model
import gws.base.search
import gws.base.ows.client
import gws.config.util
import gws.gis.source

from . import provider

gws.ext.new.finder('qgis')


class Config(gws.base.search.finder.Config):
    provider: Optional[provider.Config]
    """Provider configuration."""
    sourceLayers: Optional[gws.gis.source.LayerFilter]
    """Source layers to search for."""


class Object(gws.base.ows.client.finder.Object):
    supportsGeometrySearch = True
    provider: provider.Object

    def configure_provider(self):
        self.provider = provider.get_for(self)

    def can_run(self, search, user):
        return (
                super().can_run(search, user)
                and bool(search.shape)
                and search.shape.type == gws.GeometryType.point)
