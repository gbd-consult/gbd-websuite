"""QGIS Server-based Finder."""

import gws
import gws.base.model
import gws.base.search
import gws.base.ows.client
import gws.gis.source
import gws.types as t

from . import provider

gws.ext.new.finder('qgislocal')


class Config(gws.base.search.finder.Config):
    provider: t.Optional[provider.Config]
    """Provider configuration."""
    sourceLayers: t.Optional[gws.gis.source.LayerFilter]
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
