"""Finder for QGIS project layers."""

from typing import Optional

import gws
import gws.base.model
import gws.base.search
import gws.base.ows.client
import gws.config.util
import gws.gis.source

from . import provider


@gws.ext.config.finder('qgis')
class Config(gws.base.search.finder.Config):
    """Search in QGIS project layers via QGIS Server."""

    provider: Optional[provider.Config]
    """QGIS project to search."""
    sourceLayers: Optional[gws.gis.source.LayerFilter]
    """Source layers to search."""


@gws.ext.object.finder('qgis')
class Object(gws.base.ows.client.finder.Object):
    """Finder that queries QGIS project layers with GetFeatureInfo."""

    supportsGeometrySearch = True
    provider: provider.Object
    """QGIS provider."""

    def configure_provider(self):
        return gws.config.util.configure_provider_for(self, provider.Object)

    def can_run(self, search, user):
        return (
                super().can_run(search, user)
                and bool(search.shape)
                and search.shape.type == gws.GeometryType.point)
