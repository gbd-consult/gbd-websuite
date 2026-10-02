"""QGIS Server-based Model."""

from typing import Optional

import gws
import gws.base.model
import gws.base.ows.client
import gws.config.util
import gws.gis.source

from . import provider


@gws.ext.config.model('qgis')
class Config(gws.base.model.Config):
    """Data model for features queried from a QGIS project."""

    provider: Optional[provider.Config]
    """QGIS project the features are queried from."""
    sourceLayers: Optional[gws.gis.source.LayerFilter]
    """Source layers to query."""


@gws.ext.object.model('qgis')
class Object(gws.base.ows.client.model.Object):
    provider: provider.Object

    def configure_provider(self):
        return gws.config.util.configure_provider_for(self, provider.Object)
