"""WMS model."""

from typing import Optional

import gws
import gws.base.model
import gws.base.ows.client
import gws.config.util
import gws.gis.source

from . import provider


@gws.ext.config.model('wms')
class Config(gws.base.model.Config):
    """Data model for features queried from a WMS service."""

    provider: Optional[provider.Config]
    """WMS service the features are queried from."""
    sourceLayers: Optional[gws.gis.source.LayerFilter]
    """Source layers to query."""


@gws.ext.object.model('wms')
class Object(gws.base.ows.client.model.Object):
    """Read-only model for features queried from a WMS service."""

    provider: provider.Object
    """WMS service provider."""

    def configure_provider(self):
        return gws.config.util.configure_provider_for(self, provider.Object)
