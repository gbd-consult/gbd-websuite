"""WFS model."""

from typing import Optional

import gws
import gws.base.model
import gws.base.ows.client
import gws.config.util
import gws.gis.source

from . import provider


@gws.ext.config.model('wfs')
class Config(gws.base.model.Config):
    """Data model for features from a WFS service."""

    provider: Optional[provider.Config]
    """WFS service the features are loaded from."""
    sourceLayers: Optional[gws.gis.source.LayerFilter]
    """Feature types to query."""


@gws.ext.object.model('wfs')
class Object(gws.base.ows.client.model.Object):
    """Read-only model for features from a WFS service."""

    provider: provider.Object
    """WFS service provider."""

    def configure_provider(self):
        return gws.config.util.configure_provider_for(self, provider.Object)
