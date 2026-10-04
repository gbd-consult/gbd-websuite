"""WFS finder."""

from typing import Optional

import gws
import gws.base.ows.client
import gws.base.search
import gws.config.util
import gws.gis.source

from . import provider


@gws.ext.config.finder('wfs')
class Config(gws.base.search.finder.Config):
    """Search for features in a WFS service."""
    
    provider: Optional[provider.Config]
    """WFS service to search."""
    sourceLayers: Optional[gws.gis.source.LayerFilter]
    """Feature types to search."""


@gws.ext.object.finder('wfs')
class Object(gws.base.ows.client.finder.Object):
    """Finder that searches the feature types of a WFS service by geometry."""

    supportsGeometrySearch = True
    provider: provider.Object
    """WFS service provider."""

    def configure_provider(self):
        return gws.config.util.configure_provider_for(self, provider.Object)
