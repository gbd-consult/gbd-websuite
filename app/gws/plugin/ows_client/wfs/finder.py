"""WFS Finder."""

from typing import Optional

import gws
import gws.base.ows.client
import gws.base.search
import gws.config.util
import gws.gis.source

from . import provider


@gws.ext.config.finder('wfs')
class Config(gws.base.search.finder.Config):
    """WFS Finder configuration."""
    
    provider: Optional[provider.Config]
    """Provider configuration."""
    sourceLayers: Optional[gws.gis.source.LayerFilter]
    """Source layers to search for."""


@gws.ext.object.finder('wfs')
class Object(gws.base.ows.client.finder.Object):
    supportsGeometrySearch = True
    provider: provider.Object

    def configure_provider(self):
        return gws.config.util.configure_provider_for(self, provider.Object)
