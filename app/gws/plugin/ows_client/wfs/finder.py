"""WFS Finder."""

from typing import Optional

import gws
import gws.base.ows.client
import gws.base.search
import gws.gis.source

from . import provider

gws.ext.new.finder('wfs')


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
