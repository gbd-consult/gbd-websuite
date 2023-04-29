"""WFS Finder."""

import gws
import gws.base.ows
import gws.base.search
import gws.gis.source
import gws.types as t

from . import provider

gws.ext.new.finder('wfs')


class Config(gws.base.search.finder.Config):
    provider: t.Optional[provider.Config]
    """Provider configuration."""
    sourceLayers: t.Optional[gws.gis.source.LayerFilter]
    """Source layers to search for."""


class Object(gws.base.ows.finder.Object):
    supportsGeometrySearch = True
    provider: provider.Object

    def configure_provider(self):
        self.provider = provider.get_for(self)
