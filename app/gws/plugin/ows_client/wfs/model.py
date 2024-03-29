"""WFS model."""

import gws
import gws.base.model
import gws.base.ows.client
import gws.gis.source
import gws.types as t

from . import provider

gws.ext.new.model('wfs')


class Config(gws.base.model.Config):
    provider: t.Optional[provider.Config]
    """WFS provider"""
    sourceLayers: t.Optional[gws.gis.source.LayerFilter]
    """Source layers to search for."""


class Object(gws.base.ows.client.model.Object):
    provider: provider.Object

    def configure_provider(self):
        self.provider = provider.get_for(self)
