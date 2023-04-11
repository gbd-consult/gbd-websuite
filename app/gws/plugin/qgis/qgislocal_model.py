"""QGIS Server-based Model."""

import gws
import gws.base.model
import gws.base.ows
import gws.gis.source
import gws.types as t

from . import provider

gws.ext.new.model('qgislocal')


class Config(gws.base.model.Config):
    provider: t.Optional[provider.Config]
    """WMS provider"""
    sourceLayers: t.Optional[gws.gis.source.LayerFilter]
    """Source layers to search for."""


class Object(gws.base.ows.model.Object):
    provider: provider.Object

    def configure_provider(self):
        self.provider = provider.configure_for(self)
