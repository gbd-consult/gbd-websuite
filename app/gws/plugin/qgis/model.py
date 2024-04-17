"""QGIS Server-based Model."""

from typing import Optional

import gws
import gws.base.model
import gws.base.ows.client
import gws.gis.source

from . import provider

gws.ext.new.model('qgis')


class Config(gws.base.model.Config):
    provider: Optional[provider.Config]
    """WMS provider"""
    sourceLayers: Optional[gws.gis.source.LayerFilter]
    """Source layers to search for."""


class Object(gws.base.ows.client.model.Object):
    provider: provider.Object

    def configure_provider(self):
        self.provider = provider.get_for(self)
