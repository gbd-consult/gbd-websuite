"""QGIS Server-based Model."""

from typing import Optional

import gws
import gws.base.model
import gws.base.ows.client
import gws.config.util
import gws.gis.source

from . import provider

gws.ext.new.model('qgis')


class Config(gws.base.model.Config):
    provider: Optional[provider.Config]
    """WMS provider"""
    sourceLayers: Optional[gws.gis.source.LayerFilter]
    """Source layers to search for."""


class Object(gws.base.ows.client.model.Object):
    serviceProvider: provider.Object

    def configure_provider(self):
        return gws.config.util.configure_service_provider_for(self, provider.Object)
