"""GeoJSON Finder."""

from typing import Optional

import gws
import gws.base.search
import gws.config.util

from . import provider

gws.ext.new.finder('geojson')


class Config(gws.base.search.finder.Config):
    """GeoJSON Finder configuration."""
    
    provider: Optional[provider.Config]
    """Provider configuration."""


class Object(gws.base.search.finder.Object):
    supportsGeometrySearch = True
    serviceProvider: provider.Object

    def configure(self):
        self.configure_provider()
        self.configure_models()
        self.configure_templates()

    def configure_provider(self):
        return gws.config.util.configure_service_provider_for(self, provider.Object)

    def configure_models(self):
        return gws.config.util.configure_models_for(self, with_default=True)

    def create_model(self, cfg):
        return self.create_child(
            gws.ext.object.model,
            cfg,
            type=self.extType,
            _defaultProvider=self.serviceProvider,
        )
