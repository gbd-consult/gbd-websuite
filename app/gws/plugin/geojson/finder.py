"""The ``geojson`` finder."""

from typing import Optional

import gws
import gws.base.search
import gws.config.util

from . import provider


@gws.ext.config.finder('geojson')
class Config(gws.base.search.finder.Config):
    """Search in features from a GeoJSON file."""
    
    provider: Optional[provider.Config]
    """GeoJSON file to search in."""


@gws.ext.object.finder('geojson')
class Object(gws.base.search.finder.Object):
    """GeoJSON finder."""

    supportsGeometrySearch = True
    provider: provider.Object
    """The GeoJSON provider."""

    def configure(self):
        self.configure_provider()
        self.configure_models()
        self.configure_templates()

    def configure_provider(self):
        """Configure the GeoJSON provider.

        Returns:
            ``True`` if a provider was set.
        """
        return gws.config.util.configure_provider_for(self, provider.Object)

    def configure_models(self):
        return gws.config.util.configure_models_for(self, with_default=True)

    def create_model(self, cfg):
        """Create a GeoJSON model that uses the provider of the finder.

        Args:
            cfg: Model configuration, or ``None`` for the default model.

        Returns:
            The model.
        """
        return self.create_child(
            gws.ext.object.model,
            cfg,
            type=self.extType,
            _defaultProvider=self.provider,
        )
