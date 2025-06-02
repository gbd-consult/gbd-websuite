"""WFS tree layer."""

import gws
import gws.base.layer
import gws.config.util

from . import provider

gws.ext.new.layer('wfs')


class Config(gws.base.layer.Config, gws.base.layer.tree.Config):
    """WFS tree layer configuration."""

    provider: provider.Config
    """WFS provider."""


class Object(gws.base.layer.group.Object):
    serviceProvider: provider.Object

    def configure_group(self):
        self.configure_provider()

        def leaf_layer_maker(source_layers):
            return dict(
                type='wfsflat',
                _defaultProvider=self.serviceProvider,
                _defaultSourceLayers=source_layers,
            )

        configs = gws.base.layer.tree.layer_configs_from_layer(
            self,
            self.serviceProvider.sourceLayers,
            leaf_layer_maker,
        )

        self.configure_group_layers(configs)

    def configure_provider(self):
        return gws.config.util.configure_service_provider_for(self, provider.Object)

    def configure_metadata(self):
        if super().configure_metadata():
            return True
        self.metadata = self.serviceProvider.metadata
        return True
