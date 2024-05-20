"""QGIS Tree layer."""

from typing import Optional

import gws
import gws.base.layer
import gws.config.util

from . import provider

gws.ext.new.layer('qgis')


class Config(gws.base.layer.Config, gws.base.layer.tree.Config):
    provider: Optional[provider.Config]
    """qgis provider"""


class Object(gws.base.layer.group.Object):
    serviceProvider: provider.Object

    def configure_group(self):
        gws.config.util.configure_service_provider_for(self, provider.Object)

        configs = gws.base.layer.tree.layer_configs_from_layer(
            self,
            self.serviceProvider.sourceLayers,
            self.serviceProvider.leaf_config,
        )

        self.configure_group_layers(configs)

    def configure_metadata(self):
        if super().configure_metadata():
            return True
        self.metadata = self.serviceProvider.metadata
        return True
