"""QGIS Tree layer."""

import gws
import gws.base.layer
import gws.config.util
import gws.types as t

from . import provider

gws.ext.new.layer('qgis')


class Config(gws.base.layer.Config, gws.base.layer.tree.Config):
    provider: t.Optional[provider.Config]
    """qgis provider"""


class Object(gws.base.layer.group.Object):
    provider: provider.Object

    def configure_group(self):
        self.provider = provider.get_for(self)

        configs = gws.base.layer.tree.layer_configs_from_layer(
            self,
            self.provider.sourceLayers,
            self.provider.leaf_config,
        )

        self.configure_group_layers(configs)

    def configure_metadata(self):
        if super().configure_metadata():
            return True
        self.metadata = self.provider.metadata
        return True
