"""QGIS Tree layer."""

import gws
import gws.base.layer
import gws.types as t

from . import provider

gws.ext.new.layer('qgis')


class Config(gws.base.layer.Config, gws.base.layer.tree.Config):
    provider: t.Optional[provider.Config]
    """qgis provider"""


class Object(gws.base.layer.Object):
    provider: provider.Object

    def configure(self):
        self.provider = provider.configure_for(self)

        configs = gws.base.layer.tree.layer_configs_from_layer(
            self,
            self.provider.sourceLayers,
            self.provider.leaf_layer_config,
        )

        self.configure_group(configs)

        if not self.configure_metadata():
            self.metadata = self.provider.metadata

    def props(self, user):
        return gws.merge(super().props(user), type='group')
