"""QGIS Tree layer."""

import gws
import gws.base.layer
import gws.types as t

from . import provider


@gws.ext.config.layer('qgis')
class Config(gws.base.layer.Config, gws.base.layer.tree.Config, provider.Config):
    pass


@gws.ext.object.layer('qgis')
class Object(gws.base.layer.Object):
    provider: provider.Object

    def configure(self):
        self.provider = self.root.create_shared(provider.Object, self.config)

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
