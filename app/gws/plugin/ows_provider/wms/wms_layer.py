"""WMS tree layer."""

import gws
import gws.base.layer
import gws.types as t

from . import provider


@gws.ext.config.layer('wms')
class Config(gws.base.layer.Config, gws.base.layer.tree.Config, provider.Config):
    pass


@gws.ext.object.layer('wms')
class Object(gws.base.layer.Object):
    provider: provider.Object

    def configure(self):
        self.provider = self.root.create_shared(provider.Object, self.config)

        def leaf_layer_maker(source_layers):
            return {
                'type': 'wmsflat',
                'url': self.provider.url,
                '_provider': self.provider,
                '_sourceLayers': source_layers,
            }

        configs = gws.base.layer.tree.layer_configs_from_layer(
            self,
            self.provider.sourceLayers,
            leaf_layer_maker,
        )

        self.configure_group(configs)

        if not self.configure_metadata():
            self.metadata = self.provider.metadata

    def props(self, user):
        return gws.merge(super().props(user), type='group')
