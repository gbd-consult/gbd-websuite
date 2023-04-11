"""WMS tree layer."""

import gws
import gws.base.layer
import gws.types as t

from . import provider

gws.ext.new.layer('wms')


class Config(gws.base.layer.Config, gws.base.layer.tree.Config):
    provider: provider.Config
    """WMS provider"""


class Object(gws.base.layer.Object):
    provider: provider.Object

    def configure(self):
        self.provider = provider.get_for(self)

        def leaf_layer_maker(source_layers):
            return dict(
                type='wmsflat',
                _defaultProvider=self.provider,
                _defaultSourceLayers=source_layers,
            )

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
