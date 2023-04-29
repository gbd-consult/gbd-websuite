"""WFS tree layer."""

import gws
import gws.base.layer
import gws.types as t

from . import provider

gws.ext.new.layer('wfs')


class Config(gws.base.layer.Config, gws.base.layer.tree.Config):
    provider: provider.Config
    """WFS provider"""


class Object(gws.base.layer.group.Object):
    provider: provider.Object

    def configure_group(self):
        self.provider = provider.get_for(self)

        def leaf_layer_maker(source_layers):
            return dict(
                type='wfsflat',
                _defaultProvider=self.provider,
                _defaultSourceLayers=source_layers,
            )

        configs = gws.base.layer.tree.layer_configs_from_layer(
            self,
            self.provider.sourceLayers,
            leaf_layer_maker,
        )

        self.configure_group_layers(configs)

    def configure_metadata(self):
        if super().configure_metadata():
            return True
        self.metadata = self.provider.metadata
        return True
