"""WFS tree layer."""

import gws
import gws.base.layer
import gws.config.util

from . import provider


@gws.ext.config.layer('wfs')
class Config(gws.base.layer.Config, gws.base.layer.tree.Config):
    """Layer group with a sublayer for each WFS feature type."""

    provider: provider.Config
    """WFS service the layers are loaded from."""


@gws.ext.object.layer('wfs')
class Object(gws.base.layer.group.Object):
    provider: provider.Object

    def configure_group(self):
        if super().configure_group():
            return True
        gws.base.layer.tree.configure_group_layers_for(
            self,
            self.provider.sourceLayers,
            self.provider.create_leaf_layer_config,
        )
        return True

    def configure_provider(self):
        return gws.config.util.configure_provider_for(self, provider.Object)

    def configure_metadata(self):
        if super().configure_metadata():
            return True
        self.metadata = self.provider.metadata
        return True
