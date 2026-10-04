"""WMS tree layer."""

import gws
import gws.base.layer
import gws.config.util

from . import provider


@gws.ext.config.layer('wms')
class Config(gws.base.layer.Config, gws.base.layer.tree.Config):
    """Layer group that mirrors the layer tree of a WMS service."""

    provider: provider.Config
    """WMS service the layers are loaded from."""


@gws.ext.object.layer('wms')
class Object(gws.base.layer.group.Object):
    """Group layer that mirrors the layer tree of a WMS service, with ``wmsflat`` layers as leaves."""

    provider: provider.Object
    """WMS service provider."""

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
