"""WMS tree layer."""

import gws
import gws.base.layer
import gws.types as t

from . import provider as provider_module


@gws.ext.config.layer('wms')
class Config(gws.base.layer.image.Config, provider_module.Config, gws.base.layer.group.TreeConfig):
    pass


@gws.ext.object.layer('wms')
class Object(gws.base.layer.group.Object):
    provider: provider_module.Object

    def configure_source(self):
        self.provider = self.create_child(provider_module.Object, self.config, shared=True)

        def leaf(source_layers):
            return {
                'type': 'wmsflat',
                '_provider': self.provider,
                '_source_layers': source_layers,
            }
        self.configure_layer_tree(self.provider.source_layers, leaf)

        return True

    def configure_metadata(self):
        if not super().configure_metadata():
            self.set_metadata(self.provider.metadata)
            return True
