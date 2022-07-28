"""QGIS Tree layer."""

import gws
import gws.base.layer.group
import gws.base.layer.image
import gws.types as t

from . import provider


@gws.ext.config.layer('qgis')
class Config(gws.base.layer.image.Config, provider.Config, gws.base.layer.group.TreeConfig):
    """QGIS Project layer"""
    pass


@gws.ext.object.layer('qgis')
class Object(gws.base.layer.group.Object):
    provider: provider.Object

    def configure_source(self):
        self.provider = self.root.create_object(provider.Object, self.config, shared=True)
        self.configure_layer_tree(self.provider.source_layers, self.provider.leaf_layer_config)
        return True

    def configure_metadata(self):
        if not super().configure_metadata():
            self.set_metadata(self.provider.metadata)
            return True
