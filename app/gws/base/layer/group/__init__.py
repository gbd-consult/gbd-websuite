"""Group layer."""

import gws.base.layer
import gws.gis.extent
import gws.gis.legend

import gws.types as t


class Config(gws.base.layer.Config):
    """Group layer"""

    layers: t.List[t.ext.layer.Config]  #: layers in this group


class Object(gws.base.layer.Group):
    def configure(self):
        super().configure()

        self.layers = gws.base.layer.util.add_layers_to_object(self, self.var('layers'))
