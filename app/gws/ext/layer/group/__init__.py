"""Group layer."""

import gws.common.layer
import gws.gis.extent
import gws.gis.legend

import gws.types as t


class Config(gws.common.layer.Config):
    """Group layer"""

    layers: t.List[t.ext.layer.Config]  #: layers in this group


class Object(gws.common.layer.Group):
    def configure(self):
        self.layers = gws.common.layer.util.add_layers_to_object(self, self.var('layers'))
