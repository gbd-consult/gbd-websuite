import gws.types as t
import gws.gis.layer


class Config(gws.gis.layer.BaseConfig):
    """Group layer"""

    layers: t.List[t.ext.layer.Config]  #: layers in this group


class Object(gws.gis.layer.Base):
    def configure(self):
        super().configure()
        self.layers = gws.gis.layer.add_layers_to_object(self, self.var('layers'))

    @property
    def props(self):
        return gws.extend(super().props, {
            'type': 'group',
            'layers': self.layers,
        })
