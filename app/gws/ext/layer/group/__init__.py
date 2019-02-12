import gws.types as t
import gws.gis.layer


class Config(gws.gis.layer.BaseConfig):
    """group layer"""

    #: layers in this group
    layers: t.List[t.ext.layer.Config]


class Object(gws.gis.layer.Base):
    def __init__(self):
        super().__init__()
        self.layers: t.List[t.LayerObject] = []

    def configure(self):
        super().configure()
        self.layers = gws.gis.layer.add_layers_to_object(self, self.var('layers'))

    @property
    def props(self):
        return gws.extend(super().props, {
            'type': 'group',
            'layers': self.layers,
        })
