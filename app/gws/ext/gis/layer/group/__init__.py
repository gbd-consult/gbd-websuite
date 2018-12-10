import gws.types as t
import gws.gis.layer


class Config(gws.gis.layer.BaseConfig):
    """group layer"""

    #: layers in this group
    layers: t.List[t.ext.gis.layer.Config]


class LayerProps(gws.gis.layer.BaseProps):
    layers: t.List[t.ext.gis.layer.LayerProps]


class Object(gws.gis.layer.Base):
    def __init__(self):
        super().__init__()
        self.layers: t.List[t.LayerObject] = []

    def configure(self):
        super().configure()
        for p in self.var('layers'):
            try:
                self.layers.append(self.add_child('gws.ext.gis.layer', p))
            except Exception:
                gws.log.exception()

    @property
    def props(self):
        return gws.extend(super().props, {
            'layers': self.layers,
        })
