import gws.types as t
import gws.common.layer
import gws.gis.extent


class Config(gws.common.layer.BaseConfig):
    """Group layer"""

    layers: t.List[t.ext.layer.Config]  #: layers in this group


class Object(gws.common.layer.Base):
    def configure(self):
        super().configure()

        self.layers = gws.common.layer.add_layers_to_object(self, self.var('layers'))

    def ows_enabled(self, service):
        return (super().ows_enabled(service)
                and any(la.ows_enabled(service) for la in self.layers))

    @property
    def props(self):
        return super().props.extend({
            'type': 'group',
            'layers': self.layers,
        })
