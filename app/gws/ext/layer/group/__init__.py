import gws.types as t
import gws.gis.layer
import gws.gis.shape


class Config(gws.gis.layer.BaseConfig):
    """Group layer"""

    layers: t.List[t.ext.layer.Config]  #: layers in this group


class Object(gws.gis.layer.Base):
    def configure(self):
        super().configure()

        self.layers = gws.gis.layer.add_layers_to_object(self, self.var('layers'))
        self.configure_extent(gws.gis.shape.merge_extents(la.extent for la in self.layers))

    def ows_enabled(self, service):
        return (super().ows_enabled(service)
                and any(la.ows_enabled(service) for la in self.layers))

    @property
    def props(self):
        return gws.extend(super().props, {
            'type': 'group',
            'layers': self.layers,
        })
