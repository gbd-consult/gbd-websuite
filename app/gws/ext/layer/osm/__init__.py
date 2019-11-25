import gws.types as t
import gws.gis.layer


class Config(gws.gis.layer.BaseConfig):
    pass


class Object(gws.gis.layer.Base):
    @property
    def props(self):
        return gws.extend(super().props, {
            'type': 'osm',
        })

    def ows_enabled(self, service):
        return False
