import gws.types as t
import gws.common.layer


class Config(gws.common.layer.BaseConfig):
    pass


class Object(gws.common.layer.Base):
    @property
    def props(self):
        return gws.extend(super().props, {
            'type': 'osm',
        })

    def ows_enabled(self, service):
        return False
