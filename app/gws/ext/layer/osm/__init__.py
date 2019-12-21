import gws.common.layer


class Config(gws.common.layer.BaseConfig):
    pass


class Object(gws.common.layer.Base):
    @property
    def props(self):
        return super().props.extend({
            'type': 'osm',
        })

    def ows_enabled(self, service):
        return False
