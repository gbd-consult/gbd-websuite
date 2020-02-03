import gws.common.layer


class Config(gws.common.layer.Config):
    pass


class Object(gws.common.layer.Layer):
    @property
    def props(self):
        return super().props.extend({
            'type': 'osm',
        })

    def ows_enabled(self, service):
        return False
