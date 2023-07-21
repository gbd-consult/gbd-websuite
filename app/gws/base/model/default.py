import gws
from . import core

gws.ext.new.model('default')


class Config(core.Config):
    pass


class Object(core.Object):
    def configure(self):
        self.keyName = 'uid'
        self.geometryName = 'geometry'
        self.loadingStrategy = gws.FeatureLoadingStrategy.all
        self.configure_fields()


def get_default(root: gws.IRoot) -> gws.IModel:
    return root.create_shared(Object, uid='gws.base.model.default.Object')
