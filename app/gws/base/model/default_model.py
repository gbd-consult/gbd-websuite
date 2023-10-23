import gws
from . import core

gws.ext.new.model('default')


class Config(core.Config):
    pass


class Object(core.Object):
    def configure(self):
        self.uidName = 'uid'
        self.geometryName = 'geometry'
        self.loadingStrategy = gws.FeatureLoadingStrategy.all
