import gws
import gws.base.feature

from . import core


@gws.ext.config.model('default')
class Config(core.Config):
    pass


@gws.ext.object.model('default')
class Object(core.Object):
    def feature_from_source(self, sf):
        uid = sf.uid or gws.sha256(sf.attributes)
        f = gws.base.feature.Feature(self, self.uid + self.UID_DELIMITER + uid)
        f.attributes = sf.attributes
        f.layerName = sf.layerName
        f.shape = sf.shape
        return f
