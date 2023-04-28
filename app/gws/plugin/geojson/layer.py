"""GeoJSON layer"""

import gws
import gws.base.layer.vector
import gws.base.shape
import gws.gis.bounds
import gws.gis.crs
import gws.lib.jsonx

from . import provider

gws.ext.new.layer('geojson')


class Config(gws.base.layer.Config):
    """GeoJson layer"""

    provider: provider.Config
    """geojson provider"""


class Object(gws.base.layer.vector.Object):
    path: str
    provider: provider.Object
    source_crs: gws.ICrs
    features: list[gws.IFeature]

    def configure(self):
        self.configure_provider()
        self.configure_models()
        self.configure_bounds()

    def configure_provider(self):
        self.provider = provider.get_for(self)
        return True

    def configure_bounds(self):
        if super().configure_bounds():
            return True
        fds = self.provider.feature_data()
        if fds:
            bs = [fd.shape.bounds() for fd in fds if fd.shape]
            if bs:
                self.bounds = gws.gis.bounds.union(bs)
                return True

    def configure_models(self):
        if super().configure_models():
            return True
        self.models.append(self.configure_model({}))
        return True

    def configure_model(self, cfg):
        return self.create_child(gws.ext.object.model, cfg, type=self.extType, _defaultProvider=self.provider)
