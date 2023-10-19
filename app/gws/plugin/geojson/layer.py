"""GeoJSON layer"""

import gws
import gws.base.layer
import gws.base.shape
import gws.config.util
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
    features: list[gws.IFeature]

    def configure(self):
        self.configure_layer()
        for fd in self.provider.feature_data():
            if fd.shape:
                self.geometryType = fd.shape.type
                self.geometryCrs = fd.shape.crs
                break

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
                self.bounds = gws.gis.bounds.transform(gws.gis.bounds.union(bs), self.mapCrs)
                return True

    def configure_models(self):
        return gws.config.util.configure_models(self, with_default=True)

    def create_model(self, cfg):
        return self.create_child(
            gws.ext.object.model,
            cfg,
            type=self.extType,
            _defaultProvider=self.provider
        )
