"""GeoJSON layer"""

import gws
import gws.base.layer
import gws.base.shape
import gws.config.util
import gws.lib.bounds
import gws.lib.crs
import gws.lib.jsonx

from . import provider

gws.ext.new.layer('geojson')


class Config(gws.base.layer.Config):
    """GeoJson layer"""

    provider: provider.Config
    """geojson provider"""


class Object(gws.base.layer.vector.Object):
    path: str
    serviceProvider: provider.Object
    features: list[gws.Feature]

    def configure(self):
        self.configure_layer()
        for rec in self.serviceProvider.get_records():
            if rec.shape:
                self.geometryType = rec.shape.type
                self.geometryCrs = rec.shape.crs
                break

    def configure_provider(self):
        return gws.config.util.configure_service_provider_for(self, provider.Object)

    def configure_bounds(self):
        if super().configure_bounds():
            return True
        recs = self.serviceProvider.get_records()
        if recs:
            bs = [rec.shape.bounds() for rec in recs if rec.shape]
            if bs:
                self.bounds = gws.lib.bounds.transform(gws.lib.bounds.union(bs), self.mapCrs)
                return True

    def configure_models(self):
        return gws.config.util.configure_models_for(self, with_default=True)

    def create_model(self, cfg):
        return self.create_child(
            gws.ext.object.model,
            cfg,
            type=self.extType,
            _defaultProvider=self.serviceProvider
        )
