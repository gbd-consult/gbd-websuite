"""GeoJson model."""

from typing import Optional

import gws.base.model
import gws.config.util
import gws.gis.source

from . import provider

gws.ext.new.model('geojson')


# @TODO generally, vector models should be converted to sqlite/gpkg in order to support search

class Config(gws.base.model.Config):
    provider: Optional[provider.Config]
    """GeoJson provider"""


class Object(gws.base.model.dynamic_model.Object):
    """GeoJson Model."""

    serviceProvider: provider.Object

    def configure(self):
        self.uidName = 'id'
        self.geometryName = 'geometry'
        self.configure_model()

    def configure_provider(self):
        return gws.config.util.configure_service_provider_for(self, provider.Object)

    def find_features(self, search, user, **kwargs):
        return [
            self.feature_from_record(rec, user)
            for rec in self.serviceProvider.get_records()
        ]
