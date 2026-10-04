"""The ``geojson`` model."""

from typing import Optional

import gws.base.model
import gws.config.util
import gws.gis.source

from . import provider


# @TODO generally, vector models should be converted to sqlite/gpkg in order to support search


@gws.ext.config.model('geojson')
class Config(gws.base.model.Config):
    """Model for features from a GeoJSON file."""

    provider: Optional[provider.Config]
    """GeoJSON file with the model features."""


@gws.ext.object.model('geojson')
class Object(gws.base.model.default_model.Object):
    """GeoJSON model."""

    provider: provider.Object
    """The GeoJSON provider."""

    def configure(self):
        self.uidName = 'id'
        self.geometryName = 'geometry'
        self.configure_model()

    def configure_provider(self):
        return gws.config.util.configure_provider_for(self, provider.Object)

    def find_features(self, search, user, **kwargs):
        # fmt: off
        return [
            self.feature_from_record(rec, user) 
            for rec in self.provider.get_records(search)
        ]
        # fmt: on
