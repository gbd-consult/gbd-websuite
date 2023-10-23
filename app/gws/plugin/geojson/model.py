"""GeoJson model."""

import gws.base.model
import gws.gis.source
import gws.types as t

from . import provider

gws.ext.new.model('geojson')


# @TODO generally, vector models should be converted to sqlite/gpkg in order to support search

class Config(gws.base.model.Config):
    provider: t.Optional[provider.Config]
    """GeoJson provider"""


class Object(gws.base.model.dynamic_model.Object):
    """GeoJson Model."""

    provider: provider.Object

    def configure(self):
        self.uidName = 'id'
        self.geometryName = 'geometry'

        self.configure_provider()
        self.configure_fields()
        self.configure_uid()
        self.configure_geometry()
        self.configure_templates()

    def configure_provider(self):
        self.provider = provider.get_for(self)

    ##

    def find_features(self, search, user, **kwargs):
        fs = []
        for fd in self.provider.feature_data():
            fs.append(self.feature_from_record(fd, user))
        return fs
