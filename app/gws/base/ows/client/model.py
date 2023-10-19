"""OWS model."""

import gws
import gws.base.feature
import gws.base.model
import gws.config.util
import gws.gis.crs
import gws.gis.extent
import gws.gis.source


class Object(gws.base.model.Object):
    """Generic OWS Model."""

    provider: gws.IOwsProvider
    sourceLayers: list[gws.SourceLayer]

    def configure(self):
        self.uidName = 'uid'
        self.geometryName = 'geometry'

        self.configure_provider()
        self.configure_sources()
        self.configure_fields()
        self.configure_uid()
        self.configure_geometry()
        self.configure_templates()

    def configure_provider(self):
        pass

    def configure_sources(self):
        self.configure_source_layers()

    def configure_source_layers(self):
        return gws.config.util.configure_source_layers(self, self.provider.sourceLayers, is_queryable=True)

    def find_features(self, search, user, **kwargs):
        if not self.sourceLayers:
            return []
        return [
            self.feature_from_record(fd, user, **kwargs)
            for fd in self.provider.get_features(search, self.sourceLayers)
        ]

    def props(self, user):
        return gws.merge(
            super().props(user),
            canCreate=False,
            canDelete=False,
            canWrite=False,
        )