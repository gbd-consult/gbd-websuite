"""OWS model."""

import gws
import gws.base.feature
import gws.base.model
import gws.config.util
import gws.gis.crs
import gws.gis.extent
import gws.gis.source


class Object(gws.base.model.dynamic_model.Object):
    """Generic OWS Model."""

    provider: gws.OwsProvider
    sourceLayers: list[gws.SourceLayer]

    def configure(self):
        self.configure_model()

    def configure_provider(self):
        pass

    def configure_sources(self):
        self.configure_source_layers()

    def configure_source_layers(self):
        return gws.config.util.configure_source_layers_for(self, self.provider.sourceLayers, is_queryable=True)

    def find_features(self, search, mc):
        if not self.sourceLayers:
            return []
        return [
            self.feature_from_record(r, mc)
            for r in self.provider.get_features(search, self.sourceLayers)
        ]

    def props(self, user):
        return gws.u.merge(
            super().props(user),
            canCreate=False,
            canDelete=False,
            canWrite=False,
        )
