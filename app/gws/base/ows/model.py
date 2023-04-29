"""OWS model."""

import gws
import gws.base.model
import gws.base.feature
import gws.gis.source
import gws.gis.crs
import gws.gis.ows
import gws.gis.extent


class Object(gws.base.model.Object):
    """Generic OWS Model."""

    provider: gws.IOwsProvider
    sourceLayers: list[gws.SourceLayer]

    def configure(self):
        self.keyName = 'uid'
        self.geometryName = 'geometry'

        self.configure_provider()
        self.configure_sources()
        self.configure_fields()
        self.configure_templates()

    def configure_provider(self):
        pass

    def configure_sources(self):
        self.configure_source_layers()

    def configure_source_layers(self):
        p = self.cfg('sourceLayers')
        if p:
            self.sourceLayers = gws.gis.source.filter_layers(self.provider.sourceLayers, p)
            return True
        p = self.cfg('_defaultSourceLayers')
        if p:
            self.sourceLayers = p
            return True
        self.sourceLayers = gws.gis.source.filter_layers(self.provider.sourceLayers, is_queryable=True)
        return True

    def find_features(self, search, user, **kwargs):
        if not self.sourceLayers:
            return []
        return [
            self.feature_from_data(fd, user, **kwargs)
            for fd in self.provider.get_features(search, self.sourceLayers)
        ]
