"""Generic OWS model."""

import gws
import gws.base.feature
import gws.base.model
import gws.config.util
import gws.lib.crs
import gws.lib.extent
import gws.gis.source


class Object(gws.base.model.default_model.Object):
    """Generic OWS model.

    Read-only model that reads features from the queryable source layers of an
    OWS service provider. Clients cannot create, update or delete features.
    """

    provider: gws.OwsServiceProvider
    """Service provider."""
    sourceLayers: list[gws.SourceLayer]
    """Source layers to read features from."""

    def configure(self):
        self.configure_model()

    def configure_provider(self):
        pass

    def configure_sources(self):
        gws.u.require(self.provider, 'failed to configure service provider')

        self.configure_source_layers()

    def configure_source_layers(self):
        """Select the queryable source layers of the provider, or those given in the configuration.

        Returns:
            Always ``True``.
        """

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
