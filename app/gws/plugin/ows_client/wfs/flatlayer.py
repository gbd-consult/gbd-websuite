from typing import Optional

import gws
import gws.base.layer
import gws.base.legend
import gws.base.model
import gws.base.search
import gws.base.template
import gws.config.util
import gws.lib.metadata
import gws.lib.crs
import gws.gis.source
import gws.gis.zoom
import gws.lib.bounds
import gws.lib.extent


from . import provider

gws.ext.new.layer('wfsflat')


class Config(gws.base.layer.Config):
    """Flat WFS layer."""

    provider: Optional[provider.Config]
    """WFS provider."""
    sourceLayers: Optional[gws.gis.source.LayerFilter]
    """Source layers to use."""


class Object(gws.base.layer.vector.Object):
    serviceProvider: provider.Object
    sourceLayers: list[gws.SourceLayer]
    sourceCrs: gws.Crs

    def configure(self):
        self.configure_layer()
        if len(self.sourceLayers) != 1:
            raise gws.Error(f'wfsflat requires a single source layer')

    def configure_provider(self):
        return gws.config.util.configure_service_provider_for(self, provider.Object)

    def configure_sources(self):
        if super().configure_sources():
            return True
        self.configure_source_layers()
        return True

    def configure_source_layers(self):
        return gws.config.util.configure_source_layers_for(self, self.serviceProvider.sourceLayers)

    def configure_models(self):
        return gws.config.util.configure_models_for(self, with_default=True)

    def create_model(self, cfg):
        return self.create_child(
            gws.ext.object.model,
            cfg,
            type='wfs',
            _defaultProvider=self.serviceProvider,
            _defaultSourceLayers=self.sourceLayers
        )

    def configure_bounds(self):
        if super().configure_bounds():
            return True
        self.bounds = gws.gis.source.combined_bounds(self.sourceLayers, self.mapCrs) or self.mapCrs.bounds
        return True

    def configure_metadata(self):
        if super().configure_metadata():
            return True
        if len(self.sourceLayers) == 1:
            self.metadata = self.sourceLayers[0].metadata
            return True

    def configure_search(self):
        if super().configure_search():
            return True
        self.finders.append(self.create_finder(None))
        return True

    def create_finder(self, cfg):
        return self.create_child(
            gws.ext.object.finder,
            cfg,
            type='wfs',
            _defaultProvider=self.serviceProvider,
            _defaultSourceLayers=self.sourceLayers
        )
