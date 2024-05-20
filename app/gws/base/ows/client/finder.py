import gws
import gws.base.model
import gws.base.search
import gws.config.util
import gws.gis.source


class Object(gws.base.search.finder.Object):
    """Generic OWS Finder."""

    supportsGeometrySearch = True
    provider: gws.OwsProvider
    sourceLayers: list[gws.SourceLayer]

    def configure(self):
        self.configure_provider()
        self.configure_sources()
        self.configure_models()
        self.configure_templates()

    def configure_provider(self):
        pass

    def configure_sources(self):
        self.configure_source_layers()
        if not self.sourceLayers:
            raise gws.Error(f'no queryable layers found in {self.provider}')

    def configure_source_layers(self):
        return gws.config.util.configure_source_layers_for(self, self.provider.sourceLayers, is_queryable=True)

    def configure_models(self):
        return gws.config.util.configure_models_for(self, with_default=True)

    def create_model(self, cfg):
        return self.create_child(
            gws.ext.object.model,
            cfg,
            type=self.extType,
            _defaultProvider=self.provider,
            _defaultSourceLayers=self.sourceLayers
        )
