import gws
import gws.base.model
import gws.base.search
import gws.gis.source


class Object(gws.base.search.finder.Object):
    """Generic OWS Finder."""

    supportsGeometrySearch = True
    provider: gws.IOwsProvider
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

    def configure_models(self):
        if super().configure_models():
            return True
        self.models.append(self.configure_model(None))
        return True

    def configure_model(self, cfg):
        return self.create_child(gws.ext.object.model, cfg, type=self.extType, _defaultProvider=self.provider, _defaultSourceLayers=self.sourceLayers)

