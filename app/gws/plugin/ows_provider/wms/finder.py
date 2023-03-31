import gws
import gws.base.model
import gws.base.search
import gws.gis.source
import gws.types as t

from . import provider

gws.ext.new.finder('wms')


class Config(gws.base.search.finder.Config, provider.Config):
    sourceLayers: t.Optional[gws.gis.source.LayerFilter]
    """Source layers to search for."""


class Object(gws.base.search.finder.Object):
    supportsGeometry = True

    provider: provider.Object
    sourceLayers: list[gws.SourceLayer]

    def configure(self):
        self.configure_provider()
        self.configure_sources()
        self.configure_models()
        self.configure_templates()

    def configure_provider(self):
        if self.cfg('_provider'):
            self.provider = self.cfg('_provider')
            return True
        self.provider = self.root.create_shared(provider.Object, self.config)
        return True

    def configure_sources(self):
        if self.cfg('_sourceLayers'):
            self.sourceLayers = self.cfg('_sourceLayers')
            return True
        self.sourceLayers = gws.gis.source.filter_layers(
            self.provider.sourceLayers,
            gws.gis.source.LayerFilter(self.cfg('sourceLayers'), isQueryable=True))
        if not self.sourceLayers:
            raise gws.Error(f'no queryable layers found in {self.provider.url!r}')
        return True

    def can_run(self, search, user):
        return (
                super().can_run(search, user)
                and bool(search.shape)
                and search.shape.type == gws.GeometryType.point)
