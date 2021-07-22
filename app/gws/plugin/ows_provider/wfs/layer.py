import gws
import gws.types as t
import gws.base.layer
import gws.lib.ows
import gws.lib.proj
import gws.lib.gis
import gws.lib.gis
from . import provider, util


class Config(gws.base.layer.VectorConfig, util.WfsServiceConfig):
    pass


class Object(gws.base.layer.Vector):
    def configure(self):
        

        self.url: str = self.var('url')
        self.provider: provider.Object = gws.lib.ows.shared_provider(provider.Object, self, self.config)

        self.metadata = self.configure_metadata(self.provider.metadata)
        self.title = self.metadata.title

        self.source_layers: t.List[gws.SourceLayer] = gws.lib.gis.filter_layers(
            self.provider.source_layers,
            self.var('sourceLayers'))
        if not self.source_layers:
            raise gws.Error(f'no source layers found for {self.uid!r}')

    @property
    def description(self):
        context = {
            'layer': self,
            'service': self.provider.metadata,
            'sub_layers': self.source_layers
        }
        return self.description_template.render(context).content

    @property
    def default_search_provider(self):
        return self.root.create_object('gws.ext.search.provider.wfs', gws.Config(
            uid=self.uid + '.default_search',
            layer=self))

    def get_features(self, bounds, limit=0):
        fs = self.provider.find_features(gws.SearchArgs(
            bounds=bounds,
            limit=limit,
            source_layer_names=[sl.name for sl in self.source_layers]
        ))
        return [self.connect_feature(f) for f in fs]
