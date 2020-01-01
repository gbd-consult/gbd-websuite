import gws
import gws.common.layer
import gws.gis.util
import gws.gis.proj

import gws.types as t

from . import provider, util


class Config(gws.common.layer.VectorConfig, util.WfsServiceConfig):
    pass


class Object(gws.common.layer.Vector):
    def __init__(self):
        super().__init__()

        self.provider: provider.Object = None
        self.source_layers: t.List[t.SourceLayer] = []
        self.url = ''

    def configure(self):
        super().configure()

        util.configure_wfs_for(self)

        if not self.source_layers:
            raise gws.Error(f'no source layers found for {self.uid!r}')

    @property
    def description(self):
        context = {
            'layer': self,
            'service': self.provider.meta,
            'sub_layers': self.source_layers
        }
        return self.description_template.render(context).content

    def get_features(self, bounds, limit=0):
        fs = self.provider.find_features(t.SearchArgs(
            bounds=bounds,
            limit=limit,
            source_layer_names=[sl.name for sl in self.source_layers]
        ))
        return [self.connect_feature(f) for f in fs]

    @property
    def default_search_provider(self):
        return self.create_object('gws.ext.search.provider', t.Config(type='wfs', layer=self))
