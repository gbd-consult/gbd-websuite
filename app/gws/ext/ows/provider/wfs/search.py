import gws
import gws.common.search.provider
import gws.gis.util
import gws.gis.proj
import gws.gis.shape

import gws.types as t

from . import provider, util


class Config(gws.common.search.provider.Config, util.WfsServiceConfig):
    pass


class Object(gws.common.search.provider.Object):
    def __init__(self):
        super().__init__()

        self.provider: provider.Object = None
        self.source_layers: t.List[t.SourceLayer] = []
        self.url = ''

        self.with_geometry = 'require'
        self.with_keyword = 'no'

    def configure(self):
        super().configure()

        layer = self.var('layer')
        if layer:
            self.provider = layer.provider
            self.source_layers = layer.source_layers
            self.url = layer.url
        else:
            util.configure_wfs_for(self)

    def can_run(self, args):
        return super().can_run(args) and self.provider.operation('GetFeature')

    def run(self, layer: t.ILayer, args: t.SearchArgs) -> t.List[t.IFeature]:
        args.source_layer_names = [sl.name for sl in self.source_layers]
        return self.provider.find_features(args)
