import gws.common.search.provider
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

    def configure(self):
        super().configure()

        self.with_geometry = gws.common.search.provider.ParameterUsage.required
        self.with_keyword = gws.common.search.provider.ParameterUsage.forbidden

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
        args.tolerance = args.tolerance or self.tolerance
        return self.provider.find_features(args)
