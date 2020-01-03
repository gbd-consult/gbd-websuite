import gws.common.search.provider
import gws.gis.util
import gws.types as t

from . import provider, util


class Config(gws.common.search.provider.Config, util.WmsConfig):
    pass


class Object(gws.common.search.provider.Object):
    def __init__(self):
        super().__init__()

        self.provider: provider.Object = None
        self.source_layers: t.List[t.SourceLayer] = []
        self.url = ''

    def configure(self):
        super().configure()

        self.with_geometry = 'required'
        self.with_keyword = 'no'

        layer = self.var('layer')
        if layer:
            self.provider = layer.provider
            self.source_layers = self.var('source_layers')
            self.url = layer.url
        else:
            util.configure_wms_for(self, queryable_only=True)

    def can_run(self, args):
        return (
                super().can_run(args)
                and self.provider.operation('GetFeatureInfo')
                and args.shapes
                and args.shapes[0].type == 'Point')

    def run(self, layer: t.ILayer, args: t.SearchArgs) -> t.List[t.IFeature]:
        args.source_layer_names = [sl.name for sl in self.source_layers]
        return self.provider.find_features(args)
