"""Internal QGIS/WMS search provider."""

import gws.common.search.provider
import gws.gis.source
import gws.types as t

from . import provider


class Config(gws.common.search.provider.Config):
    """Qgis/WMS automatic search provider"""

    path: t.FilePath  #: project path
    sourceLayers: t.Optional[gws.gis.source.LayerFilterConfig]  #: source layers to use


class Object(gws.common.search.provider.Object):
    def __init__(self):
        super().__init__()

        # so that it can be found as a 'search' object
        self.klass = 'gws.ext.search.provider.qgiswms'

        self.provider: provider.Object = None
        self.source_layers: t.List[t.SourceLayer] = []

    def configure(self):
        super().configure()

        self.with_geometry = 'required'
        self.with_keyword = 'no'

        layer = self.var('layer')
        if layer:
            self.provider = layer.provider
            self.source_layers = self.var('source_layers')
        else:
            self.provider = provider.create_shared(self, self.config)
            self.source_layers = gws.gis.source.filter_layers(
                self.provider.source_layers,
                self.var('sourceLayers'),
                queryable_only=True)



    def can_run(self, args):
        return (
                super().can_run(args)
                and args.shapes
                and args.shapes[0].type == 'Point')

    def run(self, layer: t.ILayer, args: t.SearchArgs) -> t.List[t.IFeature]:
        args.source_layer_names = [sl.name for sl in self.source_layers]
        return self.provider.find_features(args)
