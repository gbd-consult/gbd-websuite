"""Internal QGIS/WMS search provider."""

import gws
import gws.types as t
import gws.base.search
import gws.lib.source
from . import provider


class Config(gws.base.search.ProviderConfig):
    """Qgis/WMS automatic search provider"""

    path: gws.FilePath  #: project path
    sourceLayers: t.Optional[gws.lib.source.Filter]  #: source layers to use


class Object(gws.base.search.Provider):
    provider: provider.Object
    source_layers: t.List[gws.lib.source.Layer]

    def configure(self):
        self.with_geometry = True

        layer = self.var('layer')
        if layer:
            self.provider = layer.provider
            self.source_layers = self.var('source_layers')
        else:
            self.provider = provider.create_shared(self.root, self.config)
            self.source_layers = gws.lib.source.filter_layers(
                self.provider.source_layers,
                self.var('sourceLayers'),
                queryable_only=True)

        if not self.source_layers:
            raise ValueError(f'{self.uid!r}: no source layers')

    def can_run(self, args):
        return (
                super().can_run(args)
                and args.shapes
                and len(args.shapes) == 1
                and args.shapes[0].type == gws.GeometryType.point)

    def run(self, args: gws.SearchArgs, layer: gws.ILayer = None) -> t.List[gws.IFeature]:
        args.source_layer_names = [sl.name for sl in self.source_layers]
        return self.provider.find_features(args)
