"""Internal QGIS/WMS search provider."""

import gws.common.search.provider
import gws.gis.source
import gws.types as t

from . import provider


class Config(gws.common.search.provider.Config):
    """Qgis/WMS automatic search provider"""

    path: t.FilePath  #: project path
    sourceLayers: t.Optional[gws.gis.source.LayerFilter]  #: source layers to use


class Object(gws.common.search.provider.Object):
    def configure(self):
        super().configure()

        self.with_geometry = gws.common.search.provider.ParameterUsage.required
        self.with_keyword = gws.common.search.provider.ParameterUsage.forbidden

        layer = self.var('layer')
        if layer:
            self.provider: provider.Object = layer.provider
            self.source_layers: t.List[t.SourceLayer] = self.var('source_layers')
        else:
            self.provider: provider.Object = provider.create_shared(self, self.config)
            self.source_layers: t.List[t.SourceLayer] = gws.gis.source.filter_layers(
                self.provider.source_layers,
                self.var('sourceLayers'),
                queryable_only=True)
        if not self.source_layers:
            gws.log.warn(f'{self.uid!r}: no source layers')
            self.active = False

    def can_run(self, args):
        return (
                super().can_run(args)
                and args.shapes
                and len(args.shapes) == 1
                and args.shapes[0].type == t.GeometryType.point)

    def run(self, layer: t.ILayer, args: t.SearchArgs) -> t.List[t.IFeature]:
        args.source_layer_names = [sl.name for sl in self.source_layers]
        return self.provider.find_features(args)
