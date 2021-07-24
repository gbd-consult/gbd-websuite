import gws
import gws.base.ows
import gws.base.search
import gws.lib.gis
import gws.lib.ows
import gws.types as t
from . import provider


@gws.ext.Config('search.provider.wms')
class Config(gws.base.search.Config, provider.Config):
    sourceLayers: t.Optional[gws.lib.gis.LayerFilter]  #: source layers to use


@gws.ext.Object('search.provider.wms')
class Object(gws.base.search.provider.Object):
    supports_geometry = True

    source_layers: t.List[gws.lib.gis.SourceLayer]
    provider: provider.Object

    def configure(self):
        layer = self.var('layer')

        if layer:
            self.provider = layer.provider
            self.source_layers = self.var('source_layers')
        else:
            self.provider = gws.base.ows.provider.shared_object(provider.Object, self, self.config)
            self.source_layers = gws.lib.gis.filter_layers(
                self.provider.source_layers,
                self.var('sourceLayers'),
                queryable_only=True)

        if not self.source_layers:
            raise gws.Error(f'no source layers in {self.uid!r}')

    def can_run(self, args):
        return (
                super().can_run(args)
                and bool(self.provider.operation('GetFeatureInfo'))
                and bool(args.shapes)
                and len(args.shapes) == 1
                and args.shapes[0].type == gws.GeometryType.point)

    def run(self, args: gws.SearchArgs, layer: gws.ILayer = None) -> t.List[gws.IFeature]:
        args.source_layer_names = [sl.name for sl in self.source_layers]
        return self.provider.find_features(args)
