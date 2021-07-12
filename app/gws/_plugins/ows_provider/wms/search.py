import gws
import gws.types as t
import gws.base.search.provider
import gws.lib.ows
import gws.lib.source
import gws.lib.gisutil
from . import provider


class Config(gws.base.search.provider.Config, provider.Config):
    sourceLayers: t.Optional[gws.lib.source.LayerFilter]  #: source layers to use


class Object(gws.base.search.provider.Object):
    def configure(self):
        

        self.capabilties = gws.base.search.provider.CAPS_GEOMETRY

        layer = self.var('layer')
        if layer:
            self.provider: provider.Object = layer.provider
            self.source_layers = self.var('source_layers')
        else:
            self.provider: provider.Object = gws.lib.ows.shared_provider(provider.Object, self, self.config)
            self.source_layers = gws.lib.source.filter_layers(
                self.provider.source_layers,
                self.var('sourceLayers'),
                queryable_only=True
            )
        if not self.source_layers:
            gws.log.warn(f'{self.uid!r}: no source layers')
            self.active = False

    def can_run(self, args):
        return (
                super().can_run(args)
                and self.provider.operation('GetFeatureInfo')
                and args.shapes
                and len(args.shapes) == 1
                and args.shapes[0].type == gws.GeometryType.point)

    def run(self, layer: gws.ILayer, args: gws.SearchArgs) -> t.List[gws.IFeature]:
        args.source_layer_names = [sl.name for sl in self.source_layers]
        return self.provider.find_features(args)
