import gws.base.search.provider
import gws.gis.ows
import gws.gis.source
import gws.gis.util

import gws.types as t

from . import provider


class Config(gws.base.search.provider.Config, provider.Config):
    sourceLayers: t.Optional[gws.gis.source.LayerFilter]  #: source layers to use


class Object(gws.base.search.provider.Object):
    def configure(self):
        super().configure()

        self.capabilties = gws.base.search.provider.CAPS_GEOMETRY

        layer = self.var('layer')
        if layer:
            self.provider: provider.Object = layer.provider
            self.source_layers = self.var('source_layers')
        else:
            self.provider: provider.Object = gws.gis.ows.shared_provider(provider.Object, self, self.config)
            self.source_layers = gws.gis.source.filter_layers(
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
                and args.shapes[0].type == t.GeometryType.point)

    def run(self, layer: t.ILayer, args: t.SearchArgs) -> t.List[t.IFeature]:
        args.source_layer_names = [sl.name for sl in self.source_layers]
        return self.provider.find_features(args)
