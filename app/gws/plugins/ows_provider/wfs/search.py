import gws.base.search.provider
import gws.gis.shape
import gws.gis.ows
import gws.gis.source

import gws.types as t

from . import provider, util


class Config(gws.base.search.provider.Config, util.WfsServiceConfig):
    pass


class Object(gws.base.search.provider.Object):
    def configure(self):
        super().configure()

        # @TODO support filters
        self.capabilties = gws.base.search.provider.CAPS_GEOMETRY

        layer = self.var('layer')
        if layer:
            self.provider: provider.Object = layer.provider
            self.source_layers = layer.source_layers
            self.url = layer.url
        else:
            self.provider: provider.Object = gws.gis.ows.shared_provider(provider.Object, self, self.config)
            self.source_layers = gws.gis.source.filter_layers(
                self.provider.source_layers,
                self.var('sourceLayers'))
            if not self.source_layers:
                raise gws.Error(f'no source layers found for {self.uid!r}')
            self.url = self.var('url')

    def can_run(self, args):
        return super().can_run(args) and self.provider.operation('GetFeature')

    def run(self, layer: t.ILayer, args: t.SearchArgs) -> t.List[t.IFeature]:
        args.source_layer_names = [sl.name for sl in self.source_layers]
        args.tolerance = args.tolerance or self.tolerance
        return self.provider.find_features(args)
