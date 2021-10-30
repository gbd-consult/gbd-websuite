import gws
import gws.base.search
import gws.lib.gis
import gws.lib.shape
import gws.types as t
from . import provider as provider_module


@gws.ext.Config('search.provider.wfs')
class Config(gws.base.search.Config, provider_module.Config):
    sourceLayers: t.Optional[gws.lib.gis.SourceLayerFilter]  #: source layers to use


@gws.ext.Object('search.provider.wfs')
class Object(gws.base.search.provider.Object):
    # @TODO support filters
    supports_geometry = True

    source_layers: t.List[gws.lib.gis.SourceLayer]
    provider: provider_module.Object

    def configure(self):
        layer = self.var('layer')
        if layer:
            self.provider = layer.provider
            self.source_layers = layer.source_layers
        else:
            self.provider = self.root.create_object(provider_module.Object, self.config, shared=True)
            self.source_layers = gws.lib.gis.filter_source_layers(
                self.provider.source_layers,
                self.var('sourceLayers'))

        if not self.source_layers:
            raise gws.Error(f'no source layers in {self.uid!r}')

    def can_run(self, args):
        return (
                super().can_run(args)
                and self.provider.operation(gws.OwsVerb.GetFeature))

    def run(self, args, layer=None):
        args.source_layer_names = [sl.name for sl in self.source_layers]
        args.tolerance = args.tolerance or self.tolerance
        return self.provider.find_features(args)
