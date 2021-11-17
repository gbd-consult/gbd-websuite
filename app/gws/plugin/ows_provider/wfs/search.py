import gws
import gws.base.search
import gws.lib.gis.source
import gws.lib.ows
import gws.types as t

from . import provider as provider_module


@gws.ext.Config('search.provider.wfs')
class Config(gws.base.search.Config, provider_module.Config):
    sourceLayers: t.Optional[gws.lib.gis.source.LayerFilterConfig]  #: source layers to use


@gws.ext.Object('search.provider.wfs')
class Object(gws.base.search.provider.Object):
    # @TODO support filters
    supports_geometry = True

    source_layers: t.List[gws.SourceLayer]
    provider: provider_module.Object

    def configure(self):
        gws.lib.ows.client.configure_layers(self, provider_module.Object, is_queryable=True)

    def can_run(self, args):
        return (
                super().can_run(args)
                and self.provider.operation(gws.OwsVerb.GetFeature))

    def run(self, args, layer=None):
        args.tolerance = args.tolerance or self.tolerance
        return self.provider.find_features(args, self.source_layers)
