import gws
import gws.base.search
import gws.gis.source
import gws.gis.ows
import gws.types as t

from . import provider as provider_module


@gws.ext.config.finder('wfs')
class Config(gws.base.search.Config, provider_module.Config):
    sourceLayers: t.Optional[gws.gis.source.LayerFilterConfig]  #: source layers to use


@gws.ext.object.finder('wfs')
class Object(gws.base.search.provider.Object):
    # @TODO support filters
    supports_geometry = True

    source_layers: t.List[gws.SourceLayer]
    provider: provider_module.Object

    def configure(self):
        gws.gis.ows.client.configure_layers(self, provider_module.Object, isQueryable=True)

    def can_run(self, args):
        return (
                super().can_run(args)
                and self.provider.operation(gws.OwsVerb.GetFeature))

    def run(self, args, layer=None):
        args.tolerance = args.tolerance or self.tolerance
        return self.provider.find_features(args, self.source_layers)
