import gws
import gws.base.search
import gws.lib.gis.source
import gws.lib.ows
import gws.types as t

from . import provider as provider_module


@gws.ext.Config('search.provider.wms')
class Config(gws.base.search.Config, provider_module.Config):
    sourceLayers: t.Optional[gws.lib.gis.source.LayerFilterConfig]  #: source layers to use


@gws.ext.Object('search.provider.wms')
class Object(gws.base.search.provider.Object, gws.IOwsClient):
    supports_geometry = True
    provider: provider_module.Object

    def configure(self):
        gws.lib.ows.client.configure_layers(self, provider_module.Object, is_queryable=True)

    def can_run(self, args):
        return (
                super().can_run(args)
                and self.provider.operation(gws.OwsVerb.GetFeatureInfo)
                and bool(args.shapes)
                and len(args.shapes) == 1
                and args.shapes[0].geometry_type == gws.GeometryType.point)

    def run(self, args: gws.SearchArgs, layer: gws.ILayer = None) -> t.List[gws.IFeature]:
        return self.provider.find_features(args, self.source_layers)
