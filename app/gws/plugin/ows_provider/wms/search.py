import gws
import gws.base.search
import gws.lib.gis
import gws.types as t

from . import provider as provider_module


@gws.ext.Config('search.provider.wms')
class Config(gws.base.search.Config, provider_module.Config):
    sourceLayers: t.Optional[gws.lib.gis.SourceLayerFilter]  #: source layers to use


@gws.ext.Object('search.provider.wms')
class Object(gws.base.search.provider.Object):
    supports_geometry = True
    source_layers: t.List[gws.lib.gis.SourceLayer]
    provider: provider_module.Object

    def configure(self):
        if self.var('_provider'):
            self.provider = self.var('_provider')
            self.source_layers = self.var('_source_layers')
        else:
            self.provider = self.root.create_object(provider_module.Object, self.config, shared=True)
            self.source_layers = gws.lib.gis.enum_source_layers(
                gws.lib.gis.filter_source_layers(self.provider.source_layers, self.var('sourceLayers')),
                is_queryable=True)

        if not self.source_layers:
            raise gws.Error(f'no source layers in {self.uid!r}')

    def can_run(self, args):
        return (
                super().can_run(args)
                and self.provider.operation(gws.OwsVerb.GetFeatureInfo)
                and bool(args.shapes)
                and len(args.shapes) == 1
                and args.shapes[0].geometry_type == gws.GeometryType.point)

    def run(self, args: gws.SearchArgs, layer: gws.ILayer = None) -> t.List[gws.IFeature]:
        args.source_layer_names = [sl.name for sl in self.source_layers]
        return self.provider.find_features(args)
