import gws
import gws.base.model
import gws.base.search
import gws.gis.source
import gws.types as t

from . import provider


@gws.ext.config.finder('wms')
class Config(gws.base.search.finder.Config, provider.Config):
    sourceLayers: t.Optional[gws.gis.source.LayerFilterConfig]
    """Source layers to search for."""


@gws.ext.object.finder('wms')
class Object(gws.base.search.finder.Object):
    supportsGeometry = True

    provider: provider.Object
    sourceLayers: t.List[gws.SourceLayer]

    def configure(self):
        if self.var('_provider'):
            self.provider = self.var('_provider')
            self.sourceLayers = self.var('_sourceLayers')
            self.modelMgr = self.var('_modelMgr')
        else:
            self.provider = self.root.create_shared(provider.Object, self.config)
            self.sourceLayers = gws.gis.source.filter_layers(
                self.var('sourceLayers') or self.provider.sourceLayers,
                gws.gis.source.LayerFilter(isQueryable=True))

            if not self.provider.get_operation(gws.OwsVerb.GetFeatureInfo):
                raise gws.Error(f'GetFeatureInfo is not supported for {self.provider.url!r}')

            if not self.sourceLayers:
                raise gws.Error(f'no queriable layers for {self.provider.url!r}')

    def can_run(self, args):
        return (
                super().can_run(args)
                and bool(args.shapes)
                and len(args.shapes) == 1
                and args.shapes[0].type == gws.GeometryType.POINT)

    def run(self, args, layer=None):
        fs = self.provider.find_source_features(args, self.sourceLayers)
        if not fs:
            return []

        model_mgr = self.modelMgr
        if not model_mgr and layer:
            model_mgr = layer.modelMgr

        model = model_mgr.get_model_for(args.user)
        return [model.feature_from_source(f) for f in fs]
