import gws
import gws.base.layer
import gws.lib.gis
import gws.types as t

from . import provider as provider_module


@gws.ext.Config('layer.wfs')
class Config(gws.base.layer.vector.Config, provider_module.Config):
    """WFS layer"""
    sourceLayers: t.Optional[gws.lib.gis.SourceLayerFilter]  #: source layers to use


@gws.ext.Object('layer.wfs')
class Object(gws.base.layer.vector.Object):
    provider: provider_module.Object
    source_crs: gws.Crs
    source_layers: t.List[gws.lib.gis.SourceLayer]
    source_style: str

    def configure(self):
        pass

    def configure_source(self):
        self.provider = self.root.create_object(provider_module.Object, self.config, shared=True)
        self.source_layers = gws.lib.gis.filter_source_layers(
            self.provider.source_layers,
            self.var('sourceLayers'))
        if not self.source_layers:
            raise gws.Error(f'no source layers found in layer={self.uid!r}')
        return True

    def configure_metadata(self):
        if not super().configure_metadata():
            self.set_metadata(self.var('metadata'), self.provider.metadata)
            return True

    def configure_search(self):
        if not super().configure_search():
            self.search_providers.append(
                self.require_child('gws.ext.search.provider.wfs', gws.Config(
                    uid=self.uid + '.default_search',
                    layer=self,
                )))
            return True

    @property
    def description(self):
        context = {
            'layer': self,
            'service': self.provider.metadata,
            'sub_layers': self.source_layers
        }
        return self.description_template.render(context).content

    def get_features(self, bounds, limit=0):
        features = self.provider.find_features(gws.SearchArgs(
            bounds=bounds,
            limit=limit,
            source_layer_names=[sl.name for sl in self.source_layers]
        ))
        return [f.connect_to(self) for f in features]
