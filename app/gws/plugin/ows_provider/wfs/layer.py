import gws
import gws.types as t
import gws.base.layer
import gws.base.ows
import gws.lib.ows
import gws.lib.proj
import gws.lib.gis
import gws.lib.gis
from . import provider


@gws.ext.Config('layer.wfs')
class Config(gws.base.layer.vector.Config, provider.Config):
    """WFS layer"""
    sourceLayers: t.Optional[gws.lib.gis.SourceLayerFilter]  #: source layers to use


@gws.ext.Object('layer.wfs')
class Object(gws.base.layer.vector.Object):
    provider: provider.Object
    source_crs: gws.Crs
    source_layers: t.List[gws.lib.gis.SourceLayer]
    source_style: str

    def configure(self):
        self.provider = provider.create(self.root, self.config, shared=True)

        if not self.has_configured_metadata:
            self.configure_metadata_from(self.provider.metadata)

        self.source_layers = gws.lib.gis.filter_source_layers(
            self.provider.source_layers,
            self.var('sourceLayers'))

        if not self.source_layers:
            raise gws.Error(f'no source layers found in layer={self.uid!r}')

        if not self.has_configured_search:
            self.search_providers.append(
                t.cast(gws.ISearchProvider, self.create_child('gws.ext.search.provider.wfs', gws.Config(
                    uid=self.uid + '.default_search',
                    layer=self,
                ))))
            self.has_configured_search = True

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
