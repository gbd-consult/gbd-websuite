import gws
import gws.base.layer
import gws.lib.gis.util
import gws.types as t
import gws.lib.ows
from . import provider as provider_module
from . import search


@gws.ext.Config('layer.wfs')
class Config(gws.base.layer.vector.Config, provider_module.Config):
    """WFS layer"""
    pass


@gws.ext.Object('layer.wfs')
class Object(gws.base.layer.vector.Object, gws.IOwsClient):
    provider: provider_module.Object

    def configure_source(self):
        gws.lib.ows.client.configure_layers(self, provider_module.Object)
        return True

    def configure_metadata(self):
        if not super().configure_metadata():
            self.set_metadata(self.var('metadata'), self.provider.metadata)
            return True

    def configure_search(self):
        if not super().configure_search():
            return gws.lib.ows.client.configure_search(self, search.Object)

    def get_features(self, bounds, limit=0):
        features = self.provider.find_features(
            gws.SearchArgs(bounds=bounds, limit=limit),
            self.source_layers)
        return [f.connect_to(self) for f in features]
