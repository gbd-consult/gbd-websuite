import gws
import gws.base.layer
import gws.gis.source
import gws.gis.ows
import gws.types as t

from . import provider as provider_module
from . import search


@gws.ext.config.layer('wmsflat')
class Config(gws.base.layer.image.Config, provider_module.Config):
    pass


@gws.ext.object.layer('wmsflat')
class Object(gws.base.layer.image.Object, gws.IOwsClient):
    provider: provider_module.Object
    source_crs: gws.ICrs

    def configure_source(self):
        gws.gis.ows.client.configure_layers(self, provider_module.Object, is_image=True)
        self.source_crs = gws.gis.crs.best_match(
            self.provider.force_crs or self.crs,
            gws.gis.source.supported_crs_list(self.source_layers))
        return True

    def configure_metadata(self):
        if not super().configure_metadata():
            self.set_metadata(self.var('metadata'), self.provider.metadata)
            return True

    def configure_zoom(self):
        if not super().configure_zoom():
            return gws.gis.ows.client.configure_zoom(self)

    def configure_search(self):
        if not super().configure_search():
            return gws.gis.ows.client.configure_search(self, search.Object)

    def configure_legend(self):
        if not super().configure_legend():
            urls = [sl.legend_url for sl in self.source_layers if sl.legend_url]
            if urls:
                self.legend = gws.Legend(
                    enabled=True,
                    urls=urls,
                    cache_max_age=self.var('legend.cacheMaxAge', default=0),
                    options=self.var('legend.options', default={}))
                return True

    @property
    def own_bounds(self):
        return gws.gis.source.combined_bounds(self.source_layers, self.source_crs)

    def mapproxy_config(self, mc, options=None):
        layers = [sl.name for sl in self.source_layers if sl.name]
        if not self.var('capsLayersBottomUp'):
            layers = reversed(layers)

        args = self.provider.operation_args(gws.OwsVerb.GetMap)

        req = gws.merge(args['params'], {
            'transparent': True,
            'layers': ','.join(layers),
            'url': args['url'],
        })

        source_uid = mc.source(gws.compact({
            'type': 'wms',
            'supported_srs': [self.source_crs.epsg],
            'concurrent_requests': self.var('maxRequests'),
            'req': req
        }))

        self.mapproxy_layer_config(mc, source_uid)
