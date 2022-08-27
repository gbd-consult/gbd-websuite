import gws
import gws.base.layer
import gws.gis.source
import gws.gis.ows
import gws.types as t

from . import provider
from . import search


@gws.ext.config.layer('wmsflat')
class Config(gws.base.layer.Config, provider.Config):
    pass


@gws.ext.object.layer('wmsflat')
class Object(gws.base.layer.Object, gws.IOwsClient):
    provider: provider.Object
    sourceCrs: gws.ICrs

    def configure(self):
        super().configure()

        self.provider = self.var('_provider') or self.root.create_shared(provider.Object, self.config)

        self.sourceLayers = self.var('_sourceLayers')
        if not self.sourceLayers:
            slf = gws.merge(
                gws.gis.source.LayerFilter(level=1, is_image=True),
                self.var('sourceLayers'))
            self.sourceLayers = gws.gis.source.filter_layers(self.provider.source_layers, slf)
        if not self.sourceLayers:
            raise gws.ConfigurationError(f'no source layers found for {self.uid!r}')

        self.sourceCrs = gws.gis.crs.best_match(
            self.provider.force_crs or self.crs,
            gws.gis.source.supported_crs_list(self.sourceLayers))

        if not self.configure_metadata():
            self.set_metadata(self.provider.metadata)

        if not self.configure_zoom():
            gws.gis.ows.client.configure_zoom(self)

        if not self.configure_search():
            slf = gws.gis.source.LayerFilter(is_queryable=True)
            queryable_layers = gws.gis.source.filter_layers(self.sourceLayers, slf)
            if queryable_layers:
                self.cFinders.add_finder(gws.Config(
                    type='wms',
                    _provider=self.provider,
                    _sourceLayers=queryable_layers
                ))
                return True

        if not self.configure_legend():
            urls = [sl.legend_url for sl in self.sourceLayers if sl.legend_url]
            if urls:
                self.legend = gws.Legend(
                    enabled=True,
                    urls=urls,
                    cache_max_age=self.var('legend.cacheMaxAge', default=0),
                    options=self.var('legend.options', default={}))

    @property
    def own_bounds(self):
        return gws.gis.source.combined_bounds(self.sourceLayers, self.sourceCrs)

    def render(self, lri):
        return gws.base.layer.lib.generic_raster_render(self, lri)

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
            'supported_srs': [self.sourceCrs.epsg],
            'concurrent_requests': self.var('maxRequests'),
            'req': req
        }))

        gws.base.layer.lib.mapproxy_layer_config(self, mc, source_uid)
