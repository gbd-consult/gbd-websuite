import gws
import gws.types as t
import gws.base.ows
import gws.base.layer
import gws.lib.legend
import gws.lib.gis
import gws.lib.net
import gws.lib.zoom
from . import provider


@gws.ext.Config('layer.wmsflat')
class Config(gws.base.layer.image.Config, provider.Config):
    sourceLayers: t.Optional[gws.lib.gis.SourceLayerFilter]  #: source layers to use


@gws.ext.Object('layer.wmsflat')
class Object(gws.base.layer.image.Object):
    source_layers: t.List[gws.lib.gis.SourceLayer]
    provider: provider.Object
    source_crs: gws.Crs

    def configure(self):
        self.provider = gws.base.ows.provider.shared_object(self.root, provider.Object, self.config)

        if not self.has_configured_metadata:
            self.configure_metadata_from(self.provider.metadata)

        self.source_crs = gws.lib.gis.best_crs(self.map.crs, self.provider.supported_crs)

        self.source_layers = gws.lib.gis.filter_source_layers(
            self.provider.source_layers,
            self.var('sourceLayers', default=gws.lib.gis.SourceLayerFilter(level=1)))

        if not self.source_layers:
            raise gws.Error(f'no source layers found in layer={self.uid!r}')

        if not self.has_configured_resolutions:
            zoom = gws.lib.zoom.config_from_source_layers(self.source_layers)
            if zoom:
                self.resolutions = gws.lib.zoom.resolutions_from_config(zoom, self.resolutions)
                self.has_configured_resolutions = True

        if not self.has_configured_search:
            queryable_layers = gws.lib.gis.enum_source_layers(self.source_layers, is_queryable=True)
            if queryable_layers:
                self.search_providers.append(
                    t.cast(gws.ISearchProvider, self.create_child('gws.ext.search.provider.wms', gws.Config(
                        uid=self.uid + '.default_search',
                        direct_layer_object=self,
                        direct_source_layers=queryable_layers
                    ))))
                self.has_configured_search = True

        if not self.has_configured_legend:
            urls = [sl.legend_url for sl in self.source_layers if sl.legend_url]
            if urls:
                self.legend = gws.Legend(
                    enabled=True,
                    urls=urls,
                    cache_max_age=self.var('legend.cacheMaxAge', default=0),
                    options=self.var('legend.options', default={}))
            self.has_configured_legend = True

    @property
    def own_bounds(self):
        return gws.lib.gis.bounds_from_source_layers(self.source_layers, self.source_crs)

    @property
    def description(self):
        context = {
            'layer': self,
            'service': self.provider.metadata,
            'sub_layers': self.source_layers
        }
        return self.description_template.render(context).content

    def mapproxy_config(self, mc, options=None):
        layers = [sl.name for sl in self.source_layers if sl.name]
        if not self.var('capsLayersBottomUp'):
            layers = reversed(layers)

        req = gws.merge({
            'url': self.provider.operation('GetMap').get_url,
            'transparent': True,
            'layers': ','.join(layers)
        }, self.var('getMapParams'))

        source_uid = mc.source(gws.compact({
            'type': 'wms',
            'supported_srs': [self.source_crs],
            'concurrent_requests': self.var('maxRequests'),
            'req': req
        }))

        self.mapproxy_layer_config(mc, source_uid)
