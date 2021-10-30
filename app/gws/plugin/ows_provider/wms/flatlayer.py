import gws
import gws.base.layer
import gws.lib.gis
import gws.lib.zoom
import gws.types as t

from . import provider as provider_module


@gws.ext.Config('layer.wmsflat')
class Config(gws.base.layer.image.Config, provider_module.Config):
    sourceLayers: t.Optional[gws.lib.gis.SourceLayerFilter]  #: source layers to use


@gws.ext.Object('layer.wmsflat')
class Object(gws.base.layer.image.Object):
    source_layers: t.List[gws.lib.gis.SourceLayer]
    provider: provider_module.Object
    source_crs: gws.Crs

    def configure(self):
        pass

    def configure_source(self):
        if self.var('_provider'):
            self.provider = self.var('_provider')
            self.source_layers = self.var('_source_layers')
        else:
            self.provider = self.root.create_object(provider_module.Object, self.config, shared=True)
            self.source_layers = gws.lib.gis.filter_source_layers(
                self.provider.source_layers,
                self.var('sourceLayers', default=gws.lib.gis.SourceLayerFilter(level=1)))

        if not self.source_layers:
            raise gws.Error(f'no source layers found in layer={self.uid!r}')

        self.source_crs = gws.lib.gis.best_crs(self.map.crs, self.provider.supported_crs)
        return True

    def configure_metadata(self):
        if not super().configure_metadata():
            self.set_metadata(self.var('metadata'), self.provider.metadata)
            return True

    def configure_zoom(self):
        if not super().configure_zoom():
            zoom = gws.lib.zoom.config_from_source_layers(self.source_layers)
            if zoom:
                self.resolutions = gws.lib.zoom.resolutions_from_config(zoom, self.resolutions)
                return True

    def configure_search(self):
        if not super().configure_search():
            queryable_layers = gws.lib.gis.enum_source_layers(self.source_layers, is_queryable=True)
            if queryable_layers:
                self.search_providers.append(
                    self.require_child('gws.ext.search.provider.wms', gws.Config(
                        uid=self.uid + '.default_search',
                        _provider=self.provider,
                        _source_layers=queryable_layers
                    )))
                return True

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
        return gws.lib.gis.bounds_from_source_layers(self.source_layers, self.source_crs)

    @property
    def description(self):
        context = {
            'layer': self,
            'service_metadata': self.provider.metadata,
            'sub_layers': self.source_layers
        }
        return self.description_template.render(context).content

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
            'supported_srs': [self.source_crs],
            'concurrent_requests': self.var('maxRequests'),
            'req': req
        }))

        self.mapproxy_layer_config(mc, source_uid)
