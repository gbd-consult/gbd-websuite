import gws
import gws.types as t
import gws.base.ows.provider
import gws.base.layer
import gws.base.layer.image
import gws.lib.legend
import gws.lib.gis
import gws.lib.zoom
from . import provider


@gws.ext.Config('layer.wmsflat')
class Config(gws.base.layer.image.Config, provider.Config):
    sourceLayers: t.Optional[gws.lib.gis.LayerFilter]  #: source layers to use


@gws.ext.Object('layer.wmsflat')
class Object(gws.base.layer.image.Object):
    source_layers: t.List[gws.lib.gis.SourceLayer]
    provider: provider.Object

    def configure(self):
        self.provider = gws.base.ows.provider.shared_object(self.root, provider.Object, self.config)

        if not self.has_configured.metadata:
            self.configure_metadata_from(self.provider.metadata)

        self.source_layers = gws.lib.gis.filter_layers(
            self.provider.source_layers,
            self.var('sourceLayers'))

        if not self.source_layers:
            raise gws.Error(f'no source layers found in layer={self.uid!r}')

        if not self.has_configured.resolutions:
            zoom = gws.lib.zoom.config_from_source_layers(self.source_layers)
            if zoom:
                self.resolutions = gws.lib.zoom.resolutions_from_config(zoom, self.resolutions)
                self.has_configured.resolutions = True

        if not self.has_configured.search:
            queryable_layers = gws.lib.gis.filter_layers(
                self.provider.source_layers,
                self.var('sourceLayers'),
                queryable_only=True
            )
            if queryable_layers:
                self.search_providers.append(
                    t.cast(gws.ISearchProvider, self.create_child('gws.ext.search.provider.wms', gws.Config(
                        uid=self.uid + '.default_search',
                        layer=self,
                        source_layers=queryable_layers
                    ))))
                self.has_configured.search = True

        if not self.has_configured.legend:
            self.legend = gws.Legend(
                enabled=True,
                source_urls=[sl.legend for sl in self.source_layers if sl.legend],
                options=self.var('legend.options', default={}))
            self.has_configured.legend = True

    @property
    def own_bounds(self):
        our_crs = gws.lib.gis.best_crs(self.map.crs, self.provider.supported_crs)
        return gws.lib.gis.bounds_from_layers(self.source_layers, our_crs)

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

        our_crs = gws.lib.gis.best_crs(self.map.crs, self.provider.supported_crs)

        req = gws.merge({
            'url': self.provider.operation('GetMap').get_url,
            'transparent': True,
            'layers': ','.join(layers)
        }, self.var('getMapParams'))

        source_uid = mc.source(gws.compact({
            'type': 'wms',
            'supported_srs': [our_crs],
            'concurrent_requests': self.var('maxRequests'),
            'req': req
        }))

        self.mapproxy_layer_config(mc, source_uid)

    def render_legend(self, context=None):
        sup = super().render_legend(context)
        if sup:
            return sup
        return gws.lib.legend.combine_urls(self.legend.source_urls, self.legend.options)
