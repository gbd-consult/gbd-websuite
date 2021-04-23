import gws
import gws.common.layer
import gws.gis.legend
import gws.gis.ows
import gws.gis.source
import gws.gis.source
import gws.gis.util
import gws.gis.zoom

import gws.types as t

from . import provider


class Config(gws.common.layer.ImageConfig, provider.Config):
    sourceLayers: t.Optional[gws.gis.source.LayerFilter]  #: source layers to use


class Object(gws.common.layer.Image):
    def configure(self):
        super().configure()

        self.provider = gws.gis.ows.shared_provider(provider.Object, self, self.config)

        self.meta = self.configure_metadata(self.provider.meta)
        self.title = self.meta.title

        self.source_layers = gws.gis.source.filter_layers(
            self.provider.source_layers,
            self.var('sourceLayers'))
        if not self.source_layers:
            raise gws.Error(f'no source layers found for {self.uid!r}')

        if not self.var('zoom'):
            zoom = gws.gis.zoom.config_from_source_layers(self.source_layers)
            if zoom:
                self.resolutions = gws.gis.zoom.resolutions_from_config(
                    zoom, self.resolutions)

        if not self.resolutions:
            raise gws.Error(f'no resolutions in {self.uid!r}')

    @property
    def default_search_provider(self):
        source_layers = gws.gis.source.filter_layers(
            self.provider.source_layers,
            self.var('sourceLayers'),
            queryable_only=True
        )
        if source_layers:
            return self.root.create_object('gws.ext.search.provider.wms', t.Config(
                uid=self.uid + '.default_search',
                layer=self,
                source_layers=source_layers))

    @property
    def own_bounds(self):
        our_crs = gws.gis.util.best_crs(self.map.crs, self.provider.supported_crs)
        return gws.gis.source.bounds_from_layers(self.source_layers, our_crs)

    @property
    def description(self):
        context = {
            'layer': self,
            'service': self.provider.meta,
            'sub_layers': self.source_layers
        }
        return self.description_template.render(context).content

    def mapproxy_config(self, mc, options=None):
        layers = [sl.name for sl in self.source_layers if sl.name]
        if not self.var('capsLayersBottomUp'):
            layers = reversed(layers)

        our_crs = gws.gis.util.best_crs(self.map.crs, self.provider.supported_crs)

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

    def configure_legend(self):
        legend = super().configure_legend() or t.LayerLegend(enabled=True)
        legend.source_legends = [sl.legend for sl in self.source_layers if sl.legend]
        return legend

    def render_legend_image(self, context=None):
        return gws.gis.legend.combine_legend_urls(self.legend.source_legends)
