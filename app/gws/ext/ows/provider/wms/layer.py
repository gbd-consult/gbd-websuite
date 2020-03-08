import gws
import gws.common.layer
import gws.gis.zoom
import gws.gis.source
import gws.gis.legend
import gws.gis.util
import gws.common.ows.provider

import gws.types as t

from . import provider, util


class Config(gws.common.layer.ImageConfig, util.WmsConfig):
    getMapParams: t.Optional[dict]  #: additional parameters for GetMap requests


class Object(gws.common.layer.Image):
    def __init__(self):
        super().__init__()

        self.provider: provider.Object = None
        self.source_layers: t.List[t.SourceLayer] = []
        self.source_legend_urls = []
        self.url = ''

    def configure(self):
        super().configure()

        util.configure_wms_for(self)

        if not self.source_layers:
            raise gws.Error(f'no source layers found for {self.uid!r}')

        if not self.var('zoom'):
            zoom = gws.gis.zoom.config_from_source_layers(self.source_layers)
            if zoom:
                self.resolutions = gws.gis.zoom.resolutions_from_config(
                    zoom, self.resolutions)

        if not self.resolutions:
            raise gws.Error(f'no resolutions in {self.uid!r}')

        self._configure_legend()

    @property
    def default_search_provider(self):
        source_layers = gws.gis.source.filter_layers(
            self.provider.source_layers,
            self.var('sourceLayers'),
            queryable_only=True
        )
        if source_layers:
            return self.create_object('gws.ext.search.provider.wms', t.Config(
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

    def render_legend(self):
        if self.legend_url:
            return super().render_legend()
        return gws.gis.legend.combine_legend_urls(self.source_legend_urls)

    def _configure_legend(self):
        self.has_legend = False

        if not self.var('legend.enabled'):
            return

        url = self.var('legend.url')
        if url:
            self.has_legend = True
            self.legend_url = url
            return

        # if no legend.url is given, use a combined source legend (see render_legend above)

        urls = [sl.legend for sl in self.source_layers if sl.legend]
        if not urls:
            return

        self.has_legend = True
        self.source_legend_urls = urls
