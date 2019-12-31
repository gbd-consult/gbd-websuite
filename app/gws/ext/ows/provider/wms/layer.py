import gws
import gws.common.layer
import gws.gis.zoom
import gws.gis.source
import gws.gis.legend
import gws.gis.util
import gws.common.ows.provider

import gws.types as t

from . import types, provider, util


class Config(gws.common.layer.ImageConfig, types.WmsConfig):
    getMapParams: t.Optional[dict]  #: additional parameters for GetMap requests


class Object(gws.common.layer.Image):
    def __init__(self):
        super().__init__()

        self.invert_axis_crs = []
        self.provider: provider.Object = None
        self.source_layers: t.List[types.SourceLayer] = []
        self.source_legend_urls = []
        self.url = ''

    def configure(self):
        super().configure()

        util.configure_wms(self)

        if not self.source_layers:
            raise gws.Error(f'no layers found in {self.uid!r}')

        if not self.var('zoom'):
            zoom = gws.gis.zoom.config_from_source_layers(self.source_layers)
            if zoom:
                self.resolutions = gws.gis.zoom.resolutions_from_config(
                    zoom, self.resolutions)

        if not self.resolutions:
            raise gws.Error(f'no resolutions in {self.uid!r}')

        self._add_default_search()
        self._add_legend()

    @property
    def own_bounds(self):
        return gws.gis.source.bounds_from_layers(self.source_layers, self.map.crs)

    def mapproxy_config(self, mc, options=None):
        layers = [sl.name for sl in self.source_layers if sl.name]
        if not self.var('capsLayersBottomUp'):
            layers = reversed(layers)

        crs = gws.gis.util.best_crs(self.map.crs, self.provider.supported_crs)

        req = gws.extend({
            'url': self.provider.operation('GetMap').get_url,
            'transparent': True,
            'layers': ','.join(layers)
        }, self.var('getMapParams'))

        source_uid = mc.source(gws.compact({
            'type': 'wms',
            'supported_srs': [crs],
            'concurrent_requests': self.var('maxRequests'),
            'req': req
        }))

        self.mapproxy_layer_config(mc, source_uid)

    @property
    def description(self):
        context = {
            'layer': self,
            'service': self.provider.meta,
            'sub_layers': self.source_layers
        }
        return self.description_template.render(context).content

    def render_legend(self):
        if self.legend_url:
            return super().render_legend()
        return gws.gis.legend.combine_legend_urls(self.source_legend_urls)

    def _add_default_search(self):
        p = self.var('search')
        if not p.enabled or p.providers:
            return

        cfg = {
            'type': 'wms'
        }

        cfg_keys = [
            'capsCacheMaxAge',
            'invertAxis',
            'maxRequests',
            'bottomUpLayers',
            'sourceLayers',
            'url',
        ]

        for key in cfg_keys:
            cfg[key] = self.var(key)

        self.add_child('gws.ext.search.provider', t.Config(gws.compact(cfg)))

    def _add_legend(self):
        self.has_legend = False

        if not self.var('legend.enabled'):
            return

        url = self.var('legend.url')
        if url:
            self.has_legend = True
            self.legend_url = url
            return

        # if no legend.url is given, use an auto legend

        urls = [sl.legend for sl in self.source_layers if sl.legend]
        if not urls:
            return

        if len(urls) == 1:
            self.has_legend = True
            self.legend_url = urls[0]
            return

        # see render_legend

        self.source_legend_urls = urls
        self.has_legend = True
