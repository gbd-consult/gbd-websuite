import gws
import gws.types as t
import gws.gis.layer
import gws.gis.source
import gws.gis.legend
import gws.ows.wms
import gws.ows.request
import gws.ows.util
import gws.gis.proj
import gws.gis.mpx
import gws.tools.misc as misc


class Config(gws.gis.layer.ProxiedConfig):
    """WMS layer"""

    capsCacheMaxAge: t.duration = '1d'  #: max cache age for capabilities documents
    display: str = 'box'
    maxRequests: int = 1  #: max concurrent requests to this source
    params: t.Optional[dict]  #: query string parameters
    sourceLayers: t.Optional[gws.gis.source.LayerFilterConfig]  #: source layers to use
    url: t.url  #: service url


class Object(gws.gis.layer.Proxied):
    def __init__(self):
        super().__init__()

        self.source_crs = ''
        self.url = ''
        self.service: gws.ows.wms.Service = None
        self.source_layers: t.List[t.SourceLayer] = []
        self.source_legend_urls = []

    def configure(self):
        super().configure()

        self.url = self.var('url')
        self.service = gws.ows.util.shared_service('WMS', self, self.config)
        self.source_crs = gws.ows.util.best_crs(self.map.crs, self.service.supported_crs)
        self.source_layers = gws.gis.source.filter_image_layers(self.service.layers, self.var('sourceLayers'))

        if not self.source_layers:
            raise gws.Error(f'no layers found in {self.uid!r}')

        self.cache_uid = misc.sha256(self.url + ' ' + ' '.join(sorted(sl.name for sl in self.source_layers)))
        self._add_default_search()

        self._add_legend()

    def mapproxy_config(self, mc, options=None):
        req = gws.extend({
            'url': self.service.operations['GetMap'].get_url,
            'transparent': True,
            'layers': ','.join(sl.name for sl in self.source_layers)
        }, self.var('params'))

        src = mc.source(self.cache_uid, gws.compact({
            'type': 'wms',
            'supported_srs': [self.source_crs],
            'concurrent_requests': self.var('maxRequests'),
            'req': req
        }))

        self.mapproxy_layer_config(mc, src)

    @property
    def description(self):
        context = {
            'layer': self,
            'service': self.service.meta,
            'sub_layers': self.source_layers
        }

        return self.description_template.render(context).content

    def render_legend(self):
        if self.legend_url:
            return super().render_legend()
        return gws.gis.legend.combine_legends(self.source_legend_urls)

    def _add_default_search(self):
        p = self.var('search')
        if not p.enabled or p.providers:
            return

        qs = [sl for sl in self.source_layers if sl.is_queryable]
        if not qs:
            return

        self.add_child('gws.ext.search.provider', t.Data({
            'type': 'wms',
            'url': self.url,
            'layers': [sl.name for sl in qs]
        }))

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

    # def configure_extent(self):
    #     e = self.var('extent')
    #     if e:
    #         return e
    #
    #     if self.source and self.source.extent:
    #         return gws.gis.proj.transform_bbox(
    #             self.source.extent,
    #             self.source.crs,
    #             self.crs
    #         )
    #     return t.cast(t.MapView, self.parent).extent
