import gws
import gws.config
import gws.types as t
import gws.gis.layer
import gws.qgis
import gws.gis.source
import gws.gis.mpx
import gws.ows.request
import gws.tools.misc as misc
import gws.ows.util
import gws.common.search


class Config(gws.gis.layer.ProxiedConfig):
    """WMS layer from a Qgis project"""

    path: t.filepath  #: qgis project path
    display: str = 'box'
    sourceLayers: t.Optional[gws.gis.source.LayerFilterConfig]  #: source layers to use


class Object(gws.gis.layer.Proxied):
    def __init__(self):
        super().__init__()

        self.source_crs = ''
        self.path = ''
        self.service: gws.qgis.Service = None
        self.source_layers: t.List[t.SourceLayer] = []

    def configure(self):
        super().configure()

        self.path = self.var('path')
        self.service = gws.qgis.shared_service(self, self.config)
        self.source_crs = gws.ows.util.best_crs(self.map.crs, self.service.supported_crs)
        self.source_layers = gws.gis.source.filter_image_layers(self.service.layers, self.var('sourceLayers'))

        if not self.source_layers:
            raise gws.Error(f'no layers found in {self.uid!r}')

        self.cache_uid = misc.sha256(self.path + ' ' + ' '.join(sorted(sl.name for sl in self.source_layers)))
        self._add_default_search()

        # if no legend.url is given, use an auto legend
        self.has_legend = self.var('legend.enabled')


    def render_bbox(self, bbox, width, height, **client_params):
        forward = {}

        cache_uid = self.cache_uid

        if not self.has_cache:
            cache_uid = self.cache_uid + '_NOCACHE'

        if 'dpi' in client_params:
            try:
                dpi = int(client_params['dpi'])
            except:
                dpi = 0
            if dpi > 90:
                forward['DPI__gws'] = str(dpi)
                cache_uid = self.cache_uid + '_NOCACHE'

        return gws.gis.mpx.wms_request(
            cache_uid,
            bbox,
            width,
            height,
            self.map.crs,
            forward)

    def render_legend(self):
        if self.legend_url:
            return super().render_legend()
        return self.service.get_legend(sl.name for sl in self.source_layers)

    def mapproxy_config(self, mc, options=None):

        source = mc.source(self.cache_uid, {
            'type': 'wms',
            'supported_srs': self.service.supported_crs,
            'forward_req_params': ['DPI__gws'],
            'concurrent_requests': self.root.var('server.qgis.maxRequests', default=0),
            'req': {
                'url': self.service.url,
                'map': self.path,
                'layers': ','.join(gws.compact(sl.name for sl in self.source_layers)),
                'transparent': True,
            }
        })

        self.mapproxy_layer_config(mc, source)

    @property
    def description(self):
        ctx = {
            'layer': self,
            'service': self.service.meta
        }
        return self.description_template.render(ctx).content
        #
        # sub_layers = self.source_layers
        # if len(sub_layers) == 1 and gws.get(sub_layers[0], 'title') == self.title:
        #     sub_layers = []
        #
        # return super().description(gws.defaults(
        #     options,
        #     sub_layers=sub_layers))

    def _add_default_search(self):
        p = self.var('search')
        if not p.enabled or p.providers:
            return

        qs = [sl for sl in self.source_layers if sl.is_queryable]
        if not qs:
            return

        self.add_child('gws.ext.search.provider', t.Data({
            'type': 'qgiswms',
            'path': self.path,
            'layers': [sl.name for sl in qs]
        }))
