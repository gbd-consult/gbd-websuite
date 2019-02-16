import gws
import gws.common.search
import gws.config
import gws.gis.layer
import gws.gis.mpx
import gws.gis.source
import gws.gis.zoom
import gws.ows.request
import gws.ows.util
import gws.qgis
import gws.types as t


class Config(gws.gis.layer.ImageConfig):
    """WMS layer from a Qgis project"""

    path: t.filepath  #: qgis project path
    display: str = 'box'
    sourceLayers: t.Optional[gws.gis.source.LayerFilterConfig]  #: source layers to use
    sourceLayerOder: gws.gis.source.LayerOrder = 'topDown'  #: order of layers in the GetCapabilities document


class Object(gws.gis.layer.Image):
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

        self.source_layers = gws.gis.source.filter_layers(
            self.service.layers,
            self.var('sourceLayers'),
            image_only=True)
        if not self.source_layers:
            raise gws.Error(f'no layers found in {self.uid!r}')

        if not self.var('zoom'):
            zoom = gws.gis.zoom.config_from_source_layers(self.source_layers)
            if zoom:
                self.resolutions = gws.gis.zoom.resolutions_from_config(
                    zoom, self.resolutions)

        self._add_default_search()

        # if no legend.url is given, use an auto legend
        self.has_legend = self.var('legend.enabled')

    def render_bbox(self, bbox, width, height, **client_params):
        forward = {}

        cache_uid = self.uid

        if not self.has_cache:
            cache_uid = self.uid + '_NOCACHE'

        if 'dpi' in client_params:
            dpi = gws.as_int(client_params['dpi'])
            if dpi > 90:
                forward['DPI__gws'] = str(dpi)
                cache_uid = self.uid + '_NOCACHE'

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
        return self.service.get_legend(self.source_layers)

    def mapproxy_config(self, mc, options=None):
        layers = [sl.name for sl in self.source_layers]
        if self.var('sourceLayerOder') == 'topDown':
            layers = reversed(layers)

        source = mc.source({
            'type': 'wms',
            'supported_srs': self.service.supported_crs,
            'forward_req_params': ['DPI__gws'],
            'concurrent_requests': self.root.var('server.qgis.maxRequests', default=0),
            'req': {
                'url': self.service.url,
                'map': self.path,
                'layers': ','.join(layers),
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

        queryable_layers = gws.gis.source.filter_layers(
            self.service.layers,
            self.var('sourceLayers'),
            queryable_only=True)
        if not queryable_layers:
            return

        self.add_child('gws.ext.search.provider', t.Data({
            'type': 'qgiswms',
            'path': self.path,
            'sourceLayers': t.Data({
                'names': [sl.name for sl in queryable_layers]
            })
        }))
