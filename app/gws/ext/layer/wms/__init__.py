import gws
import gws.gis.layer
import gws.gis.legend
import gws.gis.proj
import gws.gis.shape
import gws.gis.source
import gws.gis.zoom
import gws.ows.request
import gws.ows.util
import gws.ows.wms
import gws.tools.json2
import gws.common.search.provider

import gws.types as t

"""

NB: layer order
our configuration lists layers top-to-bottom,
this also applies by default to WMS caps (like in qgis)

for servers with bottom-up caps, set capsLayersBottomUp=True 

the order of GetMap is always bottomUp:

> A WMS shall render the requested layers by drawing the leftmost in the list bottommost, 
> the next one over that, and so on.

http://portal.opengeospatial.org/files/?artifact_id=14416
section 7.3.3.3 

"""


class WmsServiceConfig(t.Config):
    capsCacheMaxAge: t.duration = '1d'  #: max cache age for capabilities documents
    invertAxis: t.Optional[t.List[t.crsref]]  #: projections that have an inverted axis (yx)
    maxRequests: int = 0  #: max concurrent requests to this source
    capsLayersBottomUp: bool = False  #: layers are listed from bottom to top in the GetCapabilities document
    sourceLayers: t.Optional[gws.gis.source.LayerFilterConfig]  #: source layers to use
    url: t.url  #: service url


def configure_wms(target: gws.PublicObject, **filter_args):
    target.url = target.var('url')

    target.service = gws.ows.util.shared_service('WMS', target, target.config)
    target.invert_axis = target.var('invertAxis')
    target.source_layers = gws.gis.source.filter_layers(
        target.service.layers,
        target.var('sourceLayers'),
        **filter_args)


class Config(gws.gis.layer.ImageConfig, WmsServiceConfig):
    """WMS layer"""

    getMapParams: t.Optional[dict]  #: additional parameters for GetMap requests


class Object(gws.gis.layer.Image):
    def __init__(self):
        super().__init__()

        self.invert_axis = []
        self.service: gws.ows.wms.Service = None
        self.source_layers: t.List[t.SourceLayer] = []
        self.source_legend_urls = []
        self.url = ''

    def configure(self):
        super().configure()

        configure_wms(self)

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

        self.configure_extent(gws.gis.layer.extent_from_source_layers(self))

    def mapproxy_config(self, mc, options=None):
        layers = [sl.name for sl in self.source_layers if sl.name]
        if not self.var('capsLayersBottomUp'):
            layers = reversed(layers)

        crs = gws.ows.util.best_crs(self.map.crs, self.service.supported_crs)

        req = gws.extend({
            'url': self.service.operations['GetMap'].get_url,
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


class SearchConfig(gws.common.search.provider.Config, WmsServiceConfig):
    pass


class SearchObject(gws.common.search.provider.Object):
    def __init__(self):
        super().__init__()

        self.invert_axis = []
        self.service: gws.ows.wms.Service = None
        self.source_layers: t.List[t.SourceLayer] = []
        self.url = ''

    def configure(self):
        super().configure()
        self.map = self.get_closest('gws.common.map')
        configure_wms(self, queryable_only=True)

    def can_run(self, args):
        return (
                self.source_layers
                and 'GetFeatureInfo' in self.service.operations
                and args.shapes
                and args.shapes[0].type == 'Point'
                and not args.keyword
        )

    def run(self, layer: t.LayerObject, args: t.SearchArgs) -> t.List[t.FeatureInterface]:
        shape = args.shapes[0]
        crs, shape = gws.ows.util.crs_and_shape(args.crs, self.service.supported_crs, shape)
        axis = gws.ows.util.best_axis(args.crs, self.invert_axis, 'WMS', self.service.version)

        fa = t.FindFeaturesArgs({
            'axis': axis,
            'bbox': '',
            'count': args.limit,
            'crs': crs,
            'layers': [sl.name for sl in self.source_layers],
            'params': self.var('params'),
            'point': [shape.geo.x, shape.geo.y],
            'resolution': args.resolution,
        })

        gws.log.debug(f'WMS_QUERY: START')
        gws.p(fa)

        fs = self.service.find_features(fa)

        if fs is None:
            gws.log.debug('WMS_QUERY: NOT_PARSED')
            return []

        gws.log.debug(f'WMS_QUERY: FOUND {len(fs)}')
        return fs
