import gws.types as t
import gws.gis.source
import gws.ows.wms
import gws.ows.util
import gws.tools.net
import gws.tools.misc as misc
import gws.gis.proj


class Config(gws.gis.source.BaseConfig):
    """WMS source"""

    maxRequests: int = 1  #: max concurrent requests to this source
    capsCacheMaxAge: t.duration = '1d'  #: max cache age for capabilities documents
    params: t.Optional[dict]  #: query string parameters
    options: t.Optional[dict]  #: additional options
    url: t.url  #: service url


class Object(gws.gis.source.Base, t.SourceObject):
    def __init__(self):
        super().__init__()
        self.service: gws.ows.wms.Service = None
        self.url = ''

    def configure(self):
        super().configure()

        self.url = self.var('url')
        self.service = gws.ows.util.shared_service('WMS', self, self.config)
        self.crs = self._get_crs()
        self.layers = self.service.layers

    def mapproxy_config(self, mc, options=None):
        layer_names = gws.get(
            options, 'layer_names',
            gws.compact(la.name for la in self.layers))

        req = gws.extend({
            'url': self.service.operations['GetMap'].get_url,
            'transparent': True,
            'layers': ','.join(layer_names)
        }, self.var('params'))

        return mc.source(self, gws.compact({
            'type': 'wms',
            'supported_srs': [self.crs],
            ## 'forward_req_params': ['LAYERS__gws', 'DPI__gws'],
            'concurrent_requests': self.var('maxRequests'),
            'req': req
        }))

    def service_metadata(self):
        return self.service.meta

    def layer_metadata(self, layer_name):
        for la in self.service.layers:
            if la.name == layer_name:
                return la.meta

    def _get_crs(self):
        # @TODO: considering only the top-level bbox (each layer can have its own!)

        supported = self.service.supported_crs

        # crs given explicitly, does the service support it?
        crs = self.var('crs')
        if crs:
            if crs not in supported:
                raise ValueError(f'crs {crs!r} for {self.url!r} is not supported')
            return crs

        # does it support the layer (target) crs?
        # would be nice, because we don't have to reproject it
        # NB: if no crs is explicitly configured, use the service extent
        crs = self.var('crs', parent=True)
        if crs in supported:
            #gws.log.info(f'using crs {crs!r} for {self.url!r}')
            return crs

        # try to find a non-geofraphic projection

        for crs in supported:
            if crs.lower().startswith('epsg') and not gws.gis.proj.is_latlong(crs):
                gws.log.warn(f'using implicit crs {crs!r} for {self.url!r}')
                return crs

        # nothing worked, take the first crs
        return supported[0]
