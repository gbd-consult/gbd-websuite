import gws.config
import gws.types as t
import gws.gis.source
import gws.qgis
import gws.server.monitor


class Config(gws.gis.source.BaseConfig):
    """Qgis source"""

    path: t.filepath  #: path to a qgs project file


class Object(gws.gis.source.Base, t.SourceObject):
    def __init__(self):
        super().__init__()
        self.service: gws.qgis.Service = None
        self.url = ''
        self.path = ''

    def configure(self):
        super().configure()

        self.path = self.var('path')
        self.service = gws.qgis.shared_service(self, self.config)
        self.url = self.service.url
        self.layers = self.service.layers

        for la in self.layers:
            if not la.meta.image:
                # @TODO see actions/map
                la.meta.image = '/_?' + gws.as_query_string({
                    'cmd': 'mapHttpGetQgisLegend',
                    'map': self.path,
                    'layer': la.name
                })

        self.crs = self.service.supported_crs[0]

        # @TODO: smarter extent
        # mapcanvas.extent shows what's on the screen, we probably want WMSExtent or whatever...
        # dont advertise extent for now
        # self.extent = self.service.extent

        gws.server.monitor.add_directory(self.path, '\.qg[sz]$')

    def mapproxy_config(self, mc, options=None):
        layer_names = gws.get(
            options, 'layer_names',
            gws.compact(la.name for la in self.layers))

        return mc.source(self, {
            'type': 'wms',
            'supported_srs': self.service.supported_crs,
            'forward_req_params': ['LAYERS__gws', 'DPI__gws'],
            'concurrent_requests': gws.config.var('server.qgis.maxRequests', default=0),
            'req': {
                'url': self.url + '/?',
                'map': self.path,
                # NB dpi=90 must be set for QGIS, because otherwise it calculates OGC dpi=90.7
                # and rounds it _up_ to 91, thus making scales slightly larger than they are
                # @TODO this causes printing problems, removed for now
                ## 'dpi': 90,
                'layers': ','.join(layer_names),
                'transparent': True,
            }
        })
