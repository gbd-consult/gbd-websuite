import gws
import gws.common.ows.provider
import gws.gis.ows
import gws.config
import gws.tools.xml3
import gws.tools.net
import gws.gis.util
import gws.server.monitor

import gws.types as t

from . import types, parser


def create_shared(obj, cfg) -> 'Object':
    path = cfg.get('path')
    uid = path
    gws.server.monitor.add_path(path)

    return obj.create_shared_object(Object, uid, t.Config({
        'path': path,
    }))


# see https://docs.qgis.org/2.18/en/docs/user_manual/working_with_ogc/ogc_server_support.html#getlegendgraphics-request

_LEGEND_DEFAULTS = {
    'BOXSPACE': 2,
    'ICONLABELSPACE': 2,
    'ITEMFONTBOLD': False,
    'ITEMFONTCOLOR': '#000000',
    'ITEMFONTFAMILY': 'DejaVuSans',
    'ITEMFONTITALIC': False,
    'ITEMFONTSIZE': 9,
    'LAYERFONTBOLD': True,
    'LAYERFONTCOLOR': '#000000',
    'LAYERFONTFAMILY': 'DejaVuSans',
    'LAYERFONTITALIC': False,
    'LAYERFONTSIZE': 9,
    'LAYERSPACE': 4,
    'LAYERTITLE': True,
    'LAYERTITLESPACE': 4,
    'RULELABEL': True,
    'SYMBOLHEIGHT': 8,
    'SYMBOLSPACE': 2,
    'SYMBOLWIDTH': 8,
}


class Object(gws.common.ows.provider.Object, types.ProviderObject):
    def __init__(self):
        super().__init__()
        self.extent: t.Extent = ''
        self.legend_params = {}
        self.path = ''
        self.print_templates: t.List[types.PrintTemplate] = []
        self.properties: t.Dict = {}
        self.type = 'QGIS/WMS'
        self.version = '1.3.0'  # as of QGIS 3.4

    def configure(self):
        super().configure()

        self.legend_params = gws.extend(_LEGEND_DEFAULTS, self.root.var('server.qgis.legend'))

        self.path = self.var('path')
        self.url = 'http://%s:%s' % (
            self.root.var('server.qgis.host'),
            self.root.var('server.qgis.port'))

        with open(self.path) as fp:
            # @TODO qgz support
            s = fp.read()

        parser.parse(self, s)

    def find_features(self, args: t.SearchArguments) -> t.List[t.Feature]:
        # arbitrary width & height
        # @TODO: qgis scales the bbox for some reason?

        width = 1000
        height = 1000

        bbox = gws.gis.util.compute_bbox(
            args.point[0],
            args.point[1],
            self.supported_crs[0],
            args.resolution,
            width,
            height
        )

        p = {
            'BBOX': bbox,
            'CRS': self.supported_crs[0],

            'WIDTH': width,
            'HEIGHT': height,
            'I': width >> 1,
            'J': height >> 1,

            'INFO_FORMAT': 'text/xml',
            'LAYERS': args.layers,
            'MAP': self.path,
            'QUERY_LAYERS': args.layers,
            'STYLES': [''] * len(args.layers),

            'FI_LINE_TOLERANCE': 8,
            'FI_POINT_TOLERANCE': 16,
            'FI_POLYGON_TOLERANCE': 4,

            # see https://github.com/qgis/qwc2-demo-app/issues/55
            'WITH_GEOMETRY': 1,
        }

        if args.get('count'):
            p['FEATURE_COUNT'] = args.count

        p = gws.extend(p, args.get('params'))

        text = gws.gis.ows.request.get_text(self.url, service='WMS', request='GetFeatureInfo', params=p)
        return gws.gis.ows.response.parse(text, crs=self.supported_crs[0])

    def get_legend(self, source_layers):
        layers = ','.join(sl.name for sl in source_layers)
        params = gws.extend(self.legend_params, {
            'MAP': self.path,
            'LAYER': layers,
            'FORMAT': 'image/png',
            'STYLE': '',
            'VERSION': '1.1.1',
        })

        resp = gws.gis.ows.request.get(
            self.url,
            service='WMS',
            request='GetLegendGraphic',
            params=params)

        return resp.content
