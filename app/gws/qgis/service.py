import gws
import gws.types as t
import gws.ows.request
import gws.config
import gws.tools.xml3
import gws.tools.net
import gws.ows.response
import gws.gis.util

from . import types, parser


def shared_service(obj, cfg):
    path = cfg.get('path')
    uid = path
    return obj.create_shared_object(Service, uid, t.Config({
        'path': path,
    }))


class Service(gws.Object, t.ServiceInterface):
    def __init__(self):
        super().__init__()
        self.type = 'QGIS'
        self.print_templates: t.List[types.PrintTemplate] = []
        self.extent: t.Extent = ''
        self.path = ''
        self.properties: t.Dict = {}

    def configure(self):
        super().configure()

        self.path = self.var('path')
        self.url = 'http://%s:%s' % (
            self.root.var('server.qgis.host'),
            self.root.var('server.qgis.port'))

        with open(self.path) as fp:
            # @TODO qgz support
            s = fp.read()

        parser.parse(self, s)

    def find_features(self, args: t.FindFeaturesArgs):
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

        text = gws.ows.request.get_text(self.url, service='WMS', request='GetFeatureInfo', params=p)
        return gws.ows.response.parse(text, crs=self.supported_crs[0])

    def get_legend(self, source_layers):
        # see https://docs.qgis.org/2.18/en/docs/user_manual/working_with_ogc/ogc_server_support.html#getlegendgraphics-request
        layers = ','.join(sl.name for sl in source_layers)
        params = {
            'MAP': self.path,
            'LAYER': layers,
            'FORMAT': 'image/png',
            'STYLE': '',
            'VERSION': '1.1.1',
            'BOXSPACE': 0,
            'SYMBOLSPACE': 0,
            'LAYERTITLE': 'false',
            # 'RULELABEL': 'false',
        }

        resp = gws.ows.request.get(
            self.url,
            service='WMS',
            request='GetLegendGraphic',
            params=params)

        return resp.content
