import zipfile

import gws
import gws.types as t
import gws.lib.gis
import gws.lib.ows
import gws.lib.net
import gws.lib.xml2
from . import types, parser


class Config(gws.Config):
    path: gws.FilePath  #: path to a Qgis project file
    directRender: t.Optional[t.List[str]]  #: QGIS data providers that should be rendered directly
    directSearch: t.Optional[t.List[str]]  #: QGIS data providers that should be searched directly


# see https://docs.qgis.org/2.18/en/docs/user_manual/working_with_ogc/ogc_server_support.html#getlegendgraphics-request

_DEFAULT_LEGEND_PARAMS = {
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


class Object(gws.Node):
    version = ''

    metadata: gws.IMetadata
    path: str
    print_templates: t.List[types.PrintTemplate]
    properties: dict
    source_layers: t.List[gws.lib.gis.SourceLayer]
    source_text: str
    supported_crs: t.List[gws.Crs]
    url: str

    direct_render: t.Set[str]
    direct_search: t.Set[str]

    def configure(self):
        self.path = self.var('path')
        self.url = 'http://%s:%s' % (
            self.root.application.var('server.qgis.host'),
            self.root.application.var('server.qgis.port'))

        self.source_text = self._read(self.path)
        cc = parser.parse(self.source_text)

        self.metadata = cc.metadata
        self.print_templates = cc.print_templates
        self.properties = cc.properties
        self.source_layers = cc.source_layers
        self.supported_crs = cc.supported_crs
        self.version = cc.version

        self.direct_render = set(self.var('directRender', default=[]))
        self.direct_search = set(self.var('directSearch', default=[]))

    def find_features(self, args: gws.SearchArgs) -> t.List[gws.IFeature]:
        if not args.shapes:
            return []

        our_crs = args.shapes[0].crs

        ps = gws.lib.gis.prepare_wms_search(
            args.shapes[0],
            protocol_version='1.3.0',
            force_crs=None,
            supported_crs=self.supported_crs,
            invert_axis_crs=None
        )

        if not ps:
            return []

        params = gws.merge(ps.params, {
            'INFO_FORMAT': 'text/xml',
            'LAYERS': args.source_layer_names,
            'MAP': self.path,
            'QUERY_LAYERS': args.source_layer_names,
            'STYLES': [''] * len(args.source_layer_names),

            # @TODO should be configurable

            'FI_LINE_TOLERANCE': 8,
            'FI_POINT_TOLERANCE': 16,
            'FI_POLYGON_TOLERANCE': 4,

            # see https://github.com/qgis/qwc2-demo-app/issues/55
            'WITH_GEOMETRY': 1,
        })

        if args.limit:
            params['FEATURE_COUNT'] = args.limit

        params = gws.merge(params, args.params)

        text = gws.lib.ows.request.get_text(self.url, gws.OwsProtocol.WMS, gws.OwsVerb.GetFeatureInfo, params=params)
        features = gws.lib.ows.formats.read(text, crs=ps.request_crs)

        if features is None:
            gws.log.debug(f'QGIS/WMS NOT_PARSED params={params!r}')
            return []

        gws.log.debug(f'QGIS/WMS FOUND={len(features)} params={params!r}')
        return [f.transform_to(our_crs) for f in features]

    def legend_url(self, source_layers, params=None):
        # qgis legends are rendered bottom-up (rightmost first)
        # we need the straight order (leftmost first), like in the config

        params = gws.merge(_DEFAULT_LEGEND_PARAMS, {
            'MAP': self.path,
            'LAYER': ','.join(sl.name for sl in reversed(source_layers)),
            'FORMAT': 'image/png',
            'TRANSPARENT': True,
            'STYLE': '',
            'VERSION': '1.1.1',
            'DPI': 96,
            'SERVICE': 'WMS',
            'REQUEST': gws.OwsVerb.GetLegendGraphic,
        }, params)

        return gws.lib.net.add_params(self.url, params)

    def leaf_config(self, source_layers):
        default = {
            'type': 'qgisflat',
            '_provider': self,
            '_source_layers': source_layers
        }

        if len(source_layers) > 1 or source_layers[0].is_group:
            return default

        sl = source_layers[0]
        ds = sl.data_source
        prov = ds.get('provider')

        if prov not in self.direct_render:
            return default

        if prov == 'wms':
            layers = ds.get('layers')
            if not layers:
                return
            return {
                'type': 'wmsflat',
                'sourceLayers': {
                    'names': ds['layers']
                },
                'url': self.make_wms_url(ds['url'], ds['params']),
            }

        if prov == 'wmts':
            layers = ds.get('layers')
            if not layers:
                return
            return gws.compact({
                'type': 'wmts',
                'url': ds['url'].split('?')[0],
                'sourceLayer': ds['layers'][0],
                'sourceStyle': (ds['options'] or {}).get('styles'),
            })

        gws.log.warn(f'directRender not supported for {prov!r}')
        return default

    def search_config(self, source_layers):
        default = {
            'type': 'qgiswms',
            '_provider': self,
            '_source_layers': source_layers,
        }

        if len(self.source_layers) > 1 or self.source_layers[0].is_group:
            return default

        sl = source_layers[0]
        ds = sl.data_source
        prov = ds.get('provider')

        if prov not in self.direct_search:
            return default

        if prov == 'wms':
            layers = ds.get('layers')
            if layers:
                return {
                    'type': 'wms',
                    'sourceLayers': {
                        'names': ds['layers']
                    },
                    'url': self.make_wms_url(ds['url'], ds['params']),
                }

        if prov == 'postgres':
            tab = sl.data_source.get('table')

            # 'table' can also be a select statement, in which case it might be enclosed in parens
            if not tab or tab.startswith('(') or tab.upper().startswith('SELECT '):
                return

            return {
                'type': 'qgispostgres',
                '_data_source': ds
            }

        if prov == 'wfs':
            cfg = {
                'type': 'wfs',
                'url': ds['url'],
            }
            if gws.get(ds, 'typeName'):
                cfg['sourceLayers'] = {
                    'names': [ds['typeName']]
                }
            crs = gws.get(ds, 'params.srsname')
            inv = gws.get(ds, 'params.InvertAxisOrientation')
            if inv == '1' and crs:
                cfg['invertAxis'] = [crs]

            return cfg

        gws.log.warn(f'directSearch not supported for {prov!r}')
        return default

    def make_wms_url(self, url, params):
        # a wms url can be like "server?service=WMS....&bbox=.... &some-non-std-param=...
        # we need to keep non-std params for caps requests

        _std_params = {
            'service',
            'version',
            'request',
            'layers',
            'styles',
            'srs',
            'crs',
            'bbox',
            'width',
            'height',
            'format',
            'transparent',
            'bgcolor',
            'exceptions',
            'time',
            'sld',
            'sld_body',
        }
        p = {k: v for k, v in params.items() if k.lower() not in _std_params}
        return gws.lib.net.add_params(url, p)

    def print_template(self, ref: str):
        pts = self.print_templates

        if not self.print_templates:
            return

        if not ref:
            return pts[0]

        if ref.isdigit() and int(ref) < len(pts):
            return pts[int(ref)]

        for tpl in pts:
            if tpl.title == ref:
                return tpl

    def _read(self, path):
        if not path.endswith('.qgz'):
            return gws.read_file(path)

        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.filename.endswith('.qgs'):
                    with zf.open(info, 'rt') as fp:
                        return fp.read()


##


def create_from_path(root: gws.IRoot, path: str, parent: gws.Node = None, shared: bool = False) -> Object:
    return create(root, gws.Config(path=path), parent, shared)


def create(root: gws.IRoot, cfg: gws.Config, parent: gws.Node = None, shared: bool = False) -> Object:
    path = cfg.get('path')
    root.application.monitor.add_path(path)
    return root.create_object(Object, cfg, parent, shared, key=path)
