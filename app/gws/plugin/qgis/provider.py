import zipfile

import gws
import gws.lib.metadata
import gws.gis.crs
import gws.gis.extent
import gws.lib.net
import gws.types as t

from . import caps, project


class Config(gws.Config):
    path: gws.FilePath 
    """path to a Qgis project file"""
    directRender: t.Optional[t.List[str]] 
    """QGIS data providers that should be rendered directly"""
    directSearch: t.Optional[t.List[str]] 
    """QGIS data providers that should be searched directly"""
    forceCrs: t.Optional[gws.CrsName] 
    """use this CRS for requests"""


# see https://docs.qgis.org/3.22/de/docs/server_manual/services/wms.html#getlegendgraphics

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


class Object(gws.Node, gws.IOwsProvider):
    path: str
    printTemplates: t.List[caps.PrintTemplate]
    url: str
    project: project.Object

    directRender: t.Set[str]
    directSearch: t.Set[str]

    crs: gws.ICrs
    extent: t.Optional[gws.Extent]

    caps: caps.Caps

    def configure(self):
        self.path = self.var('path')
        self.root.app.monitor.add_path(self.path)

        self.url = 'http://%s:%s' % (
            self.root.app.var('server.qgis.host'),
            self.root.app.var('server.qgis.port'))

        self.project = project.from_path(self.path)
        self.caps = self.project.caps()

        self.metadata = self.caps.metadata
        self.printTemplates = self.caps.printTemplates
        self.sourceLayers = self.caps.sourceLayers
        self.version = self.caps.version

        self.crs = gws.gis.crs.get(self.var('forceCrs')) or self.caps.projectCrs

        self.directRender = set(self.var('directRender', default=[]))
        self.directSearch = set(self.var('directSearch', default=[]))

        self.extent = None
        wms_extent = self.caps.properties.get('WMSExtent')
        if wms_extent:
            self.extent = gws.gis.extent.from_list([float(v) for v in wms_extent])

    def find_features(self, args, source_layers):
        if not args.shapes:
            return []

        shape = args.shapes[0]
        if shape.type != gws.GeometryType.point:
            return []

        ps = gws.gis.ows.client.prepared_search(
            limit=args.limit,
            point=shape,
            protocol=gws.OwsProtocol.WMS,
            protocol_version='1.3.0',
            request_crs=self.force_crs,
            request_crs_format=gws.CrsFormat.EPSG,
            source_layers=source_layers,
        )

        qgis_defaults = {
            'INFO_FORMAT': 'text/xml',
            'MAP': self.path,

            # @TODO should be configurable

            'FI_LINE_TOLERANCE': 8,
            'FI_POINT_TOLERANCE': 16,
            'FI_POLYGON_TOLERANCE': 4,

            # see https://github.com/qgis/qwc2-demo-app/issues/55
            'WITH_GEOMETRY': 1,
        }

        params = gws.merge(qgis_defaults, ps.params, args.params)

        text = gws.gis.ows.request.get_text(self.url, gws.OwsProtocol.WMS, gws.OwsVerb.GetFeatureInfo, params=params)
        features = []  # gws.gis.ows.formats.read(text, crs=ps.request_crs)

        if features is None:
            gws.log.debug(f'QGIS/WMS NOT_PARSED params={params!r}')
            return []

        gws.log.debug(f'QGIS/WMS FOUND={len(features)} params={params!r}')
        return [f.transform_to(shape.crs) for f in features]

    def legendUrl(self, source_layers, params=None):
        # qgis legends are rendered bottom-up (rightmost first)
        # we need the straight order (leftmost first), like in the config

        std_params = gws.merge(_DEFAULT_LEGEND_PARAMS, {
            'MAP': self.path,
            'LAYER': ','.join(sl.name for sl in reversed(source_layers)),
            'FORMAT': 'image/png',
            'TRANSPARENT': True,
            'STYLE': '',
            'VERSION': '1.1.1',
            'DPI': 96,
            'SERVICE': gws.OwsProtocol.WMS,
            'REQUEST': gws.OwsVerb.GetLegendGraphic,
        })

        return gws.lib.net.add_params(self.url, gws.merge(std_params, params))

    def leaf_layer_config(self, source_layers):
        default = {
            'type': 'qgisflat',
            '_provider': self,
            '_sourceLayers': source_layers
        }

        if len(source_layers) > 1 or source_layers[0].isGroup:
            return default

        sl = source_layers[0]
        ds = sl.dataSource
        prov = ds.get('provider')

        if prov not in self.directRender:
            return default

        if prov == 'wms':
            layers = ds.get('layers')
            if not layers:
                return
            return {
                'type': 'wmsflat',
                'sourceLayers': {'names': ds['layers']},
                'url': self.make_ows_url(ds['url'], ds['params']),
            }

        if prov == 'wmts':
            layers = ds.get('layers')
            if not layers:
                return
            return gws.compact({
                'type': 'wmts',
                'sourceLayer': ds['layers'][0],
                'sourceStyle': (ds['options'] or {}).get('styles'),
                'url': self.make_ows_url(ds['url'], ds['params']),
            })

        # @TODO xyz

        gws.log.warn(f'directRender not supported for {prov!r}')
        return default

    def search_config(self, source_layers):
        default = {
            'type': 'qgiswms',
            '_provider': self,
            '_source_layers': source_layers,
        }

        if len(self.source_layers) > 1 or self.source_layers[0].isGroup:
            return default

        sl = source_layers[0]
        ds = sl.dataSource
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
            tab = sl.dataSource.get('table')

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

    _std_ows_params = {
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

    def make_ows_url(self, url, params):
        # a wms url can be like "server?service=WMS....&bbox=.... &some-non-std-param=...
        # we need to keep non-std params for caps requests

        p = {k: v for k, v in params.items() if k.lower() not in self._std_ows_params}
        return gws.lib.net.add_params(url, p)

    def print_template(self, ref: str) -> t.Optional[caps.PrintTemplate]:
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
