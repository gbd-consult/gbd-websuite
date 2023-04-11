"""QGIS provider."""

import gws
import gws.lib.metadata
import gws.gis.crs
import gws.gis.ows
import gws.gis.source
import gws.gis.extent
import gws.lib.net
import gws.types as t

from . import caps, project


class Config(gws.Config):
    path: t.Optional[gws.FilePath]
    """Qgis project file"""
    db: t.Optional[str]
    """Qgis project database"""
    schema: t.Optional[str]
    """Qgis project schema"""
    name: t.Optional[str]
    """Qgis project name"""
    directRender: t.Optional[list[str]]
    """QGIS data providers that should be rendered directly"""
    directSearch: t.Optional[list[str]]
    """QGIS data providers that should be searched directly"""
    forceCrs: t.Optional[gws.CrsName]
    """use this CRS for requests"""
    sqlFilters: t.Optional[dict]
    """per-layer sql filters"""


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
    source: project.Source
    project: project.Object
    printTemplates: list[caps.PrintTemplate]
    url: str

    directRender: set[str]
    directSearch: set[str]

    bounds: t.Optional[gws.Bounds]

    caps: caps.Caps

    def configure(self):
        self.source = project.Source(
            path=self.cfg('path'),
            db=self.cfg('db'),
            schema=self.cfg('schema'),
            name=self.cfg('name'),
        )
        # self.root.app.monitor.add_path(self.source.path)

        self.url = 'http://%s:%s' % (
            self.root.app.cfg('server.qgis.host'),
            self.root.app.cfg('server.qgis.port'))

        self.project = project.from_source(self.source, self)
        self.caps = self.project.caps()

        self.metadata = self.caps.metadata
        self.printTemplates = self.caps.printTemplates
        self.sourceLayers = self.caps.sourceLayers
        self.version = self.caps.version

        self.forceCrs = gws.gis.crs.get(self.cfg('forceCrs')) or self.caps.projectCrs
        self.alwaysXY = False

        self.directRender = set(self.cfg('directRender', default=[]))
        self.directSearch = set(self.cfg('directSearch', default=[]))

        self.bounds = None
        wms_extent = self.caps.properties.get('WMSExtent')
        if wms_extent:
            self.bounds = gws.Bounds(
                extent=gws.gis.extent.from_list([float(v) for v in wms_extent]),
                crs=self.caps.projectCrs)

    ##

    def call_server(self, params: dict) -> gws.lib.net.HTTPResponse:
        defaults = dict(
            # @TODO postgres
            MAP=self.source.path,
            SERVICE=gws.OwsProtocol.WMS,
            VERSION='1.3.0',
        )
        params = gws.merge(defaults, gws.to_upper_dict(params))
        res = gws.lib.net.http_request(self.url, params=params, timeout=1000)
        res.raise_if_failed()
        return res

    ##

    def get_feature_info(self, search, source_layers):
        v3 = self.version >= '1.3'

        shape = search.shape
        if not shape or shape.type != gws.GeometryType.point:
            return []

        request_crs = self.forceCrs
        if not request_crs:
            request_crs = gws.gis.crs.best_match(
                shape.crs,
                gws.gis.source.combined_crs_list(source_layers))

        box_size_m = 500
        box_size_deg = 1
        box_size_px = 500

        size = None

        if shape.crs.uom == gws.Uom.m:
            size = box_size_px * search.resolution
        if shape.crs.uom == gws.Uom.deg:
            # @TODO use search.resolution here as well
            size = box_size_deg
        if not size:
            gws.log.debug('cannot request crs {crs!r}, unsupported unit')
            return []

        bbox = (
            shape.x - (size / 2),
            shape.y - (size / 2),
            shape.x + (size / 2),
            shape.y + (size / 2),
        )

        bbox = gws.gis.extent.transform(bbox, shape.crs, request_crs)

        always_xy = self.alwaysXY or not v3
        if request_crs.isYX and not always_xy:
            bbox = gws.gis.extent.swap_xy(bbox)

        layer_names = [sl.name for sl in source_layers]

        params = {
            'BBOX': bbox,
            'CRS' if v3 else 'SRS': request_crs.to_string(gws.CrsFormat.epsg),
            'WIDTH': box_size_px,
            'HEIGHT': box_size_px,
            'I' if v3 else 'X': box_size_px >> 1,
            'J' if v3 else 'Y': box_size_px >> 1,
            'LAYERS': layer_names,
            'QUERY_LAYERS': layer_names,
            'STYLES': [''] * len(layer_names),
            'VERSION': self.version,
        }

        if search.limit:
            params['FEATURE_COUNT'] = search.limit

        params['INFO_FORMAT'] = 'text/xml'
        params['REQUEST'] = gws.OwsVerb.GetFeatureInfo

        if search.extraParams:
            params = gws.merge(params, gws.to_upper_dict(search.extraParams))

        res = self.call_server(params)

        fdata = gws.gis.ows.featureinfo.parse(
            res.text,
            default_crs=request_crs,
            always_xy=always_xy)

        if fdata is None:
            gws.log.debug(f'get_feature_info: NOT_PARSED params={params!r}')
            return []

        gws.log.debug(f'get_feature_info: FOUND={len(fdata)} params={params!r}')

        for fd in fdata:
            if fd.shape:
                fd.shape = fd.shape.transformed_to(shape.crs)

        return fdata

    ##

    def leaf_layer_config(self, source_layers):
        default = {
            'type': 'qgisflat',
            '_defaultProvider': self,
            '_defaultSourceLayers': source_layers
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

        gws.log.warning(f'directRender not supported for {prov!r}')
        return default

    def search_config(self, source_layers):
        default = {
            'type': 'qgislocal',
            '_defaultProvider': self,
            '_defaultSourceLayers': source_layers
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

        gws.log.warning(f'directSearch not supported for {prov!r}')
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


##


def get_for(obj: gws.INode) -> Object:
    p = obj.cfg('provider')
    if p:
        return obj.root.create_shared(Object, p)
    p = obj.cfg('_defaultProvider')
    if p:
        return p
    raise gws.Error(f'no provider found for {obj!r}')
