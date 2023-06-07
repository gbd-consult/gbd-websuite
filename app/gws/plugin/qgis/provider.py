"""QGIS provider."""

import gws
import gws.lib.metadata
import gws.lib.mime
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
    dbUid: t.Optional[str]
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


class Object(gws.Node, gws.IOwsProvider):
    source: project.Source
    printTemplates: list[caps.PrintTemplate]
    url: str

    directRender: set[str]
    directSearch: set[str]

    bounds: t.Optional[gws.Bounds]

    caps: caps.Caps

    def configure(self):
        self.source = project.Source(
            path=self.cfg('path'),
            dbUid=self.cfg('dbUid'),
            schema=self.cfg('schema'),
            name=self.cfg('name'),
        )
        # self.root.app.monitor.add_path(self.source.path)

        self.url = 'http://%s:%s' % (
            self.root.app.cfg('server.qgis.host'),
            self.root.app.cfg('server.qgis.port'))

        self.caps = self.qgis_project().caps()

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

    def qgis_project(self) -> project.Object:
        return project.from_source(self.source, self)

    def server_project_path(self):
        # @TODO postgres
        return self.source.path

    def server_params(self, params: dict) -> dict:
        defaults = dict(
            MAP=self.server_project_path(),
            SERVICE=gws.OwsProtocol.WMS,
            VERSION='1.3.0',
        )
        return gws.merge(defaults, gws.to_upper_dict(params))

    def call_server(self, params: dict, max_age=0) -> gws.lib.net.HTTPResponse:
        params = self.server_params(params)
        res = gws.lib.net.http_request(self.url, params=params, max_age=max_age, timeout=1000)
        res.raise_if_failed()
        return res

    ##

    def get_map(self, layer: gws.ILayer, bounds: gws.Bounds, width: float, height: float, params: dict) -> bytes:
        defaults = dict(
            REQUEST=gws.OwsVerb.GetMap,
            BBOX=bounds.extent,
            WIDTH=gws.to_rounded_int(width),
            HEIGHT=gws.to_rounded_int(height),
            CRS=bounds.crs.epsg,
            FORMAT=gws.lib.mime.PNG,
            TRANSPARENT='true',
            STYLES='',
        )

        params = gws.merge(defaults, params)

        res = self.call_server(params)
        if res.content_type.startswith('image/'):
            return res.content
        raise gws.Error(res.text)

    def get_features(self, search, source_layers):
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

        layer_names = [sl.name for sl in source_layers]

        params = {
            'BBOX': bbox,
            'CRS': request_crs.to_string(gws.CrsFormat.epsg),
            'WIDTH': box_size_px,
            'HEIGHT': box_size_px,
            'I': box_size_px >> 1,
            'J': box_size_px >> 1,
            'LAYERS': layer_names,
            'QUERY_LAYERS': layer_names,
            'STYLES': [''] * len(layer_names),
            'FEATURE_COUNT': search.limit or 100,
            'INFO_FORMAT': 'text/xml',
            'REQUEST': gws.OwsVerb.GetFeatureInfo,
        }

        if search.extraParams:
            params = gws.merge(params, gws.to_upper_dict(search.extraParams))

        res = self.call_server(params)

        fdata = gws.gis.ows.featureinfo.parse(res.text, default_crs=request_crs, always_xy=self.alwaysXY)

        if fdata is None:
            gws.log.debug(f'get_features: NOT_PARSED params={params!r}')
            return []

        gws.log.debug(f'get_features: FOUND={len(fdata)} params={params!r}')

        for fd in fdata:
            if fd.shape:
                fd.shape = fd.shape.transformed_to(shape.crs)

        return fdata

    ##

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
                'provider': {
                    'url': self.make_ows_url(ds['url'], ds['params']),
                }
            }

        if prov == 'wmts':
            layers = ds.get('layers')
            if not layers:
                return
            return {
                'type': 'wmts',
                'sourceLayer': ds['layers'][0],
                'style': ds['styles'][0],
                'provider': {
                    'url': self.make_ows_url(ds['url'], ds['params']),
                }
            }

        if prov == 'xyz':
            return {
                'type': 'tile',
                'provider': {
                    'url': self.make_ows_url(ds['url'], ds['params']),
                }
            }

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


##


def get_for(obj: gws.INode) -> Object:
    p = obj.cfg('provider')
    if p:
        return obj.root.create_shared(Object, p)
    p = obj.cfg('_defaultProvider')
    if p:
        return p
    raise gws.Error(f'no provider found for {obj!r}')
