"""QGIS provider."""

import gws
import gws.base.database
import gws.base.ows.client
import gws.config.util
import gws.plugin.postgres.provider
import gws.lib.metadata
import gws.lib.net
import gws.lib.mime
import gws.gis.crs
import gws.gis.bounds
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
    projectName: t.Optional[str]
    """Qgis project name"""
    directRender: t.Optional[list[str]]
    """QGIS data providers that should be rendered directly"""
    directSearch: t.Optional[list[str]]
    """QGIS data providers that should be searched directly"""
    forceCrs: t.Optional[gws.CrsName]
    """use this CRS for requests"""
    sqlFilters: t.Optional[dict]
    """per-layer sql filters"""


class Object(gws.OwsProvider):
    store: project.Store
    printTemplates: list[caps.PrintTemplate]
    url: str

    directRender: set[str]
    directSearch: set[str]

    bounds: t.Optional[gws.Bounds]

    caps: caps.Caps

    def configure(self):
        self.configure_store()
        # if self.store.path:
        #     self.root.app.monitor.add_file(self.store.path)

        self.url = 'http://{}:{}'.format(
            self.root.app.cfg('server.qgis.host'),
            self.root.app.cfg('server.qgis.port'))

        self.caps = self.qgis_project().caps()

        self.metadata = self.caps.metadata
        self.printTemplates = self.caps.printTemplates
        self.sourceLayers = self.caps.sourceLayers
        self.version = self.caps.version

        self.forceCrs = gws.gis.crs.get(self.cfg('forceCrs')) or self.caps.projectBounds.crs
        self.alwaysXY = False

        self.bounds = self.caps.projectBounds

        self.directRender = set(self.cfg('directRender', default=[]))
        self.directSearch = set(self.cfg('directSearch', default=[]))

    def configure_store(self):
        p = self.cfg('path')
        if p:
            self.store = project.Store(
                type=project.StoreType.file,
                path=p,
            )
            return
        p = self.cfg('projectName')
        if p:
            self.store = project.Store(
                type=project.StoreType.postgres,
                projectName=p,
                dbUid=self.cfg('dbUid'),
                schema=self.cfg('schema') or 'public',
            )
            return
        # @TODO gpkg, etc
        raise gws.Error('cannot load qgis project ("path" or "projectName" must be specified)')

    ##

    def qgis_project(self) -> project.Object:
        return project.from_store(self.root, self.store)

    def server_project_path(self):
        if self.store.type == project.StoreType.file:
            return self.store.path
        if self.store.type == project.StoreType.postgres:
            prov = gws.base.database.provider.get_for(self, self.store.dbUid, 'postgres')
            return gws.lib.net.add_params(prov.url, schema=self.store.schema, project=self.store.projectName)

    def server_params(self, params: dict) -> dict:
        defaults = dict(
            MAP=self.server_project_path(),
            SERVICE=gws.OwsProtocol.WMS,
            VERSION='1.3.0',
        )
        return gws.u.merge(defaults, gws.u.to_upper_dict(params))

    def call_server(self, params: dict, max_age=0) -> gws.lib.net.HTTPResponse:
        params = self.server_params(params)
        res = gws.lib.net.http_request(self.url, params=params, max_age=max_age, timeout=1000)
        res.raise_if_failed()
        return res

    ##

    def get_map(self, layer: gws.Layer, bounds: gws.Bounds, width: float, height: float, params: dict) -> bytes:
        defaults = dict(
            REQUEST=gws.OwsVerb.GetMap,
            BBOX=bounds.extent,
            WIDTH=gws.u.to_rounded_int(width),
            HEIGHT=gws.u.to_rounded_int(height),
            CRS=bounds.crs.epsg,
            FORMAT=gws.lib.mime.PNG,
            TRANSPARENT='true',
            STYLES='',
        )

        params = gws.u.merge(defaults, params)

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
            params = gws.u.merge(params, gws.u.to_upper_dict(search.extraParams))

        res = self.call_server(params)

        fdata = gws.base.ows.client.featureinfo.parse(res.text, default_crs=request_crs, always_xy=self.alwaysXY)

        if fdata is None:
            gws.log.debug(f'get_features: NOT_PARSED params={params!r}')
            return []

        gws.log.debug(f'get_features: FOUND={len(fdata)} params={params!r}')

        for fd in fdata:
            if fd.shape:
                fd.shape = fd.shape.transformed_to(shape.crs)

        return fdata

    ##

    def leaf_config(self, source_layers):
        cfg = self._leaf_base_config(source_layers)
        if not cfg:
            return
        finder = self._leaf_finder_config(source_layers)
        if finder:
            cfg['finders'] = [finder]
        return cfg

    def _leaf_base_config(self, source_layers):
        default = {
            'type': 'qgisflat',
            '_defaultProvider': self,
            '_defaultSourceLayers': source_layers,
        }

        if len(source_layers) > 1 or source_layers[0].isGroup:
            return default

        ds = source_layers[0].dataSource
        if not ds:
            return default

        prov = ds.get('provider')
        if prov not in self.directRender:
            return default

        if prov == 'wms':
            return self._leaf_direct_render_wms(ds)
        if prov == 'wmts':
            return self._leaf_direct_render_wmts(ds)
        if prov == 'xyz':
            return self._leaf_direct_render_xyz(ds)

        gws.log.warning(f'directRender not supported for {prov!r}')
        return default

    def _leaf_direct_render_wms(self, ds):
        layers = ds.get('layers')
        url = self._leaf_service_url(ds.get('url'), ds.get('params'))
        if not layers or not url:
            return
        return {
            'type': 'wmsflat',
            'sourceLayers': {'names': layers},
            'display': 'tile',
            'provider': {'url': url},
        }

    def _leaf_direct_render_wmts(self, ds):
        layers = ds.get('layers')
        url = self._leaf_service_url(ds.get('url'), ds.get('params'))
        if not layers or not url:
            return
        cfg = {
            'type': 'wmts',
            'sourceLayer': layers[0],
            'provider': {'url': url},
        }
        p = ds.get('styles')
        if p:
            cfg['style'] = p[0]
        return cfg

    def _leaf_direct_render_xyz(self, ds):
        url = self._leaf_service_url(ds.get('url'), ds.get('params'))
        if not url:
            return
        return {
            'type': 'tile',
            'provider': {'url': url},
        }

    def _leaf_finder_config(self, source_layers):
        if len(source_layers) > 1 or source_layers[0].isGroup:
            return

        ds = source_layers[0].dataSource
        if not ds:
            return

        prov = ds.get('provider')
        if prov not in self.directSearch:
            return

        if prov == 'wms':
            return self._leaf_direct_search_wms(ds)
        if prov == 'wfs':
            return self._leaf_direct_search_wfs(ds)
        if prov == 'postgres':
            return self._leaf_direct_search_postgres(ds)

        gws.log.warning(f'directSearch not supported for {prov!r}')

    def _leaf_direct_search_wms(self, ds):
        layers = ds.get('layers')
        url = self._leaf_service_url(ds.get('url'), ds.get('params'))
        if not layers or not url:
            return
        return {
            'type': 'wms',
            'sourceLayers': {'names': layers},
            'provider': {'url': url}
        }

    def _leaf_direct_search_wfs(self, ds):
        url = self._leaf_service_url(ds.get('url'), ds.get('params'))
        if not url:
            return
        cfg = {
            'type': 'wfs',
            'provider': {'url': url},
        }
        p = ds.get('typename')
        if p:
            cfg['sourceLayers'] = {'names': [p]}
        p = ds.get('srsname')
        if p:
            cfg['forceCrs'] = p
        p = ds.get('ignoreaxisorientation')
        if p == '1':
            cfg['alwaysXY'] = True
        p = ds.get('invertaxisorientation')
        if p == '1':
            # NB assuming this might be only '1' for lat-lon projections
            cfg['alwaysXY'] = True

        return cfg

    def _leaf_direct_search_postgres(self, ds):
        table_name = ds.get('table')

        # 'table' can also be a select statement, in which case it might be enclosed in parens
        if not table_name or table_name.startswith('(') or table_name.upper().startswith('SELECT '):
            return

        # @TODO support extra sql from ds['sql']

        pg_provider = self.postgres_provider_from_datasource(ds)

        return {
            'type': 'postgres',
            'tableName': table_name,
            'models': [{'type': 'postgres'}],
            '_defaultProvider': pg_provider
        }

    def postgres_provider_from_datasource(self, ds: dict) -> gws.plugin.postgres.provider.Object:
        cfg = gws.Config(
            host=ds.get('host'),
            port=ds.get('port'),
            database=ds.get('dbname'),
            username=ds.get('user'),
            password=ds.get('password'),
            serviceName=ds.get('service'),
            options=ds.get('options'),
        )
        url = gws.plugin.postgres.provider.connection_url(cfg)
        mgr = self.root.app.databaseMgr

        for p in mgr.providers():
            if p.extType == 'postgres' and p.url == url:
                return t.cast(gws.plugin.postgres.provider.Object, p)

        gws.log.debug(f'creating an ad-hoc postgres provider for qgis {url=}')
        p = mgr.create_provider(cfg, type='postgres')
        return t.cast(gws.plugin.postgres.provider.Object, p)

    _std_ows_params = {
        'bbox',
        'bgcolor',
        'crs',
        'exceptions',
        'format',
        'height',
        'layers',
        'request',
        'service',
        'sld',
        'sld_body',
        'srs',
        'styles',
        'time',
        'transparent',
        'version',
        'width',
    }

    def _leaf_service_url(self, url, params):
        if not url:
            return
        if not params:
            return url

        # a wms url can be like "server?service=WMS....&bbox=.... &some-non-std-param=...
        # we need to keep non-std params for caps requests

        p = {k: v for k, v in params.items() if k.lower() not in self._std_ows_params}
        return gws.lib.net.add_params(url, p)


##


def get_for(obj: gws.Node) -> Object:
    return t.cast(Object, gws.config.util.get_provider(Object, obj))
