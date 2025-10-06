"""QGIS provider."""

from typing import Optional, cast

import gws
import gws.base.database
import gws.base.ows.client
import gws.config.util
import gws.plugin.postgres.provider
import gws.base.metadata
import gws.lib.net
import gws.lib.mime
import gws.lib.osx
import gws.lib.crs
import gws.lib.bounds
import gws.gis.source
import gws.lib.extent
import gws.lib.net

from . import caps as caps_module, project


class Config(gws.Config):
    """QGIS provider configuration."""
    
    path: Optional[gws.FilePath]
    """Qgis project file."""
    dbUid: Optional[str]
    """Qgis project database."""
    schema: Optional[str]
    """Qgis project schema."""
    projectName: Optional[str]
    """Qgis project name."""
    defaultLegendOptions: Optional[dict]
    """Default options for qgis legends.."""
    directRender: Optional[list[str]]
    """Qgis data providers that should be rendered directly."""
    directSearch: Optional[list[str]]
    """Qgis data providers that should be searched directly."""
    forceCrs: Optional[gws.CrsName]
    """Use this CRS for requests."""
    extentBuffer: Optional[int]
    """Extent buffer for automatically computed bounds.."""
    useCanvasExtent: Optional[bool]
    """Use canvas extent as project extent.."""


class Object(gws.OwsProvider):
    store: project.Store
    printTemplates: list[caps_module.PrintTemplate]

    directRender: set[str]
    directSearch: set[str]

    defaultLegendOptions: dict

    caps: caps_module.Caps

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

        self.forceCrs = gws.lib.crs.get(self.cfg('forceCrs')) or self.caps.projectCrs
        self.alwaysXY = False

        self.bounds = self._project_bounds()
        self.wgsExtent = gws.lib.bounds.wgs_extent(self.bounds)

        self.directRender = self._direct_formats('directRender', {'wms', 'wmts', 'xyz'})
        self.directSearch = self._direct_formats('directSearch', {'wms', 'wfs', 'postgres'})

        self.defaultLegendOptions = self.cfg('defaultLegendOptions', default={})

    def _project_bounds(self):
        # explicit WMS extent?
        if self.caps.projectBounds:
            return gws.lib.bounds.transform(self.caps.projectBounds, self.forceCrs)

        # canvas extent?
        if self.cfg('useCanvasExtent') and self.caps.projectCanvasBounds:
            return gws.lib.bounds.transform(self.caps.projectCanvasBounds, self.forceCrs)

        # combined data extents + buffer
        b = gws.gis.source.combined_bounds(self.sourceLayers, self.forceCrs)
        if b:
            return gws.lib.bounds.buffer(b, self.cfg('extentBuffer') or 0)

        return self.forceCrs.bounds

    def _direct_formats(self, opt, allowed):
        p = self.cfg(opt)
        if not p:
            return set()

        res = set()

        for s in p:
            s = s.lower()
            if s not in allowed:
                raise gws.ConfigurationError(f'{opt} not supported for {s!r}')
            res.add(s)

        return res

    def configure_store(self):
        p = self.cfg('path')
        if p:
            pp = gws.lib.osx.parse_path(p)
            self.store = project.Store(
                type=project.StoreType.file,
                projectName=pp['name'],
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
        raise gws.ConfigurationError('cannot load qgis project ("path" or "projectName" must be specified)')

    ##

    def qgis_project(self) -> project.Object:
        return project.from_store(self.root, self.store)

    def server_project_path(self):
        if self.store.type == project.StoreType.file:
            return self.store.path
        if self.store.type == project.StoreType.postgres:
            prov = self.root.app.databaseMgr.find_provider(ext_type='postgres', uid=self.store.dbUid)
            return gws.lib.net.add_params(prov.url(), schema=self.store.schema, project=self.store.projectName)

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
        bbox = bounds.extent
        if bounds.crs.isYX and not self.alwaysXY:
            bbox = gws.lib.extent.swap_xy(bbox)

        defaults = dict(
            REQUEST=gws.OwsVerb.GetMap,
            BBOX=bbox,
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
            request_crs = gws.lib.crs.best_match(
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

        bbox = gws.lib.extent.transform(bbox, shape.crs, request_crs)

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
            'WITH_GEOMETRY': 'true',
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
        simple_cfg = {
            'type': 'qgisflat',
            '_defaultProvider': self,
            '_defaultSourceLayers': source_layers,
        }

        if len(source_layers) > 1 or source_layers[0].isGroup:
            return simple_cfg

        ds = source_layers[0].dataSource
        if not ds or not ds.get('provider'):
            return simple_cfg

        render = self._leaf_render_config(ds)
        search = self._leaf_search_config(ds)

        cfg = {}
        cfg.update(render or {})
        cfg.update(search or {})

        if not cfg.get('type'):
            cfg.update(simple_cfg)

        return cfg

    def _leaf_render_config(self, ds):
        prov = ds.get('provider')
        if prov not in self.directRender:
            return

        url = self._leaf_service_url(ds.get('url'), ds.get('params'))
        if not url:
            return

        if prov == 'wms':
            layers = ds.get('layers')
            if not layers:
                return
            return {
                'type': 'wmsflat',
                'sourceLayers': {'names': layers},
                'display': 'tile',
                'provider': {'url': url},
            }

        if prov == 'wmts':
            layers = ds.get('layers')
            if not layers:
                return
            cfg = {
                'type': 'wmts',
                'sourceLayers': {'names': layers},
                'display': 'tile',
                'provider': {'url': url},
            }
            p = ds.get('styles')
            if p:
                cfg['style'] = p[0]
            return cfg

        if prov == 'xyz':
            return {
                'type': 'tile',
                'provider': {'url': url},
            }

    def _leaf_search_config(self, ds):
        prov = ds.get('provider')
        if prov not in self.directSearch:
            return

        if prov == 'wms':
            url = self._leaf_service_url(ds.get('url'), ds.get('params'))
            layers = ds.get('layers')
            if not url or not layers:
                return
            finder = {
                'type': 'wms',
                'provider': {'url': url},
                'sourceLayers': {'names': layers},
            }
            return {'finders': [finder]}

        if prov == 'wfs':
            url = self._leaf_service_url(ds.get('url'), ds.get('params'))
            if not url:
                return
            finder = {
                'type': 'wfs',
                'provider': {'url': url},
            }
            p = ds.get('typename')
            if p:
                finder['sourceLayers'] = {'names': [p]}
            p = ds.get('srsname')
            if p:
                finder['forceCrs'] = p
            p = ds.get('ignoreaxisorientation')
            if p == '1':
                finder['alwaysXY'] = True
            p = ds.get('invertaxisorientation')
            if p == '1':
                # NB assuming this might be only '1' for lat-lon projections
                finder['alwaysXY'] = True

            return {'finders': [finder]}

        if prov == 'postgres':
            table_name = ds.get('table')

            # 'table' can also be a select statement, in which case it might be enclosed in parens
            if not table_name or table_name.startswith('(') or table_name.upper().startswith('SELECT '):
                return

            db = self.postgres_provider_from_datasource(ds)

            model = {
                'type': 'postgres',
                'tableName': table_name,
                'sqlFilter': ds.get('sql'),
                '_defaultDb': db
            }
            finder = {
                'type': 'postgres',
                'tableName': table_name,
                'sqlFilter': ds.get('sql'),
                '_defaultDb': db
            }
            return {'models': [model], 'finders': [finder]}

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

        for p in mgr.providers:
            if p.extType == 'postgres' and p.url() == url:
                return cast(gws.plugin.postgres.provider.Object, p)

        gws.log.debug(f'creating an ad-hoc postgres provider for qgis {url=}')
        p = mgr.create_provider(cfg, type='postgres')
        return cast(gws.plugin.postgres.provider.Object, p)

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
