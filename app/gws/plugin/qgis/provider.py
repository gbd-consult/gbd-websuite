"""QGIS project provider."""

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
    """QGIS project, served by QGIS Server."""
    
    path: Optional[gws.FilePath]
    """QGIS project file."""
    dbUid: Optional[str]
    """Database provider UID for projects stored in a database."""
    schema: Optional[str]
    """Database schema for projects stored in a database."""
    projectName: Optional[str]
    """Project name for projects stored in a database."""
    defaultLegendOptions: Optional[dict]
    """Legend options applied to all layers of this project."""
    directRender: Optional[list[str]]
    """Layer sources to load directly, not through QGIS Server."""
    directSearch: Optional[list[str]]
    """Layer sources to search directly, not through QGIS Server."""
    forceCrs: Optional[gws.CrsName]
    """CRS for QGIS Server requests."""
    extentBuffer: Optional[int]
    """Buffer around the extent computed from layer data."""
    useCanvasExtent: Optional[bool]
    """Use the map canvas extent when the project has no WMS extent."""
    withWatch: Optional[bool]
    """Reload the application when the QGIS project changes."""
    watchFrequency: Optional[gws.Duration]
    """Interval between checks for project changes."""


class Object(gws.OwsServiceProvider):
    """Provider for a QGIS project, served by QGIS Server.

    The provider loads and parses the project, computes the project bounds
    and sends requests to QGIS Server.
    """

    store: project.Store
    """Location of the project."""
    printTemplates: list[caps_module.PrintTemplate]
    """Print layouts of the project."""

    directRender: set[str]
    """Data source providers rendered directly, not through QGIS Server (``wms``, ``wmts``, ``xyz``)."""
    directSearch: set[str]
    """Data source providers searched directly, not through QGIS Server (``wms``, ``wfs``, ``postgres``)."""

    defaultLegendOptions: dict
    """Legend options applied to all layers of the project."""

    caps: caps_module.Caps
    """Parsed project capabilities."""
    sourceHash: str
    """Hash of the project XML when watching is enabled, otherwise empty."""

    def configure(self):
        self.configure_store()

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

        self.sourceHash = ''
        if self.cfg('withWatch'):
            self.sourceHash = self.qgis_project().sourceHash
            self.root.app.monitor.register_periodic_task(self, frequency=self.cfg('watchFrequency', default=0))
            gws.log.info(f'QGIS: monitoring: enabled: {self.server_project_path()!r}')

    def cache_hash(self):
        """Compute a hash of the provider settings that affect rendering.

        Returns:
            Hash string, built from the store and the request CRS.
        """
        return gws.u.sha256([
            vars(self.store),
            self.forceCrs.srid,
        ])

    def periodic_task(self):
        h = self.qgis_project().sourceHash
        if h != self.sourceHash:
            gws.log.info(f'QGIS: monitoring: CHANGED: {self.server_project_path()!r}')
            self.sourceHash = h
            self.root.app.monitor.schedule_reload(with_reconfigure=True)

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
        """Set the project store from ``path`` or ``projectName``.

        Raises:
            ``gws.ConfigurationError``: If neither ``path`` nor ``projectName`` is configured.
        """
        p = self.cfg('path')
        if p:
            pp = gws.lib.osx.parse_path(p)
            self.store = project.Store(
                type=project.StoreType.file,
                projectName=pp.stem,
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
        """Load the project from its store.

        Returns:
            A freshly loaded project.

        Raises:
            ``project.Error``: If the project cannot be loaded.
        """
        return project.from_store(self.root, self.store)

    def server_project_path(self):
        """Return the project address for the QGIS Server ``MAP`` parameter.

        For a file store this is the file path. For a Postgres store this is
        the database connection URL with ``schema`` and ``project`` parameters;
        when watching is enabled, the project hash is appended to invalidate
        the QGIS Server cache.

        Returns:
            Project path or URL.
        """
        if self.store.type == project.StoreType.file:
            return self.store.path
        if self.store.type == project.StoreType.postgres:
            prov = self.root.app.databaseMgr.find_provider(ext_type='postgres', uid=self.store.dbUid)
            p = {'schema': self.store.schema, 'project': self.store.projectName}
            if self.sourceHash:
                # NB the hash is to invalidate QGIS Server internal cache
                p['h'] = self.sourceHash
            return gws.lib.net.add_params(prov.url(), p)

    def server_params(self, params: dict) -> dict:
        """Add default parameters to a QGIS Server request.

        Args:
            params: Request parameters; keys are converted to upper case.

        Returns:
            Parameters with ``MAP``, ``SERVICE`` and ``VERSION`` defaults.
        """
        defaults = dict(
            MAP=self.server_project_path(),
            SERVICE=gws.OwsProtocol.WMS,
            VERSION='1.3.0',
        )
        return gws.u.merge(defaults, gws.u.to_upper_dict(params))

    def call_server(self, params: dict) -> gws.lib.net.HTTPResponse:
        """Send a request to QGIS Server.

        Args:
            params: Request parameters, completed by ``server_params``.

        Returns:
            HTTP response.

        Raises:
            ``gws.lib.net.Error``: If the request fails.
        """
        params = self.server_params(params)
        res = gws.lib.net.http_request(self.url, params=params, timeout=1000)
        res.raise_if_failed()
        return res

    ##

    def get_map(self, bounds: gws.Bounds, width: float, height: float, params: dict) -> bytes:
        """Render a map image with a GetMap request.

        Args:
            bounds: Box to render.
            width: Image width in pixels.
            height: Image height in pixels.
            params: Extra request parameters, e.g. ``LAYERS``; they override the defaults (transparent PNG).

        Returns:
            Image bytes.

        Raises:
            ``gws.Error``: If QGIS Server returns a non-image response.
        """
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
            gws.log.debug(f'cannot request crs {shape.crs!r}, unsupported unit')
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

        try:
            records = gws.base.ows.client.featureinfo.parse(res.text, default_crs=request_crs, always_xy=self.alwaysXY)
        except gws.Error as exc:
            gws.log.error(f'get_features: parse error: {exc!r}')
            return []

        gws.log.debug(f'get_features: FOUND={len(records)} params={params!r}')

        for rec in records:
            if rec.shape:
                rec.shape = rec.shape.transformed_to(shape.crs)

        return records

    ##

    def create_leaf_layer_config(self, source_layers):
        """Create the configuration of a leaf layer for the ``qgis`` tree layer.

        By default, this is a ``qgisflat`` layer. For a single source layer
        whose data source provider is listed in ``directRender`` or
        ``directSearch``, the layer is rendered directly (``wmsflat``,
        ``wmts`` or ``tile``) and gets its own finders and models.

        Args:
            source_layers: Source layers of the leaf.

        Returns:
            Layer configuration dict.
        """
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
        """Create a direct render layer configuration for a data source."""
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
        """Create direct finder and model configurations for a data source."""
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
        """Find or create a Postgres provider for a QGIS data source.

        An existing provider with the same connection URL is reused,
        otherwise a new provider is created.

        Args:
            ds: Parsed Postgres data source.

        Returns:
            Postgres provider.
        """
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
        """Add the non-standard OWS parameters of a data source to its URL."""
        if not url:
            return
        if not params:
            return url

        # a wms url can be like "server?service=WMS....&bbox=.... &some-non-std-param=...
        # we need to keep non-std params for caps requests

        p = {k: v for k, v in params.items() if k.lower() not in self._std_ows_params}
        return gws.lib.net.add_params(url, p)
