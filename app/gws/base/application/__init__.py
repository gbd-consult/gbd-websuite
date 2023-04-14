"""Core application object"""

import gws
import gws.base.action
import gws.base.auth
import gws.base.client
import gws.base.database
import gws.lib.metadata
import gws.base.project
import gws.base.web
import gws.config
import gws.gis.cache
import gws.gis.mpx.config
import gws.lib.font
import gws.lib.importer
import gws.lib.osx
import gws.server
import gws.server.monitor
import gws.spec
import gws.base.web.error
import gws.types as t

_DEFAULT_MIDDLEWARE = ['cors', 'auth']
_DEFAULT_LOCALE = ['en_CA']


class FontConfig(gws.Config):
    """Fonts configuration."""

    dir: gws.DirPath
    """directory with custom fonts"""


class Config(gws.ConfigWithAccess):
    """Main application configuration"""

    api: t.Optional[gws.base.action.manager.Config]
    """system-wide server actions"""
    auth: t.Optional[gws.base.auth.Config] = {}  # type: ignore
    """authorization methods and options"""
    cache: t.Optional[gws.gis.cache.Config] = {}  # type: ignore
    """global cache configuration"""
    client: t.Optional[gws.base.client.Config]
    """gws client configuration"""
    db: t.Optional[gws.base.database.manager.Config]
    """database configuration"""
    developer: t.Optional[dict]
    """developer options"""
    fonts: t.Optional[FontConfig]
    """fonts configuration"""
    helpers: t.Optional[list[gws.ext.config.helper]]
    """helpers configurations"""
    locales: t.Optional[list[str]]
    """default locales for all projects"""
    metadata: t.Optional[gws.Metadata]
    """application metadata"""
    middleware: t.Optional[list[str]]
    """middleware function names"""
    plugins: t.Optional[list[dict]]
    """configuration for plugins"""
    projectDirs: t.Optional[list[gws.DirPath]]
    """directories with additional projects"""
    projectPaths: t.Optional[list[gws.FilePath]]
    """additional project paths"""
    projects: t.Optional[list[gws.ext.config.project]]
    """project configurations"""
    server: t.Optional[gws.server.Config] = {}  # type: ignore
    """server engine options"""
    web: t.Optional[gws.base.web.manager.Config]
    """web server options"""


class Object(gws.Node, gws.IApplication):
    """Main Appilication object"""

    helpers: list[gws.Node]
    qgisVersion = ''
    projects: dict[str, gws.IProject]

    webMiddlewareFuncs: dict[str, t.Callable]
    webMiddlewareNames: list[str]

    _devopts: dict

    mpxUrl = ''
    mpxConfig = ''

    def configure(self):
        self._setenv('server.log.level', gws.env.GWS_LOG_LEVEL)
        self._setenv('server.web.workers', gws.env.GWS_WEB_WORKERS)
        self._setenv('server.spool.workers', gws.env.GWS_SPOOL_WORKERS)

        self.version = self.root.specs.version
        self.versionString = f'GWS version {self.version}'

        if self.cfg('server.qgis.enabled'):
            qgis_server = gws.lib.importer.import_from_path('gws/plugin/qgis/server.py')
            self.qgisVersion = qgis_server.version()
            if self.qgisVersion:
                self.versionString += f', QGis {self.qgisVersion}'

        gws.log.info('*' * 60)
        gws.log.info(self.versionString)
        gws.log.info('*' * 60)

        self._devopts = self.cfg('developer') or {}
        if self._devopts:
            gws.log.warning('developer mode enabled')

        self.webMiddlewareFuncs = {}
        self.webMiddlewareNames = self.cfg('middleware', default=_DEFAULT_MIDDLEWARE)

        self.localeUids = self.cfg('locales') or _DEFAULT_LOCALE
        self.monitor = self.create_child(gws.server.monitor.Object, self.cfg('server.monitor'))
        self.metadata = gws.lib.metadata.from_config(self.cfg('metadata'))

        p = self.cfg('fonts.dir')
        if p:
            gws.lib.font.install_fonts(p)

        # NB the order of initialization is important
        # - db
        # - helpers
        # - auth providers
        # - actions, client, web
        # - finally, projects

        self.databaseMgr = self.create_child(gws.base.database.manager.Object, self.cfg('db'))

        # # helpers are always created, no matter configured or not
        # cnf = {c.get('type'): c for c in self.cfg('helpers', default=[])}
        # for class_name in self.root.specs.real_class_names('gws.ext.helper'):
        #     desc = self.root.specs.object_descriptor(class_name)
        #     if desc.ext_type not in cnf:
        #         gws.log.debug(f'ad-hoc helper {desc.ext_type!r} will be created')
        #         cfg = gws.Config(type=desc.ext_type)
        #         cnf[desc.ext_type] = gws.config.parse(self.root.specs, cfg, 'gws.ext.config.helper')
        # self.helpers = self.root.create_many('gws.ext.helper', list(cnf.values()))

        self.authMgr = self.create_child(gws.base.auth.manager.Object, self.cfg('auth'))

        # @TODO default API
        self.actionMgr = self.create_child(gws.base.action.manager.Object, self.cfg('api'))

        self.webMgr = self.create_child(gws.base.web.manager.Object, self.cfg('web'))

        self.client = self.create_child(gws.base.client.Object, self.cfg('client'))

        projects = self.create_children(gws.ext.object.project, self.cfg('projects'))
        self.projects = {p.uid: p for p in projects}

    def post_configure(self):
        if self.cfg('server.mapproxy.enabled'):
            self.mpxUrl = f"http://{self.cfg('server.mapproxy.host')}:{self.cfg('server.mapproxy.port')}"
            self.mpxConfig = gws.gis.mpx.config.create_and_save(self.root)

        # for p in set(cfg.configPaths):
        #     root.app.monitor.add_path(p)
        # for p in set(cfg.projectPaths):
        #     root.app.monitor.add_path(p)
        # for d in set(cfg.projectDirs):
        #     root.app.monitor.add_directory(d, parser.config_path_pattern)
        #
        # if root.app.developer_option('server.auto_reload'):
        #     root.app.monitor.add_directory(gws.APP_DIR, '\.py$')

    def register_web_middleware(self, name, fn):
        self.webMiddlewareFuncs[name] = fn

    def web_middleware_list(self):
        return gws.compact(self.webMiddlewareFuncs.get(name) for name in self.webMiddlewareNames)

    def get_project(self, uid):
        return self.projects.get(uid)

    def require_helper(self, ext_type):
        for obj in self.helpers:
            if obj.ext_type == ext_type:
                return obj
        raise gws.Error(f'helper {ext_type!r} not found')

    def developer_option(self, name):
        return self._devopts.get(name)

    def _setenv(self, key, val):
        if not val:
            return
        ks = key.split('.')
        last = ks.pop()
        cfg = self.config
        for k in ks:
            if not hasattr(cfg, k):
                setattr(cfg, k, gws.Data())
            cfg = getattr(cfg, k)
        setattr(cfg, last, val)
        gws.log.info(f'environment: {key!r}={val!r}')
