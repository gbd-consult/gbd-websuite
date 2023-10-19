"""Core application object"""

import gws
import gws.base.action
import gws.base.auth
import gws.base.client
import gws.base.database
import gws.base.model
import gws.base.printer
import gws.base.project
import gws.base.search
import gws.base.storage
import gws.base.template
import gws.base.web
import gws.config
import gws.gis.cache
import gws.gis.mpx.config
import gws.lib.font
import gws.lib.importer
import gws.lib.metadata
import gws.lib.osx
import gws.server
import gws.server.monitor
import gws.spec
import gws.types as t
from . import middleware

_DEFAULT_LOCALE = ['en_CA']

_DEFAULT_TEMPLATES = [
    gws.Config(
        type='html',
        path=gws.dirname(__file__) + '/templates/project_description.cx.html',
        subject='project.description',
        access=gws.PUBLIC,
        uid='default_template.project_description',
    ),
    gws.Config(
        type='html',
        path=gws.dirname(__file__) + '/templates/layer_description.cx.html',
        subject='layer.description',
        access=gws.PUBLIC,
        uid='default_template.layer_description',
    ),
    gws.Config(
        type='html',
        path=gws.dirname(__file__) + '/templates/feature_description.cx.html',
        subject='feature.description',
        access=gws.PUBLIC,
        uid='default_template.feature_description',
    ),
    gws.Config(
        type='html',
        path=gws.dirname(__file__) + '/templates/feature_title.cx.html',
        subject='feature.title',
        access=gws.PUBLIC,
        uid='default_template.feature_title',
    ),
    gws.Config(
        type='html',
        path=gws.dirname(__file__) + '/templates/feature_label.cx.html',
        subject='feature.label',
        access=gws.PUBLIC,
        uid='default_template.feature_label',
    ),
]

_DEFAULT_PRINTER = gws.Config(
    templates=[
        gws.Config(
            uid='gws.base.project.templates.project_print',
            type='html',
            path=gws.dirname(__file__) + '/templates/project_print.cx.html',
            mapSize=(200, 180, gws.Uom.mm),
            qualityLevels=[{'dpi': 72}],
            access=gws.PUBLIC,
        ),
    ]
)


class Config(gws.ConfigWithAccess):
    """Main application configuration"""

    api: t.Optional[gws.base.action.manager.Config]
    """system-wide server actions"""
    auth: t.Optional[gws.base.auth.manager.Config] = {}  # type: ignore
    """authorization methods and options"""
    cache: t.Optional[gws.gis.cache.Config] = {}  # type: ignore
    """global cache configuration"""
    client: t.Optional[gws.base.client.Config]
    """gws client configuration"""
    database: t.Optional[gws.base.database.manager.Config]
    """database configuration"""
    developer: t.Optional[dict]
    """developer options"""
    finders: t.Optional[list[gws.ext.config.finder]]
    """global search providers"""
    fonts: t.Optional[gws.lib.font.Config]
    """fonts configuration"""
    helpers: t.Optional[list[gws.ext.config.helper]]
    """helpers configurations"""
    locales: t.Optional[list[str]]
    """default locales for all projects"""
    metadata: t.Optional[gws.Metadata]
    """application metadata"""
    models: t.Optional[list[gws.ext.config.model]]
    """global data models"""
    owsServices: t.Optional[list[gws.ext.config.owsService]]
    """OWS services configuration"""
    plugins: t.Optional[list[dict]]
    """configuration for plugins"""
    projectDirs: t.Optional[list[gws.DirPath]]
    """directories with additional projects"""
    projectPaths: t.Optional[list[gws.FilePath]]
    """additional project paths"""
    printer: t.Optional[gws.base.printer.Config]
    """print configuration"""
    projects: t.Optional[list[gws.ext.config.project]]
    """project configurations"""
    server: t.Optional[gws.server.Config] = {}  # type: ignore
    """server engine options"""
    storage: t.Optional[gws.base.storage.manager.Config]
    """database configuration"""
    templates: t.Optional[list[gws.ext.config.template]]
    """default templates"""
    web: t.Optional[gws.base.web.manager.Config]
    """web server options"""


class Object(gws.Node, gws.IApplication):
    """Main Application object"""

    qgisVersion = ''
    projectMap: dict[str, gws.IProject]
    helperMap: dict[str, gws.INode]

    _devopts: dict

    mpxUrl = ''
    mpxConfig = ''

    middlewareMgr: middleware.Manager

    def configure(self):
        self._setenv('server.log.level', gws.env.GWS_LOG_LEVEL)
        self._setenv('server.web.workers', gws.env.GWS_WEB_WORKERS)
        self._setenv('server.spool.workers', gws.env.GWS_SPOOL_WORKERS)

        self.version = self.root.specs.version
        self.versionString = f'GWS version {self.version}'

        # if self.cfg('server.qgis.enabled'):
        #     qgis_server = gws.lib.importer.import_from_path('gws/plugin/qgis/server.py')
        #     self.qgisVersion = qgis_server.version()
        #     if self.qgisVersion:
        #         self.versionString += f', QGis {self.qgisVersion}'

        gws.log.info('*' * 60)
        gws.log.info(self.versionString)
        gws.log.info('*' * 60)

        self._devopts = self.cfg('developer') or {}
        if self._devopts:
            gws.log.warning('developer mode enabled')

        self.localeUids = self.cfg('locales') or _DEFAULT_LOCALE
        self.monitor = self.create_child(gws.server.monitor.Object, self.cfg('server.monitor'))
        self.metadata = gws.lib.metadata.from_config(self.cfg('metadata'))

        self.middlewareMgr = middleware.Manager()

        p = self.cfg('fonts')
        if p:
            gws.lib.font.configure(p)

        # NB the order of initialization is important
        # - db
        # - helpers
        # - auth providers
        # - actions, client, web
        # - finally, projects

        self.databaseMgr = self.create_child(gws.base.database.manager.Object, self.cfg('database'))
        self.storageMgr = self.create_child(gws.base.storage.manager.Object, self.cfg('storage'))
        self.authMgr = self.create_child(gws.base.auth.manager.Object, self.cfg('auth'))

        helpers = self.create_children(gws.ext.object.helper, self.cfg('helpers'))
        self.helperMap = {p.extType: p for p in helpers}

        # @TODO default API
        self.actionMgr = self.create_child(gws.base.action.manager.Object, self.cfg('api'))

        self.webMgr = self.create_child(gws.base.web.manager.Object, self.cfg('web'))

        self.searchMgr = self.create_child(gws.base.search.manager.Object)
        self.finders = self.create_children(gws.ext.object.finder, self.cfg('finders'))

        self.modelMgr = self.create_child(gws.base.model.manager.Object)
        self.models = self.create_children(gws.ext.object.model, self.cfg('models'))

        self.templateMgr = self.create_child(gws.base.template.manager.Object)
        self.templates = self.create_children(gws.ext.object.template, self.cfg('templates'))
        for cfg in _DEFAULT_TEMPLATES:
            self.templates.append(self.root.create_shared(gws.ext.object.template, cfg))

        self.owsServices = self.create_children(gws.ext.object.owsService, self.cfg('owsServices'))

        self.client = self.create_child(gws.base.client.Object, self.cfg('client'))

        p = self.cfg('printer')
        if p:
            self.printer = self.create_child(gws.base.printer.Object, p)
        else:
            self.printer = self.root.create_shared(gws.base.printer.Object, _DEFAULT_PRINTER)

        projects = self.create_children(gws.ext.object.project, self.cfg('projects'))
        self.projectMap = {p.uid: p for p in projects}

    def post_configure(self):
        if self.cfg('server.mapproxy.enabled'):
            self.mpxUrl = f"http://{self.cfg('server.mapproxy.host')}:{self.cfg('server.mapproxy.port')}"
            self.mpxConfig = gws.gis.mpx.config.create_and_save(self.root)

        # NB these are populated in config.parser
        for p in self.config.get('configPaths', []):
            self.monitor.add_file(p)
        for p in self.config.get('projectPaths', []):
            self.monitor.add_file(p)
        for d in self.config.get('projectDirs', []):
            self.monitor.add_directory(d, gws.config.CONFIG_PATH_PATTERN)
        if self.developer_option('server.auto_reload'):
            self.monitor.add_directory(gws.APP_DIR, r'\.py$')

    def register_middleware(self, name, obj, depends_on=None):
        self.middlewareMgr.register(name, obj, depends_on)

    def middleware_objects(self):
        return self.middlewareMgr.sorted_objects()

    def projects_for_user(self, user):
        return [p for p in self.projectMap.values() if user.can_use(p)]

    def project(self, uid):
        return self.projectMap.get(uid)

    def helper(self, ext_type):
        return self.helperMap.get(ext_type)

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
