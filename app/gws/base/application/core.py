"""Core application object"""

from typing import Optional

import gws
import gws.base.action
import gws.base.application.middleware
import gws.base.auth
import gws.base.client
import gws.base.database
import gws.base.job
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
import gws.base.metadata
import gws.lib.osx
import gws.server.manager
import gws.server.monitor
import gws.spec

_DEFAULT_LOCALE = ['en_CA']

_DEFAULT_TEMPLATES = [
    gws.Config(
        type='html',
        path=gws.u.dirname(__file__) + '/templates/project_description.cx.html',
        subject='project.description',
        access=gws.c.PUBLIC,
        uid='default_template.project_description',
    ),
    gws.Config(
        type='html',
        path=gws.u.dirname(__file__) + '/templates/layer_description.cx.html',
        subject='layer.description',
        access=gws.c.PUBLIC,
        uid='default_template.layer_description',
    ),
    gws.Config(
        type='html',
        path=gws.u.dirname(__file__) + '/templates/feature_description.cx.html',
        subject='feature.description',
        access=gws.c.PUBLIC,
        uid='default_template.feature_description',
    ),
    gws.Config(
        type='html',
        path=gws.u.dirname(__file__) + '/templates/feature_title.cx.html',
        subject='feature.title',
        access=gws.c.PUBLIC,
        uid='default_template.feature_title',
    ),
    gws.Config(
        type='html',
        path=gws.u.dirname(__file__) + '/templates/feature_label.cx.html',
        subject='feature.label',
        access=gws.c.PUBLIC,
        uid='default_template.feature_label',
    ),
]

_DEFAULT_PRINTER = gws.Config(
    uid='gws.base.application.default_printer',
    access=gws.c.PUBLIC,
    template=gws.Config(
        type='html',
        path=gws.u.dirname(__file__) + '/templates/project_print.cx.html',
        mapSize=(200, 180, gws.Uom.mm),
    ),
    qualityLevels=[{'dpi': 72}],
)


class Config(gws.ConfigWithAccess):
    """Main application configuration"""

    actions: Optional[list[gws.ext.config.action]]
    """System-wide server actions."""
    auth: Optional[gws.base.auth.manager.Config]
    """Authorization methods and options."""
    cache: Optional[gws.gis.cache.Config]
    """Global cache configuration."""
    client: Optional[gws.base.client.Config]
    """Gws client configuration."""
    database: Optional[gws.base.database.manager.Config]
    """Database configuration."""
    developer: Optional[dict]
    """Developer options."""
    finders: Optional[list[gws.ext.config.finder]]
    """Global search providers."""
    fonts: Optional[gws.lib.font.Config]
    """Fonts configuration."""
    helpers: Optional[list[gws.ext.config.helper]]
    """Helpers configurations."""
    locales: Optional[list[str]]
    """Default locales for all projects."""
    metadata: Optional[gws.base.metadata.Config]
    """Application metadata."""
    models: Optional[list[gws.ext.config.model]]
    """Global data models."""
    owsServices: Optional[list[gws.ext.config.owsService]]
    """OWS services configuration."""
    projectDirs: Optional[list[gws.DirPath]]
    """Directories with additional projects."""
    projectPaths: Optional[list[gws.FilePath]]
    """Additional project paths."""
    printers: Optional[list[gws.ext.config.printer]]
    """Print configurations."""
    projects: Optional[list[gws.ext.config.project]]
    """Project configurations."""
    server: Optional[gws.server.Config]
    """Server engine options."""
    storage: Optional[gws.base.storage.manager.Config]
    """Database configuration."""
    templates: Optional[list[gws.ext.config.template]]
    """Default templates."""
    vars: Optional[dict]
    """Custom variables."""
    web: Optional[gws.base.web.manager.Config]
    """Web server options."""


class Object(gws.Application):
    """Main Application object"""

    _helperMap: dict[str, gws.Node]

    _developerOptions: dict

    mpxUrl = ''
    mpxConfig = ''

    def configure(self):
        self.vars = self.cfg('vars') or {}
        
        self.serverMgr = self.create_child(gws.server.manager.Object, self.cfg('server'))
        # NB need defaults from the server
        self.config.server = self.serverMgr.config

        p = self.cfg('server.log.level')
        if p:
            gws.log.set_level(p)

        self.version = self.root.specs.version
        self.versionString = f'GWS version {self.version}'

        if gws.u.is_file('/GWS_REVISION'):
            self.versionString += ' (rev. ' + gws.u.read_file('/GWS_REVISION') + ')'

        if not gws.env.GWS_IN_TEST:
            gws.log.info('*' * 80)
            gws.log.info(self.versionString)
            gws.log.info('*' * 80)

        self._developerOptions = self.cfg('developer') or {}
        if self._developerOptions:
            gws.log.warning(f'developer mode enabled: {self._developerOptions}')

        self.monitor = self.create_child(gws.server.monitor.Object, self.serverMgr.cfg('monitor'))

        self.localeUids = self.cfg('locales') or _DEFAULT_LOCALE
        self.metadata = gws.base.metadata.from_config(self.cfg('metadata'))

        self.middlewareMgr = self.create_child(gws.base.application.middleware.Object)

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

        helpers = self.create_children(gws.ext.object.helper, self.cfg('helpers'))
        self._helperMap = {p.extType: p for p in helpers}

        self.storageMgr = self.create_child(gws.base.storage.manager.Object, self.cfg('storage'))
        self.authMgr = self.create_child(gws.base.auth.manager.Object, self.cfg('auth'))

        # @TODO default API
        self.actionMgr = self.create_child(gws.base.action.manager.Object)
        self.actions = self.create_children(gws.ext.object.action, self.cfg('actions'))

        self.webMgr = self.create_child(gws.base.web.manager.Object, self.cfg('web'))

        self.searchMgr = self.create_child(gws.base.search.manager.Object)
        self.finders = self.create_children(gws.ext.object.finder, self.cfg('finders'))

        self.modelMgr = self.create_child(gws.base.model.manager.Object)
        self.models = self.create_children(gws.ext.object.model, self.cfg('models'))

        self.templateMgr = self.create_child(gws.base.template.manager.Object)
        self.templates = self.create_children(gws.ext.object.template, self.cfg('templates'))
        for cfg in _DEFAULT_TEMPLATES:
            self.templates.append(self.root.create_shared(gws.ext.object.template, cfg))

        self.jobMgr = self.create_child(gws.base.job.manager.Object)

        self.printerMgr = self.create_child(gws.base.printer.manager.Object)
        self.printers = self.create_children(gws.ext.object.printer, self.cfg('printers'))
        self.defaultPrinter = self.root.create_shared(gws.ext.object.printer, _DEFAULT_PRINTER)

        self.owsServices = self.create_children(gws.ext.object.owsService, self.cfg('owsServices'))

        self.client = self.create_child(gws.base.client.Object, self.cfg('client'))

        self.projects = self.create_children(gws.ext.object.project, self.cfg('projects'))

    def post_configure(self):
        if not self.cfg('server.mapproxy.disabled'):
            self.mpxUrl = f"http://{self.cfg('server.mapproxy.host')}:{self.cfg('server.mapproxy.port')}"
            self.mpxConfig = gws.gis.mpx.config.create_and_save(self.root)

    def activate(self):
        for p in self.root.configPaths:
            self.monitor.watch_file(p)
        for d in self.config.get('projectDirs') or []:
            self.monitor.watch_directory(d, gws.config.CONFIG_PATH_PATTERN)

    def project(self, uid):
        for p in self.projects:
            if p.uid == uid:
                return p

    def helper(self, ext_type):
        if ext_type not in self._helperMap:
            p = self.create_child(gws.ext.object.helper, type=ext_type)
            if not p:
                raise gws.Error(f'helper {ext_type!r} not found')
            gws.log.info(f'created helper {ext_type!r}')
            self._helperMap[ext_type] = p
        return self._helperMap.get(ext_type)

    def developer_option(self, key):
        return self._developerOptions.get(key)
