"""Core application object"""

import gws
import gws.base.api
import gws.base.auth
import gws.base.auth.manager
import gws.base.client
import gws.base.db
import gws.base.metadata
import gws.base.project
import gws.base.web
import gws.config
import gws.lib.cache
import gws.lib.font
import gws.lib.mpx.config
import gws.lib.os2
import gws.server
import gws.server.monitor
import gws.types as t


class FontConfig(gws.Config):
    """Fonts configuration."""

    dir: gws.DirPath  #: directory with custom fonts


class Config(gws.WithAccess):
    """Main application configuration"""

    api: t.Optional[gws.base.api.Config]  #: system-wide server actions
    auth: t.Optional[gws.base.auth.Config] = {}  # type: ignore #: authorization methods and options
    cache: t.Optional[gws.lib.cache.Config] = {}  # type: ignore #: global cache configuration
    client: t.Optional[gws.base.client.Config]  #: gws client configuration
    db: t.Optional[gws.base.db.Config]  #: database configuration
    developer: t.Optional[dict]  #: developer options
    fonts: t.Optional[FontConfig]  #: fonts configuration
    helpers: t.Optional[t.List[gws.ext.helper.Config]]  #: helpers configurations
    locales: t.Optional[t.List[str]]  #: default locales for all projects
    metaData: t.Optional[gws.base.metadata.Config]  # type: ignore #: application metadata
    projectDirs: t.Optional[t.List[gws.DirPath]]  #: directories with additional projects
    projectPaths: t.Optional[t.List[gws.FilePath]]  #: additional project paths
    projects: t.Optional[t.List[gws.base.project.Config]]  #: project configurations
    server: t.Optional[gws.server.Config] = {}  # type: ignore #: server engine options
    web: t.Optional[gws.base.web.Config]  #: web server options


class Object(gws.Node, gws.IApplication):
    """Main Appilication object"""

    dbs: t.List[gws.ISqlDbProvider]
    helpers: t.List[gws.Node]
    projects: t.List[gws.base.project.Object]

    _devopts: dict

    def configure(self):
        self.version = gws.VERSION
        self.qgis_version = ''

        self._devopts = self.var('developer') or {}
        if self._devopts:
            gws.log.warn('developer mode enabled')

        self.set_uid('APP')

        if self.var('server.qgis.enabled'):
            qgis_server = gws.import_from_path('gws/plugin/qgis/server.py')
            self.qgis_version = qgis_server.version()

        s = f'GWS version {self.version}'
        if self.qgis_version:
            s += f', QGis {self.qgis_version}'
        gws.log.info('*' * 40)
        gws.log.info(s)
        gws.log.info('*' * 40)

        self.locale_uids = self.var('locales') or ['en_CA']
        self.monitor = self.require_child(gws.server.monitor.Object, self.var('server.monitor'))
        self.metadata = self.require_child(gws.base.metadata.Object, self.var('metaData'))

        p = self.var('fonts.dir')
        if p:
            gws.lib.font.install_fonts(p)

        # NB the order of initialization is important
        # - db
        # - helpers
        # - auth providers
        # - actions, client, web
        # - finally, projects

        self.dbs = self.create_children('gws.ext.db.provider', self.var('db.providers'))

        # helpers are always created, no matter configured or not
        cnf = {c.get('type'): c for c in self.var('helpers', default=[])}
        for class_name in self.root.specs.real_class_names('gws.ext.helper'):
            desc = self.root.specs.object_descriptor(class_name)
            if desc.ext_type not in cnf:
                gws.log.debug(f'ad-hoc helper {desc.ext_type!r} will be created')
                cfg = gws.Config(type=desc.ext_type)
                cnf[desc.ext_type] = gws.config.parse(self.root.specs, cfg, 'gws.ext.helper.Config')
        self.helpers = self.create_children('gws.ext.helper', list(cnf.values()))

        self.auth = self.require_child(gws.base.auth.manager.Object, self.var('auth'))

        # @TODO default API
        self.api = self.require_child(gws.base.api.Object, self.var('api'))

        p = self.var('web.sites') or [gws.base.web.DEFAULT_SITE]
        ssl = bool(self.var('web.ssl'))
        cfgs = [gws.merge(c, ssl=ssl) for c in p]
        self.web_sites = self.create_children(gws.base.web.site.Object, cfgs)

        self.client = self.create_child_if_config(gws.base.client.Object, self.var('client'))

        self.projects = []
        for cfg in self.var('projects', default=[]):
            # @TODO: parallel config?
            self.projects.append(self.create_child(gws.base.project.Object, cfg))

    def post_configure(self):
        self.mpx_url = ''
        if self.var('server.mapproxy.enabled'):
            gws.lib.mpx.config.create_and_save(self.root)
            self.mpx_url = f"http://{self.var('server.mapproxy.host')}:{self.var('server.mapproxy.port')}"

        # for p in set(cfg.configPaths):
        #     root.application.monitor.add_path(p)
        # for p in set(cfg.projectPaths):
        #     root.application.monitor.add_path(p)
        # for d in set(cfg.projectDirs):
        #     root.application.monitor.add_directory(d, parser.config_path_pattern)
        #
        # if root.application.developer_option('server.auto_reload'):
        #     root.application.monitor.add_directory(gws.APP_DIR, '\.py$')

    def find_action(self, user, ext_type, project_uid=None):
        actions = {}

        if project_uid:
            project: gws.base.project.Object = self.root.find('gws.base.project', project_uid)
            if project and project.api:
                actions = project.api.actions_for(user, parent=self.api)

        if not actions:
            actions = self.api.actions_for(user)

        return actions.get(ext_type)

    def require_helper(self, ext_type):
        for obj in self.helpers:
            if obj.ext_type == ext_type:
                return obj
        raise gws.Error(f'helper {ext_type!r} not found')

    def developer_option(self, name):
        return self._devopts.get(name)
