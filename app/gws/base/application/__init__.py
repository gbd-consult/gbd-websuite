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


class Object(gws.Object, gws.IApplication):
    """Main Appilication object"""

    api: gws.base.api.Object
    auth: gws.base.auth.manager.Object
    client: t.Optional[gws.base.client.Object]
    dbs: t.List[gws.ISqlDbProvider]
    helpers: t.List[gws.Object]
    locale_uids: t.List[str]
    metadata: gws.base.metadata.Object
    monitor: gws.server.monitor.Object
    mpx_url: str
    projects: t.List[gws.base.project.Object]
    qgis_version: str
    version: str
    web_sites: t.List[gws.IWebSite]

    _devopts: dict

    def configure(self):
        self.version = gws.VERSION
        self.qgis_version = ''

        self._devopts = self.var('developer') or {}
        if self._devopts:
            gws.log.warn('developer mode enabled')

        self.set_uid('APP')

        if self.var('server.qgis.enabled'):
            qgis_server = gws.import_from_path(
                gws.APP_DIR + '/gws/plugin/qgis/server.py',
                'gws.plugin.qgis.server')
            self.qgis_version = qgis_server.version()

        s = f'GWS version {self.version}'
        if self.qgis_version:
            s += f', QGis {self.qgis_version}'
        gws.log.info('*' * 40)
        gws.log.info(s)
        gws.log.info('*' * 40)

        self.locale_uids = self.var('locales') or ['en_CA']
        self.monitor = self.create_child(gws.server.monitor.Object, self.var('server.monitor'))
        self.metadata = self.create_child(gws.base.metadata.Object, self.var('metaData'))

        p = self.var('fonts.dir')
        if p:
            gws.lib.font.install_fonts(p)

        # NB the order of initialization is important
        # - db
        # - helpers
        # - auth providers
        # - actions, client, web
        # - finally, projects

        self.dbs = t.cast(
            t.List[gws.ISqlDbProvider],
            self.create_children('gws.ext.db.provider', self.var('db.providers')))

        # helpers are always created, no matter configured or not
        cnf = {c.get('type'): c for c in self.var('helpers') or []}
        for typ in self.root.specs.ext_type_list('helper'):
            if typ not in cnf:
                cnf[typ] = gws.Config(type=typ)
        self.helpers = self.create_children('gws.ext.helper', list(cnf.values()))

        self.auth = self.create_child(gws.base.auth.manager.Object, self.var('auth'))

        self.api = self.create_child(gws.base.api.Object, self.var('api'))

        p = self.var('web.sites') or [gws.base.web.DEFAULT_SITE]
        ssl = bool(self.var('web.ssl'))
        cfgs = [gws.merge(c, ssl=ssl) for c in p]
        self.web_sites = t.cast(
            t.List[gws.IWebSite],
            self.create_children(gws.base.web.site.Object, cfgs))

        self.client = t.cast(
            gws.base.client.Object,
            self.create_child_if_config(gws.base.client.Object, self.var('client')))

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

    def find_action(self, ext_type, project_uid=None):
        if project_uid:
            project = t.cast(gws.base.project.Object, self.root.find('gws.base.project', project_uid))
            if project and project.api:
                action = project.api.find_action(ext_type)
                if action:
                    return action

        if self.api:
            return self.api.find_action(ext_type)

    def require_helper(self, ext_type):
        for obj in self.helpers:
            if obj.ext_type == ext_type:
                return obj
        raise gws.Error(f'helper {ext_type!r} not found')

    def developer_option(self, name):
        return self._devopts.get(name)
