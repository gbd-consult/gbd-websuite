"""Core application object"""

import gws
import gws.base.api
import gws.base.auth
import gws.base.client
import gws.base.db
import gws.base.project
import gws.base.web
import gws.lib.cache
import gws.lib.metadata
import gws.lib.mpx.config
import gws.lib.os2
import gws.server
import gws.server.monitor
import gws.types as t


# try:
#     import gws.plugins.qgis.server as qgis_server
# except ImportError:
#     qgis_server = None


class FontConfig(gws.Config):
    """Fonts configuration."""

    dir: gws.DirPath  #: directory with custom fonts


class Config(gws.WithAccess):
    """Main application configuration"""

    api: t.Optional[gws.base.api.Config]  #: system-wide server actions
    auth: t.Optional[gws.base.auth.Config]  #: authorization methods and options
    cache: t.Optional[gws.lib.cache.Config] = {}  # type: ignore #: global cache configuration
    client: t.Optional[gws.base.client.Config]  #: gws client configuration
    db: t.Optional[gws.base.db.Config]  #: database configuration
    developer: t.Optional[dict]  #: developer options
    fonts: t.Optional[FontConfig]  #: fonts configuration
    helpers: t.Optional[t.List[gws.ext.helper.Config]]  #: helpers configurations
    locales: t.Optional[t.List[str]]  #: default locales for all projects
    meta: t.Optional[gws.lib.metadata.Config]  # type: ignore #: application metadata
    projectDirs: t.Optional[t.List[gws.DirPath]]  #: directories with additional projects
    projectPaths: t.Optional[t.List[gws.FilePath]]  #: additional project paths
    projects: t.Optional[t.List[gws.base.project.Config]]  #: project configurations
    server: t.Optional[gws.server.Config] = {}  # type: ignore #: server engine options
    web: t.Optional[gws.base.web.Config]  #: web server options


class Object(gws.Node, gws.IApplication):
    """Main Appilication object"""

    api: gws.base.api.Object
    auth: gws.base.auth.Manager
    monitor: gws.server.monitor.Object
    meta: gws.lib.metadata.Object
    web_sites: t.List[gws.IWebSite]
    localeUids: t.List[str]

    _devopts: dict

    def configure(self):
        self._devopts = self.var('developer') or {}
        if self._devopts:
            gws.log.warn('DEVELOPER MODE ENABLED')

        self.set_uid('APP')

        # self.version: str = gws.VERSION
        #
        # self.qgis_version: str = qgis_server.version() if qgis_server else ''
        #
        # gws.log.info('*' * 40)
        # gws.log.info(f'GWS version {self.version}, QGis {self.qgis_version}')
        # gws.log.info('*' * 40)

        self.localeUids = self.var('locales') or ['en_CA']

        self.monitor = self.create_child(gws.server.monitor.Object, self.var('server.monitor'))

        p = self.var('meta')
        self.meta = self.create_child(gws.lib.metadata.Object, p)

        s = self.var('fonts.dir')
        if s:
            _install_fonts(s)

        # NB the order of initialization is important
        # - db
        # - helpers
        # - auth providers
        # - actions, client, web
        # - finally, projects
        #
        #         for p in self.var('db.providers', default=[]):
        #             self.create_child('gws.ext.db.provider', p)
        #
        #         for p in self.var('helpers', default=[]):
        #             self.create_child('gws.ext.helper', p)
        #
        #         self.auth: gws.IAuthManager = t.cast(gws.IAuthManager, self.create_child(gws.base.auth.Object, self.var('auth', default=gws.Data())))
        #         self.api: gws.IApi = t.cast(gws.IApi, self.create_child(gws.base.api.Object, self.var('api', default=gws.Data())))
        #
        #         p = self.var('client')
        #         self.client: t.Optional[gws.IClient] = self.create_child(gws.base.client.Object, p) if p else None
        #

        self.auth = t.cast(gws.base.auth.Manager, self.create_child(gws.base.auth.Manager, self.var('auth')))

        self.api = self.create_child(gws.base.api.Object, self.var('api'))

        self.web_sites = []

        p = self.var('web.sites', default=[gws.base.web.DEFAULT_SITE])
        for s in p:
            s.ssl = True if self.var('web.ssl') else False
            self.web_sites.append(self.root.create_object(gws.base.web.Site, s))

        for p in self.var('projects', default=[]):
            self.create_child(gws.base.project.Object, p)

    def post_configure(self):
        if self.var('server.mapproxy.enabled'):
            gws.lib.mpx.config.create_and_save(self.root)

        pass
        # for p in set(cfg.configPaths):
        #     root.application.monitor.add_path(p)
        # for p in set(cfg.projectPaths):
        #     root.application.monitor.add_path(p)
        # for d in set(cfg.projectDirs):
        #     root.application.monitor.add_directory(d, parser.config_path_pattern)
        #
        # if root.application.developer_option('server.auto_reload'):
        #     root.application.monitor.add_directory(gws.APP_DIR, '\.py$')

    def find_action(self, action_type: str, project_uid: str = None) -> t.Optional[gws.IObject]:
        if project_uid:
            project = t.cast(gws.base.project.Object, self.root.find(klass='gws.base.project', uid=project_uid))
            if project and project.api:
                action = project.api.find_action(action_type)
                if action:
                    return action

        if self.api:
            return self.api.find_action(action_type)

    def helper(self, key: str) -> gws.INode:
        base = 'gws.ext.helper'
        p = self.root.find(f'{base}.{key}')
        if not p:
            cfg = self.root.specs.read_value({'type': key}, f'{base}.{key}.Config')
            gws.log.debug(f'created an ad-hoc helper, key={key!r} cfg={cfg!r}')
            p = self.create_child(base, cfg)
        return p


def _install_fonts(source_dir):
    target_dir = '/usr/local/share/fonts'
    gws.lib.os2.run(['mkdir', '-p', target_dir], echo=True)
    for p in gws.lib.os2.find_files(source_dir):
        gws.lib.os2.run(['cp', '-v', p, target_dir], echo=True)

    gws.lib.os2.run(['fc-cache', '-fv'], echo=True)

#
#
#
#
#
# import gws.base.auth
# import gws.base.client
# import gws.base.layer
# import gws.lib.metadata
# import gws.base.project
# import gws.base.search
# import gws.base.template
# import gws.lib.zoom
# import gws.lib.os2
# import gws.server.monitor
# import gws.server.types
# import gws.base.web.site
#
#
#
#
#
# class Config(gws.WithAccess):
#     """Main application configuration"""
#
#
#
# _DEFAULT_SITE = gws.Data(host='*', root=gws.Data(dir='/data/web'))
#
#
# class Object(gws.Node, gws.IApplication):
#     """Main Appilication object"""
#
#     def configure(self):
#         self.api = self.create_child(gws.base.api.Object, self.var('api', default=gws.Data()))
#
#
#     def ___configure(self):
#
#
#
#
#     def developer_option(self, name):
#         return self._devopts.get(name)
#
#
#
#
