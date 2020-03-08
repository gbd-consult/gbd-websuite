"""Core application object"""

import gws

import gws.common.api
import gws.common.auth.types
import gws.common.auth.util
import gws.common.client
import gws.common.layer
import gws.common.project
import gws.common.search
import gws.common.template
import gws.gis.zoom
import gws.qgis.server
import gws.server.types
import gws.tools.os2
import gws.web.site

import gws.types as t


class DbConfig(t.Config):
    """Database configuration"""

    providers: t.List[t.ext.db.provider.Config]


class SeedingConfig(t.Config):
    """Seeding options"""

    maxTime: t.Duration = 600  #: max. time for a seeding job
    concurrency: int = 1  #: number of concurrent seeding jobs


class FontConfig(t.Config):
    """Fonts configuration."""

    dir: t.DirPath  #: directory with custom fonts


class SSLConfig(t.Config):
    """SSL configuration"""

    crt: t.FilePath  #: crt file location
    key: t.FilePath  #: key file location


class WebConfig(t.Config):
    """Web server configuration"""

    sites: t.Optional[t.List[gws.web.site.Config]]  #: configured sites
    ssl: t.Optional[SSLConfig]  #: ssl configuration


class Config(t.WithAccess):
    """Application configuration"""

    api: t.Optional[gws.common.api.Config]  #: system-wide server actions
    auth: t.Optional[gws.common.auth.types.Config]  #: authorization methods and options
    client: t.Optional[gws.common.client.Config]  #: gws client configuration
    db: t.Optional[DbConfig]  #: database configuration
    fonts: t.Optional[FontConfig]  #: fonts configuration
    locales: t.Optional[t.List[str]]  #: default locales for all projects
    projectDirs: t.Optional[t.List[t.DirPath]]  #: directories with additional projects
    projectPaths: t.Optional[t.List[t.FilePath]]  #: additional project paths
    projects: t.Optional[t.List[gws.common.project.Config]]  #: project configurations
    seeding: SeedingConfig = {}  #: configuration for seeding jobs
    server: t.Optional[gws.server.types.Config] = {}  #: server engine options
    timeZone: t.Optional[str] = 'UTC'  #: timezone for this server
    helpers: t.Optional[t.List[t.ext.helper.Config]]
    web: t.Optional[WebConfig] = {}  #: webserver configuration


_default_site = t.Data({
    'host': '*',
    'root': t.Data({
        'dir': '/data/web',
    })
})


#:export IApplication
class Object(gws.Object, t.IApplication):
    """Main Appilication object"""

    def __init__(self):
        super().__init__()

        self.api: t.IApi = None
        self.client: t.IClient = None
        self.qgis_version = ''
        self.version = gws.VERSION
        self.web_sites: t.List[t.IWebSite] = []

    def configure(self):
        super().configure()

        self.set_uid('APP')

        self.defaults = self.var('defaults')
        self.qgis_version = gws.qgis.server.version()

        gws.log.info(f'GWS version {self.version}, QGis {self.qgis_version}')

        s = self.var('fonts.dir')
        if s:
            _install_fonts(s)

        # NB the order of initialization is important
        # - db
        # - helpers
        # - auth providers
        # - actions, client, web
        # - finally, projects

        for p in self.var('db.providers', default=[]):
            self.add_child('gws.ext.db.provider', p)

        for p in self.var('helpers', default=[]):
            self.add_child('gws.ext.helper', p)

        gws.common.auth.init()

        for p in self.var('auth.providers', default=[]):
            self.add_child('gws.ext.auth.provider', p)
        self.add_child('gws.ext.auth.provider', t.Data({'type': 'system'}))

        p = self.var('api') or t.Data({'actions': []})

        self.api = self.add_child(gws.common.api.Object, p)

        p = self.var('client')
        if p:
            self.client = self.add_child(gws.common.client.Object, p)

        p = self.var('web.sites') or [_default_site]
        for s in p:
            s.ssl = True if self.var('web.ssl') else False
            self.web_sites.append(self.create_object('gws.web.site', s))

        for p in self.var('projects'):
            self.add_child(gws.common.project.Object, p)

    def find_action(self, action_type, project_uid=None):

        action = None

        if project_uid:
            project: t.IProject = self.find('gws.common.project', project_uid)
            if project:
                action = project.api.actions.get(action_type) if project.api else None

        if not action:
            action = self.api.actions.get(action_type) if self.api else None

        gws.log.debug(f'find_action {action_type!r} prj={project_uid!r} found={action.uid if action else None}')
        return action


def _install_fonts(source_dir):
    gws.log.info('checking fonts...')

    target_dir = '/usr/local/share/fonts'
    gws.tools.os2.run(['mkdir', '-p', target_dir], echo=True)
    for p in gws.tools.os2.find_files(source_dir):
        gws.tools.os2.run(['cp', '-v', p, target_dir], echo=True)

    gws.tools.os2.run(['fc-cache', '-fv'], echo=True)
