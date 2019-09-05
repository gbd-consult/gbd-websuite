"""Core application object"""

import gws
import gws.auth.api
import gws.auth.provider
import gws.auth.types
import gws.common.api
import gws.common.client
import gws.common.csv
import gws.common.project
import gws.common.search
import gws.common.template
import gws.gis.layer
import gws.gis.zoom
import gws.qgis.server
import gws.server.types
import gws.tools.misc as misc
import gws.tools.shell
import gws.web.site

import gws.types as t


class DbConfig(t.Config):
    """Database configuration"""

    providers: t.List[t.ext.db.provider.Config]


class SeedingConfig(t.Config):
    """Seeding options"""

    maxTime: t.duration = 600  #: max. time for a seeding job
    concurrency: int = 1  #: number of concurrent seeding jobs


class FontConfig(t.Config):
    """Fonts configuration."""

    dir: t.dirpath  #: directory with custom fonts


class SSLConfig(t.Config):
    """SSL configuration"""

    crt: t.filepath  #: crt file location
    key: t.filepath  #: key file location


class WebConfig(t.Config):
    """Web server configuration"""

    sites: t.Optional[t.List[gws.web.site.Config]]  #: configured sites
    ssl: t.Optional[SSLConfig]  #: ssl configuration


class Config(t.Config):
    """Application configuration"""

    access: t.Optional[t.Access]  #: default access mode
    api: t.Optional[gws.common.api.Config]  #: system-wide server actions
    auth: t.Optional[gws.auth.types.Config]  #: authorization methods and options
    client: t.Optional[gws.common.client.Config]  #: gws client configuration
    csv: t.Optional[gws.common.csv.Config] = {}  #: csv format options
    db: t.Optional[DbConfig]  #: database configuration
    fonts: t.Optional[FontConfig]  #: fonts configuration
    locales: t.Optional[t.List[str]] #: default locales for all projects
    projectDirs: t.Optional[t.List[t.dirpath]]  #: directories with additional projects
    projectPaths: t.Optional[t.List[t.filepath]]  #: additional project paths
    projects: t.Optional[t.List[gws.common.project.Config]]  #: project configurations
    seeding: SeedingConfig = {}  #: configuration for seeding jobs
    server: t.Optional[gws.server.types.Config] = {}  #: server engine options
    storage: t.Optional[t.ext.storage.Config]  #: storage configuration
    timeZone: t.Optional[str] = 'UTC'  #: timezone for this server
    web: t.Optional[WebConfig] = {}  #: webserver configuration


_default_site = t.Data({
    'host': '*',
    'root': t.Data({
        'dir': '/data/web',
    })
})


class Object(gws.Object):
    def __init__(self):
        super().__init__()

        self.api: gws.common.api.Object = None
        self.client: gws.common.client.Object = None
        self.qgis_version = ''
        self.storage: t.StorageInterface = None
        self.version = gws.VERSION
        self.web_sites: t.List[gws.web.site.Object] = []

    @property
    def auto_uid(self):
        return None

    def configure(self):
        super().configure()

        self.defaults = self.var('defaults')

        self.set_uid('APP')

        self.qgis_version = gws.qgis.server.version()

        gws.log.info(f'GWS version {self.version}, QGis {self.qgis_version}')

        _install_fonts(self.var('fonts.dir'))

        gws.auth.api.init()

        for p in self.var('auth.providers', default=[]):
            self.add_child('gws.ext.auth.provider', p)
        self.add_child('gws.ext.auth.provider', t.Data({'type': 'system', 'uid': 'system'}))

        for p in self.var('db.providers', default=[]):
            self.add_child('gws.ext.db.provider', p)

        for p in self.var('projects'):
            self.add_child(gws.common.project.Object, p)

        p = self.var('api') or t.Data({'actions': []})

        self.api = self.add_child(gws.common.api.Object, p)

        p = self.var('client')
        if p:
            self.client = self.add_child(gws.common.client.Object, p)

        p = self.var('web.sites') or [_default_site]
        for s in p:
            if self.var('web.ssl'):
                s.ssl = True
            site = self.create_object('gws.web.site', s)
            self.web_sites.append(site)

        self.add_child(gws.common.csv.Object, self.var('csv'))

        p = self.var('storage') or {'type': 'sqlite'}
        self.storage = self.add_child('gws.ext.storage', p)

    def find_action(self, action_type, project_uid=None):

        action = None

        if project_uid:
            project = self.find('gws.common.project', project_uid)
            if project:
                action = project.api.actions.get(action_type) if project.api else None

        if not action:
            action = self.api.actions.get(action_type) if self.api else None

        gws.log.debug(f'find_action {action_type!r} prj={project_uid!r} found={action.uid if action else None}')
        return action


def _install_fonts(source_dir):
    gws.log.info('checking fonts...')

    if source_dir:
        target_dir = '/usr/local/share/fonts'
        gws.tools.shell.run(['mkdir', '-p', target_dir], echo=True)
        for p in misc.find_files(source_dir):
            gws.tools.shell.run(['cp', '-v', p, target_dir], echo=True)

    gws.tools.shell.run(['fc-cache', '-fv'], echo=True)
