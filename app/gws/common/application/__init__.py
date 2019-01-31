"""Core application object"""

import gws
import gws.auth.api
import gws.auth.provider
import gws.auth.types
import gws.common.api
import gws.common.client
import gws.common.project
import gws.common.csv
import gws.types as t
import gws.server.types
import gws.tools.misc as misc
import gws.tools.shell as sh
import gws.web.types
import gws.qgis.server


class DbConfig(t.Config):
    """database configuration"""

    providers: t.List[t.ext.db.provider.Config]


class SeedingConfig(t.Config):
    """seeding options"""

    maxTime: t.duration = 600  #: max. time for a seeding job
    concurrency: int = 1  #: number of concurrent seeding jobs


class Config(t.Config):
    """main gws application configuration"""

    access: t.Optional[t.Access]  #: default access mode
    api: t.Optional[gws.common.api.Config]  #: system-wide server actions
    auth: t.Optional[gws.auth.types.Config]  #: authorization methods and options
    client: t.Optional[gws.common.client.Config]  #: gws client configuration
    db: t.Optional[DbConfig]  #: database configuration
    locale: t.Optional[str] = 'en_CA'  #: default locale for all projects
    seeding: SeedingConfig = {}  #: configuration for seeding jobs
    projects: t.Optional[t.List[gws.common.project.Config]]  #: project configurations
    projectDirs: t.Optional[t.List[t.dirpath]]  #: directories with additional projects
    projectPaths: t.Optional[t.List[t.filepath]]  #: additional project paths
    server: t.Optional[gws.server.types.Config] = {}  #: server engine options
    timeZone: t.Optional[str] = 'UTC'  #: timezone for this server
    web: t.Optional[gws.web.types.Config] = {}  #: webserver configuration
    fonts: t.Optional[t.dirpath]  #: directory with the custom fonts
    csv: t.Optional[gws.common.csv.Config] = {}  #: csv format options


class Object(gws.Object):
    version = gws.VERSION
    qgis_version = ''
    client = None

    def configure(self):
        super().configure()

        self.uid = ''

        self.qgis_version = gws.qgis.server.version()

        gws.log.info(f'GWS version {self.version}, QGis {self.qgis_version}')

        _install_fonts(self.var('fonts'))

        gws.auth.api.init()

        for p in self.var('auth.providers', default=[]):
            self.add_child('gws.ext.auth.provider', p)
        self.add_child('gws.ext.auth.provider', t.Data({'type': 'system', 'uid': 'system'}))

        for p in self.var('db.providers', default=[]):
            self.add_child('gws.ext.db.provider', p)

        for p in self.var('projects'):
            self.add_child(gws.common.project.Object, p)

        p = self.var('api') or t.Data({'actions': []})

        if _is_remote_admin_enabled():
            gws.log.info('REMOTE_ADMIN: enabled')
            p.actions.append(t.Data({'type': 'remoteadmin'}))

        self.api = self.add_child(gws.common.api.Object, p)

        p = self.var('client')
        if p:
            self.client = self.add_child(gws.common.client.Object, p)

        self.add_child(gws.common.csv.Object, self.var('csv'))

    def action(self, action_type):
        if not self.api:
            return None
        return self.api.actions.get(action_type)


def _is_remote_admin_enabled():
    try:
        with open(gws.REMOTE_ADMIN_PASSWD_FILE) as fp:
            p = fp.read().strip()
            return len(p) > 0
    except:
        return False


def _install_fonts(source_dir):
    gws.log.info('checking fonts...')

    if source_dir:
        target_dir = '/usr/local/share/fonts'
        sh.run(['mkdir', '-p', target_dir], echo=True)
        for p in misc.find_files(source_dir):
            sh.run(['cp', '-v', p, target_dir], echo=True)

    sh.run(['fc-cache', '-fv'], echo=True)


def _allow_all_wrapper(app, uid):
    return app.add_child(gws.Object, t.Config({
        'uid': uid,

    }))
