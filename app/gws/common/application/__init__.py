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
import gws.gis.layer
import gws.gis.zoom
import gws.common.search


class LayerDefaults(t.Config):
    crs: t.crsref = 'EPSG:3857'
    cache: gws.gis.layer.CacheConfig = {}
    grid: gws.gis.layer.GridConfig = {}
    clientOptions: gws.gis.layer.ClientOptions = {}
    opacity: float = 1
    legend: gws.gis.layer.LegendConfig = {}
    zoom: t.Optional[gws.gis.zoom.Config]
    extent: t.Optional[t.Extent]
    search: t.Optional[gws.common.search.Config] = {}


class Defaults(t.Config):
    layer: LayerDefaults = {}


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
    csv: t.Optional[gws.common.csv.Config] = {}  #: csv format options
    db: t.Optional[DbConfig]  #: database configuration
    defaults: Defaults = {}  #: default configuration options
    fonts: t.Optional[t.dirpath]  #: directory with the custom fonts
    locale: t.Optional[str] = 'en_CA'  #: default locale for all projects
    projectDirs: t.Optional[t.List[t.dirpath]]  #: directories with additional projects
    projectPaths: t.Optional[t.List[t.filepath]]  #: additional project paths
    projects: t.Optional[t.List[gws.common.project.Config]]  #: project configurations
    seeding: SeedingConfig = {}  #: configuration for seeding jobs
    server: t.Optional[gws.server.types.Config] = {}  #: server engine options
    timeZone: t.Optional[str] = 'UTC'  #: timezone for this server
    web: t.Optional[gws.web.types.Config] = {}  #: webserver configuration


class Object(gws.Object):
    version = gws.VERSION
    qgis_version = ''
    client = None

    def configure(self):
        super().configure()

        self.defaults = self.var('defaults')

        self.uid = 'APP'

        self.qgis_version = gws.qgis.server.version()

        gws.log.info(f'GWS version {self.version}, QGis {self.qgis_version}')

        #_install_fonts(self.var('fonts'))

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

    def find_action(self, action_type, project_uid=None):

        action = None

        if project_uid:
            project = self.find('gws.common.project', project_uid)
            if project:
                action = project.api.actions.get(action_type) if project.api else None

        if not action:
            action = self.api.actions.get(action_type) if self.api else None

        gws.log.debug(f'find_action {action_type!r} prj={project_uid!r} found={action.uid if action else 0}')
        return action


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
