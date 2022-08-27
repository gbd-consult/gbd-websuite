"""Core application object"""

import gws
import gws.base.action
import gws.base.auth
import gws.base.auth.manager
import gws.base.client
import gws.base.db
import gws.base.metadata
import gws.base.project
import gws.base.web
import gws.config
import gws.gis.cache
import gws.gis.mpx.config
import gws.lib.font
import gws.server
import gws.server.monitor
import gws.spec
import gws.base.web.error
import gws.types as t


class FontConfig(gws.Config):
    """Fonts configuration."""

    dir: gws.DirPath  #: directory with custom fonts


class Config(gws.ConfigWithAccess):
    """Main application configuration"""

    api: t.Optional[gws.base.action.collection.Config]  #: system-wide server actions
    auth: t.Optional[gws.base.auth.Config] = {}  # type: ignore #: authorization methods and options
    cache: t.Optional[gws.gis.cache.Config] = {}  # type: ignore #: global cache configuration
    client: t.Optional[gws.base.client.Config]  #: gws client configuration
    db: t.Optional[gws.base.db.collection.Config]  #: database configuration
    developer: t.Optional[t.Dict]  #: developer options
    fonts: t.Optional[FontConfig]  #: fonts configuration
    helpers: t.Optional[t.List[gws.ext.config.helper]]  #: helpers configurations
    locales: t.Optional[t.List[str]]  #: default locales for all projects
    metadata: t.Optional[gws.base.metadata.Config]  # application metadata
    projectDirs: t.Optional[t.List[gws.DirPath]]  #: directories with additional projects
    projectPaths: t.Optional[t.List[gws.FilePath]]  #: additional project paths
    projects: t.Optional[t.List[gws.ext.config.project]]  #: project configurations
    server: t.Optional[gws.server.Config] = {}  # type: ignore #: server engine options
    web: t.Optional[gws.base.web.Config]  #: web server options


class Object(gws.Node, gws.IApplication):
    """Main Appilication object"""

    cActions: gws.base.action.collection.Object
    cDatabases: gws.base.db.collection.Object
    helpers: t.List[gws.Node]
    qgisVersion = ''
    projects: t.Dict[str, gws.IProject]

    _devopts: dict

    def configure(self):
        self.version = self.root.specs.version

        self._devopts = self.var('developer') or {}
        if self._devopts:
            gws.log.warn('developer mode enabled')

        if self.var('server.qgis.enabled'):
            qgis_server = gws.import_from_path('gws/plugin/qgis/server.py')
            self.qgisVersion = qgis_server.version()

        s = f'GWS version {self.version}'
        if self.qgisVersion:
            s += f', QGis {self.qgisVersion}'

        gws.log.info('*' * 60)
        gws.log.info(s)
        gws.log.info('*' * 60)

        self.localeUids = self.var('locales') or ['en_CA']
        self.monitor = self.create_child(gws.server.monitor.Object, self.var('server.monitor'))
        self.metadata = gws.base.metadata.from_config(self.var('metadata'))

        p = self.var('fonts.dir')
        if p:
            gws.lib.font.install_fonts(p)

        # NB the order of initialization is important
        # - db
        # - helpers
        # - auth providers
        # - actions, client, web
        # - finally, projects

        self.cDatabases = self.create_child(gws.base.db.collection.Object, self.var('db'))

        # # helpers are always created, no matter configured or not
        # cnf = {c.get('type'): c for c in self.var('helpers', default=[])}
        # for class_name in self.root.specs.real_class_names('gws.ext.helper'):
        #     desc = self.root.specs.object_descriptor(class_name)
        #     if desc.ext_type not in cnf:
        #         gws.log.debug(f'ad-hoc helper {desc.ext_type!r} will be created')
        #         cfg = gws.Config(type=desc.ext_type)
        #         cnf[desc.ext_type] = gws.config.parse(self.root.specs, cfg, 'gws.ext.config.helper')
        # self.helpers = self.root.create_many('gws.ext.helper', list(cnf.values()))

        self.auth = self.create_child(gws.base.auth.manager.Object, self.var('auth'), required=True)

        # @TODO default API
        self.cActions = self.create_child(gws.base.action.collection.Object, self.var('api'), required=True)

        cfg = self.var('web.sites')
        if not cfg:
            cfg = [gws.Config(host='*', root=gws.base.web.DocumentRootConfig(dir='/data/web'))]
        if self.var('web.ssl'):
            cfg = [gws.merge(c, ssl=True) for c in cfg]
        self.webSites = self.create_children(gws.base.web.site.Object, cfg)

        self.client = self.create_child(gws.base.client.Object, self.var('client'))

        projects = self.create_children(gws.ext.object.project, self.var('projects'))
        self.projects = {p.uid: p for p in projects}

    def post_configure(self):
        self.mpx_url = ''
        if self.var('server.mapproxy.enabled'):
            gws.gis.mpx.config.create_and_save(self.root)
            self.mpx_url = f"http://{self.var('server.mapproxy.host')}:{self.var('server.mapproxy.port')}"

        # for p in set(cfg.configPaths):
        #     root.app.monitor.add_path(p)
        # for p in set(cfg.projectPaths):
        #     root.app.monitor.add_path(p)
        # for d in set(cfg.projectDirs):
        #     root.app.monitor.add_directory(d, parser.config_path_pattern)
        #
        # if root.app.developer_option('server.auto_reload'):
        #     root.app.monitor.add_directory(gws.APP_DIR, '\.py$')

    def find_project(self, uid):
        return self.projects.get(uid)

    def command_descriptor(self, command_category, command_name, params, user, strict_mode):
        """Obtain and populate a command descriptor for a request."""

        desc = self.root.specs.command_descriptor(command_category, command_name)
        if not desc:
            gws.log.error(f'command not found {command_category!r}:{command_name!r}')
            raise gws.base.web.error.NotFound()

        try:
            desc.request = self.root.specs.read(params, desc.tArg, strict_mode=strict_mode)
        except gws.spec.ReadError as exc:
            gws.log.exception()
            raise gws.base.web.error.BadRequest()

        project = None
        project_uid = desc.request.get('projectUid')

        if project_uid:
            project = self.find_project(project_uid)
            if not project:
                gws.log.error(f'project not found {command_category!r}:{command_name!r} {project_uid!r}')
                raise gws.base.web.error.NotFound()

        if project:
            cas = t.cast(gws.base.action.collection.Object, getattr(project, 'cActions', None))
            if cas:
                action = cas.find(desc.tOwner)
                if action:
                    if not user.can_use(action):
                        gws.log.error(f'action forbidden {command_category!r}:{command_name!r} project={project_uid!r}')
                        raise gws.base.web.error.Forbidden()
                    desc.methodPtr = getattr(action, desc.methodName)
                    return desc

        action = self.cActions.find(desc.tOwner)
        if action:
            if not user.can_use(action):
                gws.log.error(f'action forbidden {command_category!r}:{command_name!r}')
                raise gws.base.web.error.Forbidden()
            desc.methodPtr = getattr(action, desc.methodName)
            return desc

        gws.log.error(f'action not found {command_category!r}:{command_name!r}')
        raise gws.base.web.error.NotFound()

    def actions_for(self, user, project=None):
        d = {}
        for a in self.cActions.actions:
            if user.can_use(a):
                d[a.extType] = a
        if project:
            cas = t.cast(gws.base.action.collection.Object, getattr(project, 'cActions', None))
            if cas:
                for a in cas.actions:
                    if user.can_use(a):
                        d[a.extType] = a
        return list(d.values())

    def require_helper(self, ext_type):
        for obj in self.helpers:
            if obj.ext_type == ext_type:
                return obj
        raise gws.Error(f'helper {ext_type!r} not found')

    def developer_option(self, name):
        return self._devopts.get(name)
